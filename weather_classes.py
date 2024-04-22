import os
from typing import Optional, Callable, Sequence

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from nodes_classes import EulerIntegrator, NODEGradientModule, IntegratedNODE


# ==================================================================================================================== #
# DATA HANDLING
# ==================================================================================================================== #
class WeatherDataSet(Dataset):
    def __init__(
            self, csv_path, preprocess_path, n_prior_states, n_future_estimates, row_start: int = 0,
            row_end: int = -1, process_csv=True, normalize: bool = True, columns_to_use: Optional[Sequence] = None,
            dtype=float
    ):
        """
        Create a data set to load sequences of input and target data
        :param csv_path: path to CSV file used
        :param preprocess_path: path to save preprocessed array
        :param n_prior_states: Number of derivatives/prior states used in the estimate
                              (size of input vector is 1 + n_states * (n_prior_states + 1))
        :param n_future_estimates: Number of future states to estimate, separated by 1 hour intervals
                              (size of output vector is n_states * n_future_estimates)
        :param row_start: Initial row of CSV to load (allows separation of training/validation data)
        :param row_end: Final row of CSV to load
        :param normalize: Whether to normalize data
        :param columns_to_use: Optional[Sequence], which columns of csv to use as time/state.
        :param dtype: data type for output
        """
        self.csv_path = csv_path
        self.preprocess_path = preprocess_path
        self.row_start: int = row_start

        if row_end < 0:
            # Convert to positive index by counting rows in file
            with open(csv_path) as file:
                self.row_end = sum(1 for _ in file) - row_end - 1
        else:
            self.row_end: int = row_end

        self.n_prior_states = n_prior_states
        self.n_future_estimates = n_future_estimates
        self.normalize = normalize

        if columns_to_use is not None:
            self.columns_to_use = columns_to_use
        else:
            self.columns_to_use = [
                'Hour', 'Precip_mm', 'Atm_pressure_mb', 'Global Radiation (Kj/m²)', 'Air_temp_C',
                'Rel_Humidity_percent', 'Wind_dir_deg', 'Gust'
            ]

        self.dtype = dtype

        self.n_states = len(self.columns_to_use) - 1  # Assume one column is time
        self.n_states_to_return = self.n_prior_states + self.n_future_estimates + 1

        if process_csv:
            self.seq_starts, self.seq_ends, self.too_high_start_index, self.time_state_tensor = self.process_csv()
        else:
            self.seq_starts, self.seq_ends, self.too_high_start_index, self.time_state_tensor = None, None, None, None

        self.target_index_offset = self.n_prior_states + 1

    def process_csv(self):
        df = pd.read_csv(self.csv_path, skiprows=self.row_start, nrows=1 + self.row_end - self.row_start)

        # Derive states from df
        time_state_arr = df[self.columns_to_use].values

        if self.normalize:
            x_max = np.max(time_state_arr, axis=0, keepdims=True)
            x_min = np.min(time_state_arr, axis=0, keepdims=True)
            half_range = 0.5 * (x_max - x_min)
            midpoint = 0.5 * (x_max + x_min)
            half_range[0, 0] = 1.  # Do not transform time
            midpoint[0, 0] = 0.
            time_state_arr -= midpoint  # Put midpoint at 0
            time_state_arr /= half_range  # Transform range to [-1, +1]

        time_difference = np.diff(df['Index'])
        seq_starts = np.insert(np.nonzero(time_difference > 1)[0] + 1, 0, 0)
        seq_ends = np.append(seq_starts[1:], len(time_difference))

        len_seq = seq_ends - seq_starts
        too_low_length = self.n_states_to_return - 1
        seq_starts = seq_starts[len_seq > too_low_length]
        seq_ends = seq_ends[len_seq > too_low_length]
        too_high_start_index = seq_ends - too_low_length

        # Save pre-processed time/state array
        os.makedirs(os.path.dirname(self.preprocess_path), exist_ok=True)  # Make directory if it does not yet exist
        pd.DataFrame(time_state_arr).to_csv(self.preprocess_path)
        # np.savetxt(self.preprocess_path, time_state_arr, delimiter=',')

        return seq_starts, seq_ends, too_high_start_index, torch.as_tensor(time_state_arr, dtype=self.dtype).T

    def truncated_data(self, start: int = 0, end: int = -1):
        return self.seq_starts[start:end], self.seq_ends[start:end], self.too_high_start_index[start:end]

    def split_data(self, frac_validation: float = 0., frac_test: float = 0.):
        """Turn processed dataset into train/validation/test dataset (leftover used for train)"""
        n_data = len(self)

        n_data_validation = int(round(frac_validation * n_data))
        n_data_test = int(round(frac_test * n_data))
        n_data_train = n_data - n_data_validation - n_data_test

        idces_start = (0, n_data_train, n_data_train + n_data_validation)
        idces_end = (n_data_train, n_data_train + n_data_validation + n_data_test)

        # Assign same data array to new datasets, but truncate to subset of start/end indices
        datasets = []
        for idx_start, idx_end in zip(idces_start, idces_end):
            if idx_start == idx_end:
                continue  # Null dataset

            dataset = WeatherDataSet(
                csv_path=self.csv_path, preprocess_path=self.preprocess_path,
                n_prior_states=self.n_prior_states, n_future_estimates=self.n_future_estimates,
                row_start=self.row_start, row_end=self.row_end, process_csv=False, columns_to_use=self.columns_to_use
            )
            dataset.seq_starts, dataset.seq_ends, dataset.too_high_start_index = \
                self.truncated_data(idx_start, idx_end)
            dataset.time_state_tensor = self.time_state_tensor
            datasets.append(dataset)

        return datasets

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, index):
        # Choose random subsequence of the given sequence
        initial_index = np.random.randint(self.seq_starts[index], self.too_high_start_index[index])

        # Parse states
        data_tensor = self.time_state_tensor[:, initial_index:initial_index + self.n_states_to_return]
        # data_tensor = torch.as_tensor(pd.read_csv(
        #     self.preprocess_path, skiprows=initial_index, nrows=self.n_states_to_return
        # ).values).T
        input_time_states = data_tensor[:, :self.target_index_offset]
        output_states = data_tensor[1:, self.target_index_offset:]
        return input_time_states, output_states

    def __repr__(self):
        return f'WeatherDataSet(len = {len(self)} | {self.n_prior_states + 1} input / {self.n_future_estimates} output)'


# ==================================================================================================================== #
# LOSS FUNCTION
# ==================================================================================================================== #
def reduce_loss(loss, reduction):
    """
    Generic loss reduction
    :param loss: (B,)/() Tensor, non-reducted (possible batched with dimension B) loss
    :param reduction: str, default='mean', Reduction method. 'none' -> (B,)/() Tensor with MSE,
                      'mean' -> average of batches, 'sum' -> sum of batches
    :return: reduced loss
    """
    if 'mean' in reduction.lower():
        return loss.mean()
    elif 'sum' in reduction.lower():
        return loss.sum()
    elif 'none' in reduction.lower():
        return loss.squeeze()
    else:
        return NotImplementedError(f'Reduction method "{reduction}" is not implemented!')


def weather_matrix_loss_function(output: Tensor, target: Tensor, weighting_matrix: Tensor, reduction: str = 'mean'):
    """
    The Loss function is the weighted MSE e^T Q e where e is the difference between the output and target. Q must be a
    positive definite matrix in order for the loss to be convex. It is assumed
    :param output: (B,n)/(n,) Tensor
    :param target: (B,n)/(n,) Tensor
    :param weighting_matrix: (n,n) Tensor
    :param reduction: str, default='mean', Reduction method. 'none' -> (B,)/() Tensor with MSE,
                      'mean' -> average of batches, 'sum' -> sum of batches
    :return: Tensor, reduced weighted MSE of output vs. target
    """
    error = output - target
    return reduce_loss(error[..., None, :] @ weighting_matrix @ error[..., :, None], reduction)


def weather_weighted_loss_function(output: Tensor, target: Tensor, weights: Tensor, reduction: str = 'mean'):
    return reduce_loss((weights * (output - target)**2).sum(dim=-1), reduction)


def create_matrix_from_weights(
        weights, num_future_predictions, discount_for_future: float = 1., return_matrix=True
):
    """
    Converts a set of state weights to the positive definite weighting matrix.
    :param weights: Tensor, (n,) tensor of weights for n states
    :param num_future_predictions: int, number of future predictions H used, resulting in a (n*H, n*H) weighting matrix
    :param discount_for_future: float, a multiplier to weight future states (<1 -> reduces impact,
                                >1 -> increases impact on loss)
    :param return_matrix: True -> return (n*H,n*H) diagonal matrix instead of (n*H,) vector
    :return: weighting matrix to use in ``weather_matrix_loss_function''
    """
    n_states = weights.numel()
    weighting_vector = torch.empty(size=(n_states * num_future_predictions,)).to(weights)
    for idx in range(num_future_predictions):
        weighting_vector[idx*n_states:(idx+1)*n_states] = weights * discount_for_future**idx

    if return_matrix:
        return torch.diag(weighting_vector)
    else:
        return weighting_vector


# ==================================================================================================================== #
# Neural Network Modules
# ==================================================================================================================== #
def generate_dense_layers(
        in_features: int, out_features: int, n_layers: int = 1, activation_type: str = 'ReLU', dtype=float
):
    # Choose activation function based on type
    if activation_type.lower() == 'relu':
        def gen_activation_fn():
            return nn.ReLU(inplace=True)
    elif activation_type.lower() == 'elu':
        def gen_activation_fn():
            return nn.ELU(inplace=True)
    elif activation_type.lower() == 'silu':
        def gen_activation_fn():
            return nn.SiLU(inplace=True)
    elif "leaky" in activation_type.lower():
        def gen_activation_fn():
            return nn.LeakyReLU(inplace=True)
    else:
        raise NotImplementedError(f'Activation type "{activation_type}" is not implemented!')

    # Time is broken into a cosine and sine wave with a period of 1 day, so there are 2 "time" inputs in addition
    # to the number of features in/out
    layer = nn.Sequential()

    # Create the layers
    for idx_layer in range(1, n_layers + 1, 1):
        _out_features = out_features if idx_layer == n_layers else in_features
        layer.append(nn.Sequential(
            nn.Linear(in_features, _out_features, dtype=dtype),
            gen_activation_fn()
        ))

    return layer


class WeatherNODE(NODEGradientModule):
    """
    An implementation of NODEGradientModule whose Neural Network dynamics have the following architecture:
    (1) Dense Layer
    (2) Activation (default is ReLU)
    """
    def __init__(self, n_features: int, n_layers: int = 1, activation_type: str = 'ReLU', dtype=float):
        super(WeatherNODE, self).__init__()

        self.n_features: int = n_features
        self.n_layers: int = n_layers
        self.activation_type: str = activation_type

        # dx/dt = f(t, x), where t is integration time and x is the feature vector
        self.dynamic_layer = generate_dense_layers(
            in_features=1 + self.n_features, out_features=self.n_features, n_layers=self.n_layers,
            activation_type=self.activation_type, dtype=dtype
        )

    @staticmethod
    def cat_state_with_time(time: Tensor, state: Tensor):
        """
        Returns a concatenation [t; x]
        """
        return torch.cat((time.expand(size=state[..., 0:1].shape), state), dim=-1)

    def forward(self, integration_time, trig_time_and_state):
        """
        Dynamics function of the NODE dx/dt = f(t, x)

        :param integration_time: Integration time (unused), independent of true time for each element of batch
        :param trig_time_and_state: Concatenation of cos(time), sin(time), and state for (possibly) batched input
        :return:
        """
        dxdt = self.dynamic_layer(self.cat_state_with_time(integration_time, trig_time_and_state))
        return dxdt


class WeatherPredictor(nn.Module):
    """
    Predictor for weather data. The input data are a set of states at the current time and a number of previous times.
    The output is a number of future state predictions. Each layer is:
    """
    def __init__(
            self, n_states: int, n_prior_states: int, n_predictions: int,
            n_layers: int, activation_type: str = 'ReLU',
            use_node=True, integrator: Optional[Callable] = None, dtype = float
    ):
        super(WeatherPredictor, self).__init__()

        # Unpack inputs
        self.n_states = n_states
        self.n_prior_states = n_prior_states
        self.n_predictions = n_predictions
        self.n_layers = n_layers
        self.activation_type = activation_type
        self.use_node = use_node

        if integrator is not None:
            self.integrator = integrator
        else:
            self.integrator = EulerIntegrator()

        self.n_features = self.n_states * (1 + self.n_prior_states)
        self.day_frequency = 2*torch.pi / 24.

        if use_node:
            self.integration_layer = IntegratedNODE(
                WeatherNODE(n_features=2 + self.n_features, n_layers=self.n_layers,
                            activation_type=self.activation_type, dtype=dtype),
                integrator=integrator
            )
            self.unpack_inputs = self.unpack_node_inputs
            self.unpack_outputs = self.unpack_node_outputs
        else:
            self.integration_layer = generate_dense_layers(
                in_features=2 + self.n_features, out_features=self.n_features, n_layers=self.n_layers,
                activation_type=self.activation_type, dtype=dtype
            )
            self.unpack_inputs = self.unpack_discrete_inputs
            self.unpack_outputs = self.unpack_discrete_outputs

        # Storage of training information
        self.train_losses = []
        self.test_losses = []
        self.weights = []

    def unpack_node_inputs(self, _inputs: Tensor):
        """
        Convert the inputs to features used by NODE.
        :param _inputs: Tensor, inputs of dimension ([num_batches, ]1 + num_states, 1 + num_previous_states).
                        For the NODE, the prior states are converted into derivatives.
        :return _features: Tensor, features of dimension ([num_batches, ]1 + num_states*(1 + num_previous_states),)
        """
        _t0 = _inputs[..., 0:1, 0]

        _transformed_state = torch.empty_like(_inputs, device=_inputs.device)[..., 1:, :]
        _transformed_state[..., :, 0] = _inputs[..., 1:, 0]  # Assign state

        # Assign derivatives of state
        _diff_time_state = _inputs
        for _idx_assign in range(1, self.n_prior_states*self.n_states, self.n_states):
            _diff_time_state = _diff_time_state.diff(dim=-1)
            _transformed_state[..., :, _idx_assign] = \
                _diff_time_state[..., 1:, 0] / _diff_time_state[..., 0:1, 0]

        # Flatten and return
        return torch.cat((
            torch.cos(self.day_frequency * _t0), torch.sin(self.day_frequency * _t0),
            _transformed_state.flatten(start_dim=-2)
        ), dim=-1)

    def unpack_discrete_inputs(self, _inputs: Tensor):
        """
        Convert the inputs to features used by discrete NN
        :param _inputs: Tensor, inputs of dimension ([num_batches, ]1 + num_states, 1 + num_previous_states).
        :return: _features: Tensor, features of dimension ([num_batches, ]2 + num_states*(1 + num_previous_states),)
        """
        _t0 = _inputs[..., 0:1, 0]

        # Time is broken into a cosine and sine wave with a period of 1 day, so there are 2 "time" inputs in addition
        # to the number of features in/out.
        return torch.cat((
            torch.cos(self.day_frequency * _t0), torch.sin(self.day_frequency * _t0),
            _inputs[..., 1:, :].flatten(start_dim=-2)
        ), dim=-1)

    def unpack_node_outputs(self, _time, _output):
        """
        Convert output (which is state with its derivatives) to next time and state with derivatives
        :param _time: Tensor, current time
        :param _output: Output of integration (i.e. state with derivatives at _time)
        :return: tuple, output prediction and input into Integrator for next timestep
        """
        _output[..., 0] = torch.cos(self.day_frequency * _time)  # Advance to new time
        _output[..., 1] = torch.sin(self.day_frequency * _time)
        return _output[..., 2:2+self.n_states], _output  # Return state prediction and output vector

    def unpack_discrete_outputs(self, _time, _output):
        """
        Convert output (which is state with its derivatives) to next time and state with derivatives
        :param _time: Tensor, current time
        :param _output: Output of integration (i.e. state with its previous states at _time)
        :return: tuple, output prediction and input into Integrator for next timestep
        """
        _time = _time.view((-1, *torch.ones(size=(_output.ndim-1,), dtype=torch.int)))
        _new_features = torch.cat((_time, _time, _output), dim=-1)

        # Time is broken into a cosine and sine wave with a period of 1 day, so there are 2 "time" inputs in addition
        # to the number of features in/out.
        _new_features[..., 0] = torch.cos(self.day_frequency * _new_features[..., 0])
        _new_features[..., 1] = torch.sin(self.day_frequency * _new_features[..., 1])

        return _output[..., :self.n_states], _new_features

    def forward(self, _inputs):
        """Predict states given the inputs for 1, ..., self.n_predictions hours into the future"""
        _time = _inputs[..., 0, 0].clone()
        _in_features = self.unpack_inputs(_inputs)

        # Integrate to get predictions
        _prediction_shape = list(_inputs.shape[:-1])
        _prediction_shape[-1] -= 1  # Remove time from prediction
        _predicted_states = torch.empty(size=(*_prediction_shape, self.n_predictions))
        for idx_predict in range(self.n_predictions):
            _output = self.integration_layer(_in_features)  # Integrate the state
            _time += 1.  # Advance 1 hour
            _predicted_states[..., :, idx_predict], _in_features = self.unpack_outputs(_time, _output)
        return _predicted_states


def train(model, train_loader, optimizer, criterion, device, verbose=True):
    """
    Function to train the model, using the optimizer to tune the parameters.
    """
    model.train()  # Change model to training mode

    running_loss = 0.0
    n_batches = len(train_loader)
    idx_report = max(1, int(n_batches / 5))
    if verbose:
        print(f'\n')

    for idx, (inputs, labels) in enumerate(train_loader):
        if verbose and idx % idx_report == 0:
            print(f'Batch #{idx+1}/{n_batches} (Ave. Loss = {running_loss / (idx+1):.4f})...')
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.flatten(start_dim=-2), labels.flatten(start_dim=-2))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / n_batches


def test(model, test_loader, criterion, device):
    """
    Function to evaluate the model using the testing dataset
    """
    model.eval()  # Change model to evaluation mode

    n_batches = len(test_loader)
    total_loss = 0.
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs.flatten(start_dim=-2), targets.flatten(start_dim=-2)).item()
    return total_loss / n_batches
