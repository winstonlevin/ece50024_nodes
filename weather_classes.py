import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# ==================================================================================================================== #
# DATA HANDLING
# ==================================================================================================================== #
class WeatherDataSet(Dataset):
    def __init__(self, csv_path, n_prior_states, n_future_estimates, row_start: int = 0, row_end: int = -1):
        """
        Create a data set to load sequences of input and target data
        :param csv_path: path to CSV file used
        :param n_prior_states: Number of derivatives/prior states used in the estimate
                              (size of input vector is 1 + n_states * (n_prior_states + 1))
        :param n_future_estimates: Number of future states to estimate, separated by 1 hour intervals
                              (size of output vector is n_states * n_future_estimates)
        :param row_start: Initial row of CSV to load (allows separation of training/validation data)
        :param row_end: Final row of CSV to load
        """
        self.csv_path = csv_path
        self.row_start: int = row_start

        if row_end < 0:
            # Convert to positive index by counting rows in file
            with open(csv_path) as file:
                self.row_end = sum(1 for _ in file) - row_end - 1
        else:
            self.row_end: int = row_end

        self.n_prior_states = n_prior_states
        self.n_future_estimates = n_future_estimates

        self.time_state_df, self.seq_starts, self.seq_ends, self.too_high_start_index = self.process_csv()
        self.n_states = self.time_state_df.shape[1] - 1
        self.target_index_offset = self.n_prior_states + 1
        self.length_inputs = 1 + self.n_states * self.n_prior_states
        self.length_targets = self.n_states * self.n_future_estimates

    def process_csv(self):
        df = pd.read_csv(self.csv_path, skiprows=self.row_start, nrows=1 + self.row_end - self.row_start)
        df = df[:self.row_end]
        time_difference = np.diff(df['Index'])
        seq_starts = np.insert(np.nonzero(time_difference > 1)[0] + 1, 0, 0)
        seq_ends = np.append(seq_starts[1:], len(time_difference))

        len_seq = seq_ends - seq_starts
        too_low_length = self.n_prior_states + self.n_future_estimates
        seq_starts = seq_starts[len_seq > too_low_length]
        seq_ends = seq_ends[len_seq > too_low_length]
        too_high_start_index = seq_ends - too_low_length

        time_state_df = df[['Hour', 'Precip_mm', 'Atm_pressure_mb', 'Global Radiation (Kj/m²)', 'Air_temp_C',
                       'Rel_Humidity_percent', 'Wind_dir_deg', 'Gust']]

        return time_state_df, seq_starts, seq_ends, too_high_start_index

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, index):
        initial_index = np.random.randint(0, self.too_high_start_index[index])
        df = self.time_state_df[self.seq_starts[index] + initial_index:self.seq_ends[index]]
        data_tensor = torch.as_tensor(np.asarray(df, dtype=float).T)

        input_time_states = torch.cat((data_tensor[0:1, 0], data_tensor[1:, :self.target_index_offset]))
        output_states = data_tensor[1:, self.target_index_offset:]

        return input_time_states, output_states


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

