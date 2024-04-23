import sys
from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


class EulerIntegrator:
    def __init__(self, n_steps: int = 5):
        self.n_steps = n_steps

    def solve_ivp(self, fun: Callable, t_span: Tensor, y0: Tensor):
        """
        Solve IVP using Euler's method parameterized by the number of integration steps.

        :param fun: Dynamic function dy/dt = f(t, y)
        :param t_span:
        :param y0:
        :return:
        """
        dt = (t_span[1] - t_span[0]) / float(self.n_steps)
        t = torch.as_tensor(t_span[0], dtype=y0.dtype, device=y0.device).clone()
        y = y0.clone()

        for i_step in range(self.n_steps):
            y += dt * fun(t, y)
            t += dt
        return y

    def __call__(self, fun: Callable, t_span: Tensor, y0: Tensor):
        return self.solve_ivp(fun, t_span, y0)


class RKIntegrator:
    def __init__(self, tol: float = 1e-3, max_iter: float = 100):
        self.init_tol = tol
        self.max_iter = max_iter

    def solve_ivp(self, fun: Callable, t_span: Tensor, y0: Tensor):
        """
        Solve IVP using 4/5th order Runga-Kutta method with an adaptive step size parameterized by error tolerance.

        :param fun: Dynamic function dy/dt = f(t, y)
        :param t_span:
        :param y0:
        :return:
        """
        #Set initial step size
        dt = (t_span[1] - t_span[0]) / 20

        t = torch.as_tensor(t_span[0], dtype=y0.dtype, device=y0.device).clone()
        y = y0.clone()

        final_t = t_span[1]
        h = dt
        numiters = 0
        tol = self.init_tol

        #TODO set mex iterations
        #if taking too long, maybe increase tolerance
        while t < final_t:
            #ensure not to step past final time
            h = min(h, final_t - t)
            #do RK 4 and 5 steps
            rk4, rk5 = self.rk45_step(fun, t_span, y, h)
            error = torch.max(torch.norm(rk5 - rk4))
            if error < tol or h <= 0.001:
                y += rk5
                t += h
                tol = self.init_tol

            else:
                numiters += 1
            if numiters >= self.max_iter:
                tol *= 10
                numiters = 0
            #Need to limit the next step size to between (1/2 and 2)
            nextStepSize = 0.9 * (tol / error) ** (1 / 5)
            bestCase = min(2, nextStepSize)
            h *= max(0.01, bestCase)

            #implement min step size
            h = max(0.001, h)
        return y

    def rk45_step(self, fun: Callable, t_span: Tensor, y0: Tensor, dt: Tensor):
        """Tableau for Dormand-Prince method used (see https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method)"""
        t0 = t_span[0]
        # dt = t_span[1] - t_span[0]
        k1 = fun(t0, y0)
        k2 = fun(t0 + 0.2*dt, y0 + 0.2*k1*dt)
        k3 = fun(t0 + 0.3*dt, y0 + (0.075*k1 + 0.225*k2) * dt)
        k4 = fun(t0 + 0.8*dt, y0 + (44/45*k1 - 56/15*k2 + 32/9*k3) * dt)
        k5 = fun(t0 + 8/9*dt, y0 + (19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4) * dt)
        k6 = fun(t0 + dt, y0 + (9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5) * dt)
        k7 = fun(t0 + dt, y0 + (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6) * dt)

        y_4th_order = y0 + 35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6
        y_5th_order = y0 + 5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7

        return y_4th_order, y_5th_order

    def __call__(self, fun: Callable, t_span: Tensor, y0: Tensor):
        return self.solve_ivp(fun, t_span, y0)


class NODEGradientModule(nn.Module, ABC):
    """
    Abstract class implementing generic costate dynamics calculation.
    """
    @abstractmethod
    def forward(self, time, state):
        pass

    def forward_with_grad(self, time: Tensor, state: Tensor, grad_outputs: Tensor):
        """Compute state and adjoint dynamics (time derivative of cost w.r.t. time, state, and parameters"""

        batch_size = state.shape[0]

        dxdt = self.forward(time, state)  # n-D Tensor: batch_size x (non-batched state dimension)

        time_adjoint_dynamics, costate_dynamics, *parameter_adjoint_dynamics = torch.autograd.grad(
            (dxdt,), (time, state) + tuple(self.parameters()), grad_outputs=grad_outputs,
            allow_unused=True, retain_graph=True
        )

        # Combine all parameters into flattened vector
        parameter_adjoint_dynamics = torch.cat(
            [dlamtheta_dt.flatten() for dlamtheta_dt in parameter_adjoint_dynamics], dim=0
        )

        # Combine costate and adjoint dynamics into lambda (-dL/dt; -dL/dx; -dL/dtheta) vector
        dlamdt = -torch.cat((
            time_adjoint_dynamics.expand(batch_size, 1),
            costate_dynamics.view(batch_size, -1),
            parameter_adjoint_dynamics[None, :].expand(batch_size, -1)
        ), dim=1)  # 2-D Tensor: batch_size x (1 + n_features + n_parameters)
        return dxdt, dlamdt

    def flatten_parameters(self):
        """Flatten parameters into a single dimension"""
        parameters_1d = []
        for p in self.parameters():
            parameters_1d.append(p.flatten())
        return torch.cat(parameters_1d)


class IntegrateNODE(torch.autograd.Function):
    """
    Implementation of NODE integration using Algorithm 1 from NODE paper:
    https://doi.org/10.48550/arXiv.1806.07366

    Inherits from PyTorch automatic differentiation (AD) function so that the AD is properly interfaced when using
    PyTorch Neural Network modules.
    """
    # Signature typing is supposed to be variable, but PyCharm does not pick up on this, so:
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
            ctx,
            state: Tensor, t_span: Tensor, parameters_1d: Tensor,  # "Variables"
            dyn_fun: NODEGradientModule, integrator: Callable  # Helper functions
                ):
        """
        Implementation of the FORWARD pass (integrate state from initial to final time without gradient calculations).

        :param ctx: Object passed to the "backward" method with necessary variables saved
        :param state: Initial state (at t = t_span[0])
        :param t_span: 2-element Tensor containing initial and final time
        :param parameters_1d: flattened parameters
        :param dyn_fun: derivative function dx/dt = f(x, t)
        :param integrator: Function to integrate yf = F(dy/dt [Callable], t_span, y0)
        :return new_state: Final state (at t = t_span[-1])
        """
        with torch.no_grad():
            # Integrate forward without calculating gradient
            new_state = integrator(dyn_fun, t_span, state).clone()

        ctx.dyn_fun = dyn_fun
        ctx.integrator = integrator
        ctx.save_for_backward(  # Give to "ctx.saved_tensors" for backward call
            t_span.clone(),
            new_state.clone(),
            parameters_1d
        )
        return new_state

    # Signature typing is supposed to be variable, but PyCharm does not pick up on this, so:
    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, dcost_dstate):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        fun = ctx.dyn_fun
        integrator = ctx.integrator
        t_span, z, parameters_1d = ctx.saved_tensors  # See forward for where these are saved
        batch_size = z.shape[0]  # For unflattening states
        z_shape = z.shape[1:]
        n_features = z.shape[1:].numel()
        n_params = parameters_1d.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(_time, _augmented_state):
            """
            The dynamics of the states, costates
            t_i - is tensor with size: batch_size
            aug_z_i - is tensor with size: batch_size, n_features*2 + n_params + 1
            """
            _state, _costate = _augmented_state[:, :n_features], _augmented_state[:, n_features:2 * n_features]

            # Unflatten state / costate
            _state = _state.view(batch_size, *z_shape)
            _costate = _costate.view(batch_size, *z_shape)

            with torch.set_grad_enabled(True):
                _time = _time.detach().requires_grad_(True)
                _state = _state.detach().requires_grad_(True)

                # The forward gradient is used to calculate the dynamics of the loss w.r.t. the time, states, and
                # parameters. Even through we propagate backward, we use the "forward" dynamics because the integration
                # occurs backwards in time (from tf = 1 to t0 = 0). I.e.:
                # y0 = xy + int(dy/dt, tf -> t0) : dy/dt is FORWARD in time
                _dxdt, _dlam_dt = fun.forward_with_grad(_time, _state, grad_outputs=_costate)

            # Return flattened dynamics (lambda already flattened)
            _dxdt = _dxdt.view(batch_size, n_features)
            return torch.cat((_dxdt, _dlam_dt), dim=1)

        dcost_dstate = dcost_dstate.view(batch_size, n_features)  # flatten dLdz for convenience
        with torch.no_grad():
            # Propagate augmented dynamics WITHOUT gradient (b/c already calculated in adjoint dynamics)
            dxdt_f = fun(t_span[-1], z).view(batch_size, n_features)  # Flattened dynamics at final time
            adjoint_terminal_time = torch.bmm(
                torch.transpose(dcost_dstate.unsqueeze(-1), 1, 2), dxdt_f.unsqueeze(-1)
            )[:, 0]  # Time adjoint at final time

            adjoint_states = torch.zeros(
                batch_size, n_features + 1 + n_params, dtype=dcost_dstate.dtype, device=dcost_dstate.device
            )  # "Adjoint" vector is derivative of loss w.r.t. states, parameters, and time.
            adjoint_states[:, :n_features] = dcost_dstate  # Costates at final time
            adjoint_states[:, n_features:n_params] = adjoint_terminal_time
            augmented_state = torch.cat(
                (z.view(batch_size, n_features), adjoint_states), dim=-1
            )  # Combined states and adjoints for propagation
            augmented_state = integrator(augmented_dynamics, torch.flip(t_span, dims=(0,)), augmented_state)

            # Unpack cost w.r.t. states, time, and parameters
            costate = augmented_state[:, n_features:2*n_features].view(batch_size, *z_shape)
            adjoint_initial_time = augmented_state[:, 2*n_features:2*n_features+1]
            adjoint_params = augmented_state[:, 2*n_features+1:]
            adjoints_time = torch.cat(
                (adjoint_initial_time, adjoint_terminal_time), dim=1
            )

        # Return partial derivative of cost w.r.t. each extra input to the "forward" call. If the input is not a Tensor
        # (e.g. the "fun" input), return None. In our forward call, we have the extra inputs:
        # state, t_span, parameters_1d, fun
        # Therefore, we return:
        # dL/dx (costate), dL/dt (adjoints_time), dL/dtheta (adjoint_params), None (dyn_fun), None (integrator)
        return costate, adjoints_time, adjoint_params, None, None


class IntegratedNODE(nn.Module):
    """
    This module outputs the integrated Neural Ordinary Differential Equation in its forward pass. It uses the custom-
    defined ``IntegrateNODE'' module in order to perform the backwards/forward integration. The dynamics are defined by
    the custom-defined ``NODEGradientModule,'' which calculates the NN and cost dynamics.
    """
    def __init__(self, dyn_fun: NODEGradientModule, integrator: Callable):
        """
        :param dyn_fun: derivative function dx/dt = f(x, t)
        :param integrator: Function to integrate yf = F(dy/dt [Callable], t_span, y0)
        """
        super(IntegratedNODE, self).__init__()
        self.dyn_fun = dyn_fun
        self.integrator = integrator

    def forward(self, initial_state, t_span: Optional[Tensor] = None):
        if t_span is None:
            # Default time-span: 0 to 1
            t_span = torch.as_tensor((0., 1.), dtype=initial_state.dtype, device=initial_state.device)
        z = IntegrateNODE.apply(initial_state, t_span, self.dyn_fun.flatten_parameters(), self.dyn_fun, self.integrator)
        return z


class Conv2dNODE(NODEGradientModule):
    """
    An implementation of NODEGradientModule whose Neural Network dynamics have the following architecture:
    (1) Convolution: (x, t) -> h
    (2) ReLU activation of h: dx/dt = ReLU(h)
    """
    def __init__(self, n_features):
        super(Conv2dNODE, self).__init__()
        self.dynamic_layer = nn.Sequential(
            nn.Conv2d(
                n_features+1, n_features, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def cat_state_with_time(time: Tensor, state: Tensor):
        """
        Returns a concatenation [x; t]
        """
        batch_size, _, width, height = state.shape
        return torch.cat((state, time.expand(batch_size, 1, width, height)), dim=1)

    def forward(self, time, state):
        """
        Dynamics function of the NODE dx/dt = f(t, x)

        :param time:
        :param state:
        :return:
        """
        dxdt = self.dynamic_layer(self.cat_state_with_time(time, state))
        return dxdt


class MNISTClassifier(nn.Module):
    """
    Classifier for MNIST Data. The input data is a 28x28 greyscale image of a digit. The output data is a classification
    from 0 to 9. The NN architecture is:

    (1) INPUT LAYER
        (1a.) 2D convolution of each pixel to a feature vector
        (1b.) ReLU activation of features

    (2) HIDDEN LAYER
        Integration of Convolutional NODE for features (if use_node) or a conventional convolution layer

    (3) POOL LAYER
        Pool features of each pixel into single feature vector

    (3) OUTPUT LAYER
        Linear Transformation from feature vector to 10 possible classifications.
    """
    def __init__(self, n_features: int = 64, use_node=True, integrator: Optional[Callable] = None):
        if integrator is None:
            integrator = EulerIntegrator()

        super(MNISTClassifier, self).__init__()
        self.n_features = n_features
        self.use_node = use_node

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=3, padding=1),  # in_channel (1 for grayscale), out_channels
            nn.ReLU(inplace=True)  # inplace=True argument modifies the input tensor directly, saving memory.
        )
        if use_node:
            self.hidden_layer = IntegratedNODE(Conv2dNODE(n_features), integrator=integrator)
        else:
            self.hidden_layer = nn.Sequential(
                nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(n_features, 10)

        # Storage of training information
        self.train_losses = []
        self.test_accuracies = []

    def forward(self, _state):
        _state = self.input_layer(_state)  # Extract features from each pixel
        _state = self.hidden_layer(_state)  # Propagate NODE
        _state = self.pool_layer(_state)  # Pool features from each pixel
        _state = torch.flatten(_state, 1)  # Remove extra dimensions for linear transform
        _state = self.output_layer(_state)  # Convert pool of features to classifications
        return _state

class WeatherForecaster(nn.Module):
    """
    Forecaster for Weather Data. The input data is a set of weather data from a horizon t-H. The output data is a forecast
    of weather for t+H. The NN architecture is:

    (1) HIDDEN LAYER
        Integration of Convolutional NODE for features (if use_node) or a conventional convolution layer
    """
    def __init__(self, n_features: int = 64, use_node=True, integrator: Optional[Callable] = None, H: int = 2):
        if integrator is None:
            integrator = EulerIntegrator()

        super(MNISTClassifier, self).__init__()
        self.n_features = n_features
        self.use_node = use_node
        self.H = H

        if use_node:
            self.hidden_layer = IntegratedNODE(Conv2dNODE(n_features), integrator=integrator)
        else:
            self.hidden_layer = nn.Sequential(
                nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Storage of training information
        self.train_losses = []
        self.test_accuracies = []

    #TODO
    #Returns D+1 * 7 tensor of states
    #Dims will be weird becuase of batching
    #Will need to pass in sin and cos time terms (time variant)
    def forward(self, _state):
        state_list = _state
        H = self.H
        nextState  = _state
        for t in range(H + 1):
            if t == 0:
                nextState = self.hidden_layer(nextState)
                state_list = nextState
            else:
                nextState = self.hidden_layer(nextState)
                state_list = torch.cat((state_list, nextState), 0)
        return state_list


def train(model, train_loader, optimizer, criterion, device, verbose=True):
    """
    Function to train the model, using the optimizer to tune the parameters.
    """
    model.train()
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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / n_batches


def test(model, test_loader, device):
    """
    Function to evaluate the model using the testing dataset
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
