import torch
from torch import Tensor


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

