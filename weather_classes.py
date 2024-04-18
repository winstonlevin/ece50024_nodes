from torch import Tensor


def weather_loss_function(output: Tensor, target: Tensor, weighting_matrix: Tensor, reduction: str = 'mean'):
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
    _error = output - target
    _loss = _error[..., None, :] @ weighting_matrix @ _error[..., :, None]
    if 'mean' in reduction.lower():
        return _loss.mean()
    elif 'sum' in reduction.lower():
        return _loss.sum()
    elif 'none' in reduction.lower():
        return _loss.squeeze()
    else:
        return NotImplementedError(f'Reduction method "{reduction}" is not implemented!')
