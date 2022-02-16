# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import linspace, Tensor

def left_right_gradient (input: Tensor, length: Tensor) -> Tensor:
    """
    Create a horizontal gradient which starts from the left of the given image.

    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(0., 1., width).to(input.device)
    field = field.repeat(samples, 1, height, 1)
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field

def right_left_gradient (input: Tensor, length: Tensor) -> Tensor:
    """
    Create a horizontal gradient which starts from the right of the given image.

    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(1., 0., width).to(input.device)
    field = field.repeat(samples, 1, height, 1)
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field