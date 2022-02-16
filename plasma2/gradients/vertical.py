# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import linspace, Tensor

def top_bottom_gradient (input: Tensor, length: Tensor):
    """
    Create a vertical gradient which starts from the top of the given image.
    
    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(0., 1., height).to(input.device)
    field = field.repeat(samples, 1, width, 1).permute(0, 1, 3, 2).contiguous()
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field

def bottom_top_gradient (input: Tensor, length: Tensor) -> Tensor:
    """
    Create a vertical gradient which starts from the bottom of the given image.

    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(1., 0., height).to(input.device)
    field = field.repeat(samples, 1, width, 1).permute(0, 1, 3, 2).contiguous()
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field