# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import ones, Tensor
from torch.nn.functional import conv2d, pad

def box_filter (input: Tensor, radius: int) -> Tensor: # TEST
    """
    Apply a box filter to a 2D image.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W).
        radius (int): Filter window radius.

    Returns:
        Tensor: Filtered image with shape (N,C,H,W).
    """
    _,channels,_,_ = input.shape
    # Build kernels
    kernel_size = 2 * radius + 1
    kernel = ones(channels, 1, kernel_size, kernel_size).to(input.device) / (kernel_size ** 2)
    # Filter
    padded_input = pad(input, (radius, radius, radius, radius), mode="replicate")
    result = conv2d(padded_input, kernel, groups=channels)
    return result