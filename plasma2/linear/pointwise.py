# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import cat, clamp, tensor, Tensor

from ..conversion import rgb_to_yuv, yuv_to_rgb

def contrast (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply contrast adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, channels, width, height = input.shape
    result = (weight + 1.) * input.flatten(start_dim=1)
    result = result.view(-1, channels, width, height)
    result = result.clamp(min=-1., max=1.)
    return result

def exposure (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply exposure adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, channels, width, height = input.shape
    input = (input + 1.) / 2.
    result = (weight + 1.) * input.flatten(start_dim=1)
    result = result.view(-1, channels, width, height)
    result =  2. * result - 1.
    result = result.clamp(min=-1., max=1.)
    return result

def saturation (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply saturation adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = (weight + 1.) * u.flatten(start_dim=1)
    v = (weight + 1.) * v.flatten(start_dim=1)
    u = u.view(-1, 1, height, width)
    v = v.view(-1, 1, height, width)
    y = y.expand_as(u)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def color_balance (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply color balance adjustment on an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor): Scalar temperature and tint weights with shape (N,2) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    temp, tint = weight.split(1, dim=1)
    u = 0.1 * (tint - temp) + u.flatten(start_dim=1)
    v = 0.1 * (tint + temp) + v.flatten(start_dim=1)
    u = u.view(-1, 1, height, width)
    v = v.view(-1, 1, height, width)
    y = y.expand_as(u)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result