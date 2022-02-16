# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, clamp, tensor, Tensor

from ..conversion import rgb_to_yuv, yuv_to_rgb

def contrast (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply contrast adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, channels, width, height = input.shape
    result = input.flatten(start_dim=1) * (weight + 1.)
    result = result.view(-1, channels, width, height).clamp(min=-1., max=1.)
    return result

def saturation (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply saturation adjustment to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = u.flatten(start_dim=1) * (weight + 1.)
    v = v.flatten(start_dim=1) * (weight + 1.)
    u = u.view(-1, 1, height, width)
    v = v.view(-1, 1, height, width)
    y = y.expand_as(u)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def temperature (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply temperature adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = u.flatten(start_dim=1) - 0.1 * weight
    v = v.flatten(start_dim=1) + 0.1 * weight
    u = u.view(-1, 1, height, width)
    v = v.view(-1, 1, height, width)
    y = y.expand_as(u)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def tint (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply tint adjustment to an image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    yuv = rgb_to_yuv(input)
    y, u, v = yuv.split(1, dim=1)
    u = u.flatten(start_dim=1) + 0.1 * weight
    v = v.flatten(start_dim=1) + 0.1 * weight
    u = u.view(-1, 1, height, width)
    v = v.view(-1, 1, height, width)
    y = y.expand_as(u)
    yuv = cat([y, u, v], dim=1)
    result = yuv_to_rgb(yuv)
    return result