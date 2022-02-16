# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import cat, lerp, where, Tensor

from ..blending import blend_soft_light
from ..conversion import rgb_to_luminance, rgb_to_yuv, yuv_to_rgb
from ..filters import bilateral_filter, gaussian_filter, guided_filter

def clarity (input: Tensor, weight: Tensor) -> Tensor: # INCOMPLETE # Scale radius with pixel size
    """
    Apply coarse-scale local contrast to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    # Compute base layer
    y, u, v = rgb_to_yuv(input).split(1, dim=1)
    y = guided_filter(y, y, 11, 0.02)
    yuv = cat([y, u, v], dim=1)
    base_layer = yuv_to_rgb(yuv)
    # Interpolate
    base_colors = base_layer.flatten(start_dim=1).clamp(min=-1., max=1.)
    input_colors = input.flatten(start_dim=1)
    result_colors = base_colors.lerp(input_colors, weight + 1.)
    # Reshape
    result = result_colors.view(-1, 3, height, width)
    result = result.clamp(min=-1., max=1.)
    return result

def highlights (input: Tensor, weight: Tensor, tonal_range: float=1.) -> Tensor:
    """
    Apply highlight attentuation to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].
        tonal_range (float): Tonal range of the filter in range [0., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    # Compute mask
    mask = -rgb_to_luminance(input)
    mask = mask + (1. - tonal_range)
    mask = mask.clamp(min=-1., max=0.)
    # Blend
    mask = -weight * mask.flatten(start_dim=1)
    mask = mask.view(-1, 1, height, width)
    result = blend_soft_light(input, mask)
    return result

def shadows (input: Tensor, weight: Tensor, tonal_range: float=1.) -> Tensor:
    """
    Apply shadow attentuation to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].
        tonal_range (float): Tonal range of the filter in range [0., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    # Compute mask
    luma = -rgb_to_luminance(input)
    mask = bilateral_filter(luma, luma, kernel_size=(5, 11), grid_size=(16, 64, 64))
    mask = mask - (1. - tonal_range)
    mask = mask.clamp(min=0., max=1.)
    # Blend
    mask = weight * mask.flatten(start_dim=1)
    mask = mask.view(-1, 1, height, width)
    result = blend_soft_light(input, mask)
    return result

def sharpen (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply sharpness enhancement to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    # Compute base layer
    base_layer = gaussian_filter(input, kernel_size=(3, 3))
    # Interpolate
    base_colors = base_layer.flatten(start_dim=1)
    input_colors = input.flatten(start_dim=1)
    result_colors = base_colors.lerp(input_colors, weight + 1.)
    # Reshape
    result = result_colors.view(-1, 3, height, width)
    result = result.clamp(min=-1., max=1.)
    return result

def texture (input: Tensor, weight: Tensor) -> Tensor: # INCOMPLETE # Scale radius with pixel size
    """
    Apply fine-scale local contrast to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width = input.shape
    # Compute base layer
    luminance = rgb_to_luminance(input)
    base_layer = guided_filter(input, luminance, 7, 0.015)
    # Interpolate
    base_colors = base_layer.flatten(start_dim=1).clamp(min=-1., max=1.)
    input_colors = input.flatten(start_dim=1)
    result_colors = base_colors.lerp(input_colors, weight + 1.)
    # Reshape
    result = result_colors.view(-1, 3, height, width)
    result = result.clamp(min=-1., max=1.)
    return result