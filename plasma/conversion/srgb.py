# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import where, Tensor

def srgb_to_linear (input: Tensor) -> Tensor:
    """
    Convert sRGB to linear RGB.

    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [-1., 1.].
    
    Returns:
        Tensor: Linear RGB image with shape (N,3,...) in range [-1., 1.].
    """
    input = (input + 1.) / 2.
    input = input.clamp(min=1e-4)
    linear = where(input > 0.0404482362771082, ((input + 0.055) / 1.055).pow(2.4), input / 12.92)
    linear = 2. * linear - 1.
    return linear

def linear_to_srgb (input: Tensor) -> Tensor:
    """
    Convert linear RGB to sRGB.

    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [-1., 1.].
    
    Returns:
        Tensor: sRGB image with shape (N,3,...) in range [-1., 1.].
    """
    input = (input + 1.) / 2.
    input = input.clamp(min=1e-4)
    srgb = where(input > 0.00313066844250063, 1.055 * input.pow(1. / 2.4) - 0.055, input * 12.92)
    srgb = 2. * srgb - 1.
    return srgb