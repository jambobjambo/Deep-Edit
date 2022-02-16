# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import tensor, Tensor

def rgb_to_yuv (input: Tensor) -> Tensor:
    """
    Convert RGB to YUV.

    The shape of the input tensor is preserved.

    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [-1., 1.].

    Returns:
        Tensor: YUV image with shape (N,3,...) in range [0., 1.]
    """
    input = (input + 1.) / 2.
    RGB_TO_YUV = tensor([
        [0.2126, 0.7152, 0.0722],
        [-0.09991, -0.33609, 0.436],
        [0.615, -0.55861, -0.05639]
    ]).float().to(input.device)
    yuv = RGB_TO_YUV.matmul(input.flatten(start_dim=2)).view_as(input)
    return yuv

def yuv_to_rgb (input: Tensor) -> Tensor:
    """
    Convert YUV to RGB.

    The shape of the input tensor is preserved.

    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [0., 1.].

    Returns:
        Tensor: RGB image with shape (N,3,...) in range [-1., 1.]
    """
    YUV_TO_RGB = tensor([
        [1., 0., 1.28033],
        [1., -0.21482, -0.38059],
        [1., 2.12798, 0.]
    ]).float().to(input.device)
    rgb = YUV_TO_RGB.matmul(input.flatten(start_dim=2)).view_as(input)
    rgb = (2.0 * rgb - 1.0).clamp(min=-1., max=1.)
    return rgb

def rgb_to_luminance (input: Tensor) -> Tensor:
    """
    Convert RGB to luminance.

    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [-1., 1.]

    Returns:
        Tensor: Luminance image with shape (N,1,...) in range [-1., 1.]
    """
    y, _, _ = rgb_to_yuv(input).split(1, dim=1)
    luminance = y * 2. - 1.
    return luminance