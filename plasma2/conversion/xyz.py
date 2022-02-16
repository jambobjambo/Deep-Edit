# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import tensor, Tensor

def rgb_to_xyz (input: Tensor) -> Tensor:
    """
    Convert linear RGB to D65 XYZ.
    
    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [-1., 1.].
    
    Returns:
        Tensor: XYZ image with shape (N,3,...) in range [0., 1.].
    """
    input = (input + 1.) / 2.
    RGB_TO_XYZ = tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ]).float().to(input.device)
    xyz = RGB_TO_XYZ.matmul(input.flatten(start_dim=2)).view_as(input)
    return xyz

def xyz_to_rgb (input: Tensor) -> Tensor:
    """
    Convert D65 XYZ to linear RGB.
    
    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [0., 1.].
    
    Returns:
        Tensor: RGB image with shape (N,3,...) in range [-1., 1.].
    """
    XYZ_TO_RGB = tensor([
        [3.240479, -1.53715, -0.498535],
        [-0.969256, 1.875991, 0.041556],
        [0.055648, -0.204043, 1.057311]
    ]).float().to(input.device)
    rgb = XYZ_TO_RGB.matmul(input.flatten(start_dim=2)).view_as(input)
    rgb = 2. * rgb - 1.
    rgb = rgb.clamp(min=-1., max=1.)
    return rgb