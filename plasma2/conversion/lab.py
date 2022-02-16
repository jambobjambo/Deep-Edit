# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import cat, diag, tensor, where, Tensor

from .xyz import xyz_to_rgb, rgb_to_xyz

def xyz_to_lab (input: Tensor):
    """
    Convert D65 XYZ pixels to Lab.
    
    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [0., 1.].
    
    Returns:
        Tensor: Lab image with shape (N,3,...), with 0 <= L <= 100, -127 <= a, b <= 127.
    """
    # Constants
    eps = 216. / 24389.
    k = 24389. / 27.
    d65 = tensor([0.95047, 1., 1.08883]).float().to(input.device)
    # Reference white
    d65_xyz_colors = diag(1. / d65).matmul(input.flatten(start_dim=2)).clamp(min=1e-4) # prevent NaN
    d65_xyz = d65_xyz_colors.view_as(input).clamp(min=1e-4)
    # Convert
    f_xyz = where(d65_xyz > eps, d65_xyz.pow(1. / 3.), (k * d65_xyz + 16.) / 116.)
    f_x, f_y, f_z = f_xyz.split(1, dim=1)
    l = 116. * f_y - 16.
    a = 500. * (f_x - f_y)
    b = 200. * (f_y - f_z)
    lab = cat([l, a, b], dim=1).view_as(input)
    return lab

def lab_to_xyz (input: Tensor):
    """
    Convert Lab to D65 XYZ.
    
    Parameters:
        input (Tensor): Input XYZ pixel array with shape (N,3,...).
    
    Returns:
        Tensor: XYZ image with shape (N,3,...) in range [0., 1.].
    """
    # Constants
    eps = 216. / 24389.
    eps_1_3 = eps ** (1. / 3.)
    k = 24389. / 27.
    d65 = tensor([0.95047, 1., 1.08883]).float().to(input.device)
    # Convert
    l, a, b = input.split(1, dim=1)
    f_y = (l + 16.) / 116.
    f_x = (a / 500.) + f_y
    f_z = f_y - (b / 200.)
    f_xyz = cat([f_x, f_y, f_z], dim=1).clamp(min=1e-4)
    d65_xyz = where(f_xyz > eps_1_3, f_xyz.pow(3.), (116. * f_xyz - 16.) / k)
    # Reference white
    xyz_colors = diag(d65).matmul(d65_xyz.flatten(start_dim=2))
    xyz = xyz_colors.view_as(input)
    return xyz