# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import cat, tensor, Tensor

def xyz_to_xyy (input: Tensor) -> Tensor:
    """
    Convert XYZ to xyY.

    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [0., 1.].
    
    Returns:
        Tensor: xyY image with shape (N,3,...) in range [0., 1.].
    """
    X, Y, Z = input.split(1, dim=1)
    s = (X + Y + Z).clamp(min=1e-4)
    x = X / s
    y = Y / s
    xyY = cat([x, y, Y], dim=1)
    return xyY

def xyy_to_xyz (input: Tensor) -> Tensor:
    """
    Convert xyY to XYZ.
    
    Parameters:
        input (Tensor): Input image with shape (N,3,...) in range [0., 1.].
    
    Returns:
        Tensor: XYZ image with shape (N,3,...) in range [0., 1.].
    """
    x, y, Y = input.split(1, dim=1)
    X = x * Y / y
    Z = (1 - x - y) * Y / y
    XYZ = cat([X, Y, Z], dim=1)
    return XYZ