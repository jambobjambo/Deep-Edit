# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import stack, zeros_like, Tensor
from torch.nn.functional import grid_sample

def discrete_curve_1d (input: Tensor, lut: Tensor) -> Tensor:
    """
    Apply a 1D look-up table to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        lut (Tensor): Lookup table with shape (L,) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    # Create volume
    batch, _, _, _ = input.shape
    lut = lut.to(input.device)
    volume = lut.repeat(batch, 1, 1, 1)
    # Create grid
    colors = input.permute(0, 2, 3, 1)
    wg = colors.flatten(2)
    hg = zeros_like(wg)
    grid = stack([wg, hg], dim=3)
    # Sample
    result = grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=False)
    result = result.squeeze(dim=1).view_as(colors).permute(0, 3, 1, 2)
    return result

def discrete_curve_3d (input: Tensor, cube: Tensor) -> Tensor:
    """
    Apply a 3D look-up table to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        cube (Tensor): Lookup table with shape (L,L,L,3) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    # Create volume
    batch, _, _, _ = input.shape
    cube = cube.to(input.device)
    volume = cube.repeat(batch, 1, 1, 1, 1).permute(0, 4, 1, 2, 3)
    # Create grid
    grid = input.permute(0, 2, 3, 1).unsqueeze(dim=1)
    # Sample
    result = grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=False)
    result = result.squeeze(dim=2)
    return result