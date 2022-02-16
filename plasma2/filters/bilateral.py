# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import cat, linspace, meshgrid, ones, ones_like, stack, where, Tensor
from torch.nn.functional import grid_sample, interpolate, pad
from typing import Optional, Tuple

from ..conversion import rgb_to_luminance
from .gaussian import gaussian_filter_3d

def bilateral_filter (input: Tensor, guide: Tensor, kernel_size: Tuple[int, int], grid_size: Optional[Tuple[int, int, int]]=None) -> Tensor:
    """
    Apply the joint bilateral filter to an image.

    We utilize the Bilateral Grid as described by Chen et al.
    https://people.csail.mit.edu/sparis/publi/2007/siggraph/Chen_07_Bilateral_Grid.pdf

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W).
        guide (Tensor): Guide image with shape (N,1,H,W).
        kernel_size (tuple): Kernel size in intensity and spatial dimensions (Ki,Ks).
        grid_size (tuple): Bilateral grid size. If `None`, a suitable default will be used.

    Returns:
        Tensor: Filtered image with shape (N,C,H,W).
    """
    kernel_size = (kernel_size[0], kernel_size[1], kernel_size[1])
    grid_size = grid_size if grid_size is not None else (16, 512, 512)
    channels = input.split(1, dim=1)
    result = []
    for channel in channels:
        grid = splat_bilateral_grid(channel, guide, grid_size)
        grid = gaussian_filter_3d(grid, kernel_size)
        channel = slice_bilateral_grid(grid, guide, homogenous=True)
        result.append(channel)
    result = cat(result, dim=1)
    return result

def splat_bilateral_grid (input: Tensor, guide: Tensor, grid_size: Tuple[int, int, int]) -> Tensor:
    """
    Splat an image into a homogenous bilateral grid.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W).
        guide (Tensor): Splatting guide map with shape (N,1,H,W) in range [-1., 1.].
        grid_size (tuple): Grid size in each dimension (I,Sy,Sx).

    Returns:
        tuple: Bilateral grid with shape (N,D,I,Sy,Sx), where D = C + 1.
    """
    samples, _,_,_ = input.shape
    intensity_bins, spatial_bins_y, spatial_bins_x = grid_size
    # Downsample
    downsampled_input = interpolate(input, size=(spatial_bins_y, spatial_bins_x), mode="bilinear", align_corners=False) # NxCxSxS
    downsampled_guide = interpolate(guide, size=(spatial_bins_y, spatial_bins_x), mode="bilinear", align_corners=False) # Nx1xSxS
    # Create volume
    input_grid = downsampled_input.unsqueeze(dim=2) # NxCx1xSxS
    weight_grid = ones(samples, 1, 1, spatial_bins_y, spatial_bins_x).to(input.device) # Nx1x1xSxS
    input_grid = cat([input_grid, weight_grid], dim=1) # NxDx1xSxS
    input_volume = pad(input_grid, (0, 0, 0, 0, 0, intensity_bins - 1, 0, 0), "constant", 0.) # NxCxIxSxS
    # Create sample grid
    ig, hg, wg = meshgrid(linspace(-1., 1., intensity_bins), linspace(-1., 1., spatial_bins_y), linspace(-1., 1., spatial_bins_x))
    ig = ig.repeat(samples, 1, 1, 1).to(input.device) - (downsampled_guide + 1.)
    hg = hg.repeat(samples, 1, 1, 1).to(input.device)
    wg = wg.repeat(samples, 1, 1, 1).to(input.device)
    sample_grid = stack([wg, hg, ig], dim=4)
    # Sample
    intensity_grid = grid_sample(input_volume, sample_grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return intensity_grid

def slice_bilateral_grid (input: Tensor, guide: Tensor, homogenous: bool=False) -> Tensor:
    """
    Slice a bilateral grid to an image.

    Parameters:
        input (Tensor): Input bilateral grid with shape (N,C,I,Sy,Sx).
        guide (Tensor): Slicing guide map with shape (N,1,H,W) in range [-1., 1.].
        homogenous (bool): Whether a homogenous divide is to be performed. The last channel is assumed to be the homogenous coordinate.

    Returns:
        Tensor: Sliced image with shape (N,D,H,W), where D = C-1 if homogenous else C.
    """
    _, channels, _, _, _ = input.shape
    samples, _, height, width = guide.shape
    # Create slice grid
    hg, wg = meshgrid(linspace(-1., 1., height), linspace(-1., 1., width))
    hg = hg.repeat(samples, 1, 1).unsqueeze(dim=3).to(input.device)
    wg = wg.repeat(samples, 1, 1).unsqueeze(dim=3).to(input.device)
    slice_grid = guide.permute(0, 2, 3, 1).contiguous()     # NxHxWx1
    slice_grid = cat([wg, hg, slice_grid], dim=3)           # NxHxWx3
    slice_grid = slice_grid.unsqueeze(dim=1)                # Nx1xHxWx3
    # Sample
    result = grid_sample(input, slice_grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    result = result.squeeze(dim=2)  # NxDxHxW
    # Check for homogenous divide
    if not homogenous:
        return result
    # Perform homogenous divide
    result, weight = result.split(channels-1, dim=1)
    weight = where(weight <= 0, ones_like(weight), weight) # Prevent divide by zero
    result = result / weight
    return result