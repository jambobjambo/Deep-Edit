# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from imageio import imread
from torch import float32, stack, tensor, zeros_like, Tensor
from torch.nn.functional import grid_sample
from torchvision.transforms import ToTensor

def color_sample_1d (input: Tensor, lut: Tensor) -> Tensor:
    """
    Apply a 1D look-up table to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in range [-1., 1.].
        lut (Tensor): Lookup table with shape (L,) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    # Create volume
    batch,_,_,_ = input.shape
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

def color_sample_3d (input: Tensor, cube: Tensor) -> Tensor:
    """
    Apply a 3D look-up table to an image.

    Parameters:
        input (Tensor): RGB image with shape (N,3,H,W) in range [-1., 1.].
        cube (Tensor): Lookup table with shape (L,L,L,3) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    # Create volume
    batch,_,_,_ = input.shape
    cube = cube.to(input.device)
    volume = cube.repeat(batch, 1, 1, 1, 1).permute(0, 4, 1, 2, 3)
    # Create grid
    grid = input.permute(0, 2, 3, 1).unsqueeze(dim=1)
    # Sample
    result = grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=False)
    result = result.squeeze(dim=2)
    return result

def cuberead (path: str) -> Tensor:
    """
    Load a 3D LUT from file.

    Parameters:
        path (str): Path to CUBE file.

    Returns:
        Tensor: 3D LUT with shape (L,L,L,3) in range [-1., 1.].
    """
    # Read coeffients
    with open(path) as file:
        domain_min = tensor([ 0., 0., 0. ], dtype=float32)
        domain_max = tensor([ 1., 1., 1. ], dtype=float32)
        rows = []
        for line in file:
            tokens = line.split()
            if not tokens:
                continue
            elif tokens[0][0] == "#":
                continue
            elif tokens[0] == "TITLE":
                continue
            elif tokens[0] == "LUT_3D_SIZE":
                size = int(tokens[1])
            elif tokens[0] == "DOMAIN_MIN":
                domain_min = tensor([float(x) for x in tokens[1:]], dtype=float32)
            elif tokens[0] == "DOMAIN_MAX":
                domain_max = tensor([float(x) for x in tokens[1:]], dtype=float32)
            else:
                rows.append([float(x) for x in tokens])
    # Create cube
    cube = tensor(rows, dtype=float32)
    cube = cube.view(size, size, size, 3)
    # Rescale
    cube = (cube - domain_min) / (domain_max - domain_min)
    cube = 2 * cube - 1.
    return cube

def lutread (path: str) -> Tensor:
    """
    Load a 1D LUT from file.

    The LUT must be encoded as a 16-bit TIFF file.

    Parameters:
        path (str): Path to LUT file.

    Returns:
        Tensor: 1D LUT with shape (L,) in range [-1., 1.].
    """
    # Load
    image = imread(path) / 65536
    lut = ToTensor()(image).float()
    # Slice
    lut = lut[0] if lut.ndim > 2 else lut
    lut = lut[lut.shape[0] // 2]
    # Scale
    lut = 2. * lut - 1.
    return lut