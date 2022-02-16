# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from imageio import imread
from torch import float32, tensor, Tensor
from torchvision.transforms import ToTensor

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