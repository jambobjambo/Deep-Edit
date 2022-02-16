# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import tensor, Tensor
from torch.nn.functional import conv2d, pad

def laplacian_of_gaussian_filter (input: Tensor) -> Tensor:
    """
    Apply a 5x5 Laplacian-of-Gaussian filter to an image.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W).

    Returns:
        Tensor: Filtered image with shape (N,C,H,W).
    """
    _,channels,_,_ = input.shape
    # Build kernels
    gaussian_kernel = 1. / 16. * tensor([
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ])
    laplacian_kernel = tensor([ # CHECK # Normalize?
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., -24., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]
    ])
    gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1).to(input.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1).to(input.device)
    # Apply Gaussian
    gaussian = pad(input, (2, 2, 2, 2), mode="reflect")
    gaussian = conv2d(gaussian, gaussian_kernel, groups=channels)
    # Apply Laplacian
    laplacian = pad(gaussian, (2, 2, 2, 2), mode="reflect")
    laplacian = conv2d(laplacian, laplacian_kernel, groups=channels)
    # Compute absolute response
    response = laplacian.abs()
    return response