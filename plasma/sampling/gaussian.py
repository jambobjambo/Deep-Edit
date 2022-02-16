# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import arange, exp, tensor, Tensor
from torch.nn.functional import conv2d, conv3d, pad
from typing import Tuple

def gaussian_kernel_1d (kernel_size: int, sigma: float = -1.) -> Tensor:
    """
    Normalized 1D Gaussian kernel.
    This operation is NOT differentiable w.r.t its arguments.

    Parameters:
        kernel_size (int): Kernel size, should be odd.
        sigma (float): Gaussian standard deviation. If less than 1, it is automatically computed from the kernel size.

    Returns:
        Tensor: Normalized Gaussian kernel with shape (K,).
    """
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8 if sigma < 0 else sigma # From OpenCV ::getGaussianKernel
    x = arange(kernel_size).float() - kernel_size // 2
    x = x + 0.5 if kernel_size % 2 == 0 else x
    kernel = exp((-x.pow(2.) / (2. * sigma ** 2)))
    return kernel / kernel.sum()

def gaussian_blur_2d (input: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    """
    Apply Gaussian smoothing to a 2D image.
    This operation is not differentiable w.r.t the kernel size.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W).
        kernel_size (tuple): Kernel size in each dimension (Ky,Kx).

    Returns:
        Tensor: Filtered image with shape (N,C,H,W).
    """
    _,channels,_,_ = input.shape
    kernel_size_y, kernel_size_x = kernel_size
    # Compute kernels
    kernel_x = gaussian_kernel_1d(kernel_size_x).to(input.device)
    kernel_y = gaussian_kernel_1d(kernel_size_y).to(input.device)
    # Reshape
    kernel_x = kernel_x.expand(channels, 1, 1, -1)
    kernel_y = kernel_y.expand(channels, 1, 1, -1).permute(0, 1, 3, 2).contiguous()
    # Seperable convolution
    result = conv2d(input, kernel_x, padding=(0, kernel_size_x // 2), groups=channels)
    result = conv2d(result, kernel_y, padding=(kernel_size_y // 2, 0), groups=channels)
    return result

def gaussian_blur_3d (input: Tensor, kernel_size: Tuple[int, int, int]) -> Tensor:
    """
    Apply Gaussian smoothing to a 3D volume.
    This operation is not differentiable w.r.t the kernel size.

    Parameters:
        input (Tensor): Input volume with shape (N,C,D,H,W).
        kernel_size (tuple): Kernel size in each dimension (Kz,Ky,Kx).

    Returns:
        Tensor: Filtered volume with shape (N,C,D,H,W).
    """
    _,channels,_,_,_ = input.shape
    kernel_size_z, kernel_size_y, kernel_size_x = kernel_size
    # Compute kernels
    kernel_x = gaussian_kernel_1d(kernel_size_x).to(input.device)
    kernel_y = gaussian_kernel_1d(kernel_size_y).to(input.device)
    kernel_z = gaussian_kernel_1d(kernel_size_z).to(input.device)
    # Reshape
    kernel_x = kernel_x.expand(channels, 1, 1, 1, -1)
    kernel_y = kernel_y.expand(channels, 1, 1, 1, -1).permute(0, 1, 2, 4, 3).contiguous()
    kernel_z = kernel_z.expand(channels, 1, 1, 1, -1).permute(0, 1, 4, 2, 3).contiguous()
    # Seperable convolution
    result = conv3d(input, kernel_x, padding=(0, 0, kernel_size_x // 2), groups=channels)
    result = conv3d(result, kernel_y, padding=(0, kernel_size_y // 2, 0), groups=channels)
    result = conv3d(result, kernel_z, padding=(kernel_size_z // 2, 0, 0), groups=channels)
    return result

def laplacian_of_gaussian_2d (input: Tensor) -> Tensor:
    """
    Apply a 5x5 Laplacian-of-Gaussian filter to a 2D image.

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