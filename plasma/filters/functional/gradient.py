# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, linspace, meshgrid, Tensor
from torch.nn.functional import interpolate

def radial_gradient (input: Tensor, radius: Tensor) -> Tensor:
    """
    Create a radial gradient which starts from the center of the given image.

    We use the equation: f(x) = 2|cx|^3 - 3|cx|^2 + 1 where c = 1 / radius.
    This operation is differentiable w.r.t the radius.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        radius (Tensor | float): Normalized radius with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    extent = min(width, height)
    hg, wg = meshgrid(linspace(-1., 1., extent), linspace(-1., 1., extent))
    hg = hg.repeat(samples, 1, 1, 1).to(input.device)
    wg = wg.repeat(samples, 1, 1, 1).to(input.device)
    field = cat([hg, wg], dim=1)
    field = field.norm(dim=1, p=2, keepdim=True)
    field = field.flatten(start_dim=1) / radius
    field = field.view(-1, 1, extent, extent).clamp(max=1.)
    mask = 2 * field.abs().pow(3) - 3 * field.abs().pow(2) + 1
    mask = interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask

def top_bottom_gradient (input: Tensor, length: Tensor):
    """
    Create a vertical gradient which starts from the top of the given image.
    
    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(0., 1., height).to(input.device)
    field = field.repeat(samples, 1, width, 1).permute(0, 1, 3, 2).contiguous()
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field

def bottom_top_gradient (input: Tensor, length: Tensor) -> Tensor:
    """
    Create a vertical gradient which starts from the bottom of the given image.

    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(1., 0., height).to(input.device)
    field = field.repeat(samples, 1, width, 1).permute(0, 1, 3, 2).contiguous()
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field

def left_right_gradient (input: Tensor, length: Tensor) -> Tensor:
    """
    Create a horizontal gradient which starts from the left of the given image.

    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length with shape (N,1) in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(0., 1., width).to(input.device)
    field = field.repeat(samples, 1, height, 1)
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field

def right_left_gradient (input: Tensor, length: Tensor) -> Tensor:
    """
    Create a horizontal gradient which starts from the right of the given image.

    This operation is differentiable w.r.t the length.

    Parameters:
        input (Tensor): Input image with shape (N,C,H,W) in range [-1., 1.].
        length (Tensor | float): Normalized length in range [0., 1.].

    Returns:
        Tensor: Gradient mask with shape (N,1,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    field = linspace(1., 0., width).to(input.device)
    field = field.repeat(samples, 1, height, 1)
    field = field.flatten(start_dim=1) / length
    field = field.view(-1, 1, height, width).clamp(max=1.)
    field = 1. - field
    return field