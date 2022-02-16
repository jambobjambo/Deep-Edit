# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
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