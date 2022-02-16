# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, split, stack, Tensor
from torch.nn.functional import cosine_similarity

from ..conversion import rgb_to_yuv, yuv_to_rgb

def selective_color (input: Tensor, basis: Tensor, weight: Tensor) -> Tensor:
    """
    Apply selective color adjustment on a given image.

    All `M` filters are applied simultaneously.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        basis (Tensor): Basis RGB colors with shape (M,3) in range [0., 1.].
        weight (Tensor): Per-basis hue, saturation, and luminance adjustments with shape (N,M,3) in range [-1., 1.].
    
    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    _, _, height, width  = input.shape
    bases, _ = basis.shape
    # Convert to YUV
    yuv = rgb_to_yuv(input)
    y = yuv[:,:1,...]                                               # Nx1xHxW
    uv = yuv[:,1:,...]                                              # Nx2xHxW
    # Compute weight maps
    relevance = _selective_color_weight_map(input, basis)           # NxMxHxW
    relevance = relevance.flatten(start_dim=2)                      # NxMx(H*W)
    hue_weight, sat_weight, lum_weight = weight.split(1, dim=2)     # NxMx1
    hue = (relevance * hue_weight).view(-1, bases, height, width)   # NxMxHxW
    sat = (relevance * sat_weight).view(-1, bases, height, width)   # NxMxHxW
    lum = (relevance * lum_weight).view(-1, bases, height, width)   # NxMxHxW
    # Adjust hues
    rotations = stack([ hue.cos(), -hue.sin(), hue.sin(), hue.cos() ], dim=4)   # NxMxHxWx4
    rotations = rotations.view(-1, bases, height, width, 2, 2)      # NxMxHxWx2x2
    uv = uv.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=4)       # NxHxWx2x1
    for rotation in rotations.split(1, dim=1):
        uv = rotation.squeeze(dim=1) @ uv                           # NxHxWx2x1
    uv = uv.squeeze(dim=4).permute(0, 3, 1, 2).contiguous()         # Nx2xHxW
    # Adjust saturation
    sat = sat.sum(dim=1, keepdim=True).clamp(min=-1., max=1.)       # Nx1xHxW
    uv = uv * (sat + 1.)                                            # Nx2xHxW
    # Adjust luminance
    lum = lum.sum(dim=1, keepdim=True).clamp(min=-1., max=1.)       # Nx1xHxW
    y = y * (0.5 * lum + 1.)
    # Convert to RGB
    yuv = cat([y, uv], dim=1)
    result = yuv_to_rgb(yuv)
    return result

def _selective_color_weight_map (input: Tensor, basis: Tensor) -> Tensor:
    """
    Compute the color weight map for selective coloring.
    
    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        basis (Tensor): Basis colors with shape (M,3) in range [0., 1.].
    
    Returns:
        Tensor: Color weight map with shape (N,M,H,W) in range [0., 1.].
    """
    samples, _, height, width = input.shape
    bases, _ = basis.shape
    # Convert basis
    basis = (2.0 * basis - 1.0).transpose(0, 1).unsqueeze(dim=0)    # 1x3xM
    uv_basis = rgb_to_yuv(basis)[:,1:,...]                          # 1x2xM
    uv_basis = uv_basis.view(1, 2, -1, 1, 1)                        # 1x2xMx1x1
    uv_basis = uv_basis.repeat(samples, 1, 1, height, width)        # Nx2xMxHxW
    # Convert all to YUV
    uv_colors = rgb_to_yuv(input)[:,1:,...].unsqueeze(dim=2)        # Nx2x1xHxW
    uv_colors = uv_colors.repeat(1, 1, bases, 1, 1)                 # Nx2xMxHxW
    # Compare
    weight_map = cosine_similarity(uv_colors, uv_basis, dim=1)
    weight_map = weight_map.clamp(min=0.)
    return weight_map