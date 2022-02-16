# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import where, Tensor

def blend_overlay (base: Tensor, overlay: Tensor) -> Tensor:
    """
    Blend two images using overlay blending.

    Parameters:
        base (Tensor): Base image with shape (...) in range [-1., 1.].
        overlay (Tensor): Overlay image with shape (...) in range [-1., 1.].

    Returns:
        Tensor: Blended image with shape (...) in range [-1., 1.].
    """
    # Rescale
    base = (base + 1.) / 2.
    overlay = (overlay + 1.) / 2.
    # Compute sub blending modes
    multiply = 2. * base * overlay
    screen = 1. - 2. * (1. - base) * (1. - overlay)
    # Blend and rescale
    result = where(base < 0.5, multiply, screen)
    result = 2. * result - 1.
    return result