# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import Tensor
from torch.nn.functional import conv2d, interpolate

from .box import box_filter

def guided_filter (input: Tensor, guide: Tensor, radius: int, eps: float) -> Tensor:
    """
    Apply the guided image filter to a 2D image.

    http://kaiminghe.com/publications/pami12guidedfilter.pdf

    Parameters:
        input (Tensor): Input image with shape (N,C,Sx,Sy).
        guide (Tensor): Guide image with shape (N,1,H,W).
        radius (int): Filter window radius.
        eps (float): Ridge regularization coefficient.

    Returns:
        Tensor: Filtered image with shape (N,C,H,W).
    """
    # Upsample input
    _, _, height, width = guide.shape
    input = interpolate(input, size=(height, width), mode="bilinear", align_corners=False)
    # Guide variance
    guide_mean = box_filter(guide, radius)
    guide_mean_2 = box_filter(guide.pow(2.), radius)
    guide_variance = guide_mean_2 - guide_mean.pow(2.)
    # Input covariance
    input_mean = box_filter(input, radius)
    input_guide_mean = box_filter(input * guide, radius)
    input_guide_covariance = input_guide_mean - guide_mean * input_mean
    # Compute linear model
    a = input_guide_covariance / (guide_variance + eps)
    b = input_mean - a * guide_mean
    # Apply model
    a_mean = box_filter(a, radius)
    b_mean = box_filter(b, radius)
    return a_mean * guide + b_mean