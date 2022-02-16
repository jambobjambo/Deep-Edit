# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, diag_embed, ones_like, tensor, where, Tensor

from ..conversion import linear_to_srgb, rgb_to_xyz ,srgb_to_linear, xyy_to_xyz, xyz_to_rgb, xyz_to_xyy

def color_balance (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply color balance adjustment (temperature and tint) on an RGB image.

    Parameters:
        input (Tensor): Input RGB image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor): Scalar weight with shape (N,2) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    # Convert to XYZ
    _, _, height, width = input.shape
    input = srgb_to_linear(input)
    xyz = rgb_to_xyz(input)
    # Apply Bradford transformation # Used by Photoshop and Lightroom
    d65_white = tensor([[ 0.95047, 1.0, 1.08883 ]]).float().to(input.device).unsqueeze(dim=2) # for 2 degree observer
    dst_white = _temperature_tint_to_xyz(weight).unsqueeze(dim=2)
    BRADFORD = tensor([
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 1.0296000]
    ]).float().to(input.device)
    BRADFORD_INV = tensor([
        [0.9869929, -0.1470543, 0.1599627],
        [0.4323053, 0.5183603, 0.0492912],
        [-0.0085287, 0.0400428, 0.9684867]
    ]).float().to(input.device)
    d65_cone = BRADFORD @ d65_white
    dst_cone = BRADFORD @ dst_white
    scale = diag_embed((d65_cone / dst_cone).squeeze(dim=2))
    adaptation = BRADFORD_INV @ scale @ BRADFORD
    xyz = adaptation @ xyz.flatten(start_dim=2)
    xyz = xyz.view(-1, 3, height, width)
    # Convert to sRGB
    result = xyz_to_rgb(xyz)
    result = linear_to_srgb(result)
    return result

def _temperature_tint_to_xyz (input: Tensor) -> Tensor:
    """
    Compute the XYZ coordinate for a given temperature-tint vector.

    Parameters:
        input (Tensor): Temperature-tint vector with shape (N,2) in range [-1., 1.].

    Returns:
        Tensor: White point XYZ ccoordinate with shape (N,3) in range [0., 1.].
    """
    # Convert to absolute temperature
    temp, tint = input.split(1, dim=1)
    temp = _temperature_to_kelvin(temp)
    # Interpolate on Planckian locus
    x_blackbody = -0.2661239 * 1e+9 / temp.pow(3.) - 0.2343589 * 1e+6 / temp.pow(2.) + 0.8776956 * 1e+3 / temp + 0.179910
    x_daylight = -3.0258469 * 1e+9 / temp.pow(3.) + 2.1070379 * 1e+6 / temp.pow(2.) + 0.2226347 * 1e+3 / temp + 0.240390
    x = where(temp < 4000, x_blackbody, x_daylight)
    y_low  = -1.1063814 * x.pow(3) - 1.34811020 * x.pow(2) + 2.18555832 * x - 0.20219683
    y_mid = -0.9549476 * x.pow(3) - 1.37418593 * x.pow(2) + 2.09137015 * x - 0.16748867
    y_high = 3.0817580 * x.pow(3) - 5.87338670 * x.pow(2) + 3.75112997 * x - 0.37001483
    y = where(temp < 2222, y_low, y_mid)
    y = where(temp < 4000, y, y_high)
    # Convert to XYZ
    Y = ones_like(y)
    xyY = cat([x, y, Y], dim=1)
    xyz = xyy_to_xyz(xyY)
    X, Y, Z = xyz.split(1, dim=1)
    Y = Y / (0.3 * tint + 1.)
    xyz = cat([X, Y, Z], dim=1) # This tint correction is incorrect, but it's used by DT
    return xyz

def _temperature_to_kelvin (input: Tensor) -> Tensor: # TEST
    """
    Convert temperature in a relative range to absolute Kelvin.

    Parameters:
        input (Tensor): Relative temperature with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Absolute temperature in Kelvin.
    """
    MIN_TEMP = 1667.0
    D65_TEMP = 6503.6
    MAX_TEMP = 20000.0
    kelvin = MIN_TEMP * input * (input - 1.) / 2. - D65_TEMP * (input + 1) * (input - 1.) + MAX_TEMP * input * (input + 1.) / 2.
    return kelvin