# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from torch import tensor, where, Tensor

def natural_cubic_curve (input: Tensor, control: Tensor) -> Tensor:
    """
    Apply a natural cubic tone curve to an image.

    The control query points are fixed at [-1.0, -0.33, 0.33, 1.0].
    Note that this function does not clamp the output tensor to any range.
    
    Parameters:
        input (Tensor): Input image with shape (N,...) in range [-1., 1.].
        control (Tensor): Control value points with shape (N,4) in range [-1., 1.].

    Returns:
        Tensor: Result image with shape (N,...) in range [-1., 1.].
    """
    _, channels, height, width = input.shape
    x_0, x_1, x_2, x_3 = -1, -1. / 3., 1. / 3., 1.
    y_0, y_1, y_2, y_3 = control.split(1, dim=1)
    x = input.flatten(start_dim=1)
    # Piecewise linear curve
    m_1 = (y_1 - y_0) / (x_1 - x_0)
    m_2 = (y_2 - y_1) / (x_2 - x_1)
    m_3 = (y_3 - y_2) / (x_3 - x_2)
    l_1 = m_1 * (x - x_0) + y_0
    l_2 = m_2 * (x - x_1) + y_1
    l_3 = m_3 * (x - x_3) + y_3
    # Cubic corrections
    z_1 = 6 * (m_3 * x_1 + m_2 * x_2 - m_3 * x_2 + 2 * m_2 * x_3 + 2 * m_1 * x_1 - 2 * m_1 * x_3 - 3 * m_2 * x_1) / (4 * (x_0 * x_1 + x_2 * x_3 - x_0 * x_3) - (x_1 + x_2) ** 2)
    z_2 = 6 * (m_2 * x_1 + m_1 * x_2 - m_1 * x_1 + 2 * m_2 * x_0 + 2 * m_3 * x_2 - 2 * m_3 * x_0 - 3 * m_2 * x_2) / (4 * (x_0 * x_1 + x_2 * x_3 - x_0 * x_3) - (x_1 + x_2) ** 2)
    a_1 = z_1 / (6 * (x_0 - x_1))
    b_1 = 2 * z_1 / (6 * (x_1 - x_0))
    a_2 = (2 * z_1 + z_2) / (6 * (x_1 - x_2))
    b_2 = (2 * z_2 + z_1) / (6 * (x_2 - x_1))
    a_3 = z_2 / (3 * (x_2 - x_3))
    b_3 = z_2 / (6 * (x_3 - x_2))
    c_1 = a_1 * (x - x_1) ** 2 * (x - x_0) + b_1 * (x - x_1) * (x - x_0) ** 2
    c_2 = a_2 * (x - x_2) ** 2 * (x - x_1) + b_2 * (x - x_2) * (x - x_1) ** 2
    c_3 = a_3 * (x - x_3) ** 2 * (x - x_2) + b_3 * (x - x_3) * (x - x_2) ** 2
    # Final curve
    y = where(x > x_1, l_2 + c_2, l_1 + c_1)
    y = where(x > x_2, l_3 + c_3, y)
    result = y.view(-1, channels, height, width)
    return result

def tonal_exposure (input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply tonal exposure adjustment to an image.

    Parameters:
        input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
        weight (Tensor | float): Scalar weight with shape (N,1) in range [-1., 1.].

    Returns:
        Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
    """
    samples, _, _, _ = input.shape
    ANCHORS = tensor([
        # x = [-1, 0, 1]
        [-1., -1., -1.],            # c_0
        [-0.874, -1. / 3., 0.318],  # c_1
        [-0.686, 1. / 3., 0.812],   # c_2
        [-0.254, 1., 1.]            # c_3
    ])
    ANCHORS = ANCHORS.repeat(samples, 1, 1).to(input.device)
    control = 0.5 * ANCHORS[:,:,0] * weight * (weight - 1.) - ANCHORS[:,:,1] * (weight + 1) * (weight - 1) + 0.5 * ANCHORS[:,:,2] * weight * (weight + 1)
    result = natural_cubic_curve(input, control)
    result = result.clamp(min=-1., max=1.)
    return result