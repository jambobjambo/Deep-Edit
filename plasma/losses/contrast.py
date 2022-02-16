# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch.nn import Module
from torch.nn.functional import l1_loss

from ..sampling import laplacian_of_gaussian_2d

class ContrastLoss (Module):
    """
    Contrast loss.
    Mertens et al.
    https://mericam.github.io/papers/exposure_fusion_reduced.pdf
    """

    def __init__ (self):
        super(ContrastLoss, self).__init__()

    def forward (self, input, target):
        input_laplacian = laplacian_of_gaussian_2d(input)
        target_laplacian = laplacian_of_gaussian_2d(target)
        loss = l1_loss(input_laplacian, target_laplacian)
        return loss