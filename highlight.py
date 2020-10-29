# 
#   Deep Edit
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import tensor, zeros_like, Tensor
from torch.nn import Module
from torch.nn.functional import l1_loss

class HighlightLoss (Module):
    """
    Highlight regularization loss.

    Parameters:
        threshold (float): Saturation threshold in range [0., 1.).
    """

    def __init__ (self, threshold=0.99):
        super(HighlightLoss, self).__init__()
        self.register_buffer("threshold", tensor(threshold))

    def forward (self, input: Tensor, target: Tensor=None):
        input = (input + 1.) / 2.
        input_zebras = (input - self.threshold).clamp(min=0.)
        input_zebras = input_zebras / (1. - self.threshold)
        if target is not None:
            target = (target + 1.) / 2.
            target_zebras = (target - self.threshold).clamp(min=0.)
            target_zebras = target_zebras / (1. - self.threshold)
        else:
            target_zebras = zeros_like(input_zebras)
        loss = l1_loss(input_zebras, target_zebras)
        return loss