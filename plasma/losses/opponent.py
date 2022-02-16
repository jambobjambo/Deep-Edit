# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch.nn import Module
from torch.nn.functional import l1_loss

from ..conversion import rgb_to_yuv

class ColorOpponentLoss (Module):
    """
    Color opponent loss, which responds to chromaticity differences.
    """

    def __init__ (self):
        super(ColorOpponentLoss, self).__init__()

    def forward (self, input, target):
        input_yuv, target_yuv = rgb_to_yuv(input), rgb_to_yuv(target)
        loss_chroma = l1_loss(input_yuv[:,1:,:,:], target_yuv[:,1:,:,:])
        return loss_chroma