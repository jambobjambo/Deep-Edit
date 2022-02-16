# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch.nn import Module, Sequential
from torch.nn.functional import mse_loss
from torchvision.models import vgg16

class PerceptualLoss (Module):
    """
    Perceptual Losses for Realtime Style Transfer and Super Resolution.
    Justin Johnson, Alexandre Alahi, Li Fei-Fei.
    https://arxiv.org/pdf/1603.08155.pdf
    """

    def __init__ (self):
        super(PerceptualLoss, self).__init__()
        # Extract feature layers
        self.relu1_2 = Sequential()
        self.relu2_2 = Sequential()
        self.relu3_3 = Sequential()
        self.relu4_3 = Sequential()
        vgg_features = vgg16(pretrained=True, progress=False).features
        for x in range(4):
            self.relu1_2.add_module(str(x), vgg_features[x])
        for x in range(4, 9):
            self.relu2_2.add_module(str(x), vgg_features[x])
        for x in range(9, 16):
            self.relu3_3.add_module(str(x), vgg_features[x])
        for x in range(16, 23):
            self.relu4_3.add_module(str(x), vgg_features[x])
        # Freeze
        for param in self.parameters():
            param.requires_grad = False

    def forward (self, input, target_content, target_style=None):
        # Compute content loss
        input_activations = self._compute_activations(input)
        content_activations = self._compute_activations(target_content)
        content_loss = mse_loss(input_activations[1], content_activations[1]) # Use relu_2_2
        # Check
        if target_style is None:
            return content_loss
        # Compute style loss
        style_activations = self._compute_activations(target_style)
        style_loss = 0.
        for input_activation, style_activation in zip(input_activations, style_activations):
            style_loss = style_loss + mse_loss(self._gram_matrix(input_activation), self._gram_matrix(style_activation))
        return content_loss, style_loss

    def _compute_activations (self, input):
        # Normalize input
        input = (input + 1.) / 2. # [-1., 1.] -> [0., 1.]
        mean = input.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = input.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        input = (input - mean) / std
        # Compute activations
        feat1_2 = self.relu1_2(input)
        feat2_2 = self.relu2_2(feat1_2)
        feat3_3 = self.relu3_3(feat2_2)
        feat4_3 = self.relu4_3(feat3_3)
        return feat1_2, feat2_2, feat3_3, feat4_3

    def _gram_matrix (self, input): # Must be batched tensor
        batch, channels, height, width = input.shape
        features = input.view(batch, channels, height * width) # flatten to depth vector
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (channels * height * width) # compute outer gram as A A^T
        return gram