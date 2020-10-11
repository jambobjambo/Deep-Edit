# 
#   Deep Edit
#   Copyright (c) 2020 Homedeck, LLC.
#

from plasma.filters import highlights, selective_color, shadows, temperature, tint, tone_curve
from torch import cat, tensor, zeros, zeros_like, Tensor
from torch.nn import Linear, Module, ReLU, Sequential, Tanh
from torch.nn.functional import interpolate
from torchvision.models import resnet34
from torchsummary import summary

class DeepEdit (Module):

    def __init__ (self):
        super(DeepEdit, self).__init__()
        # Model
        self.model = resnet34(pretrained=True, progress=True)
        in_features = self.model.fc.in_features
        self.model.fc = Sequential(
            Linear(in_features, 1024),
            ReLU(inplace=True),
            Linear(1024, 256),
            ReLU(inplace=True),
            Linear(256, 64),
            ReLU(),
            Linear(64, 11),
            Tanh()
        )
        # Constant buffers
        self.register_buffer("x_s", tensor(0.8))
        self.register_buffer("x_h", tensor(-0.9))
        self.register_buffer("v_r", tensor(2.5))
        self.register_buffer("t_0", tensor(-1.))
        self.register_buffer("selective_lum", zeros(1, 3, 1))
        self.register_buffer("basis", tensor([
            [1.0, 0.65, 0.0],   # orange
            [1.0, 1.0, 0.0],    # yellow
            [0.0, 1.0, 0.0]     # green
        ]))

    def forward (self, input: Tensor) -> Tensor:
        weights = self.weights(input)
        result = self.filter(input, weights)
        return result

    def weights (self, input: Tensor) -> Tensor:
        """
        Compute the editing coefficients for a given image.

        Parameters:
            input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].

        Returns:
            Tensor: Editing coefficients with shape (N,11) in range [-1., 1.].
        """
        input = interpolate(input, size=(512, 512), mode="bilinear", align_corners=False)
        weights = self.model(input)
        return weights

    def filter (self, input: Tensor, weights: Tensor) -> Tensor:
        """
        Apply editing forward model.

        Parameters:
            input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
            weights (Tensor): Editing coefficients with shape (N,11) in [-1., 1.].
        
        Returns:
            Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
        """
        batch, _, _, _ = input.shape
        tone_weights, chroma_weights, selective_weights = weights[:,:3], weights[:,3:5], weights[:,5:]
        # Fixed
        input = shadows(input, self.x_s)
        input = highlights(input, self.x_h)
        # Tone
        controls = cat([self.t_0.expand(batch, 1), tone_weights], dim=1)
        input = tone_curve(input, controls)
        # Chromaticity
        x_temp, x_tint = chroma_weights.split(1, dim=1)
        input = temperature(input, x_temp)
        input = tint(input, x_tint)
        # Selective color
        x_selective = selective_weights.view(-1, 3, 2)              # Nx3x2
        x_selective_lum = self.selective_lum.repeat(batch, 1, 1)    # Nx3x1
        x_selective = cat([x_selective, x_selective_lum], dim=2)    # Nx3x3
        input = selective_color(input, self.basis, x_selective)
        return input

    
if __name__ == "__main__":
    model = DeepEdit()
    summary(model, (3, 1024, 1024), batch_size=8)