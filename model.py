# 
#   Deep Edit
#   Copyright (c) 2020 Homedeck, LLC.
#

from plasma.filters import clarity, contrast, exposure, highlights, selective_color, shadows, temperature, tint
from plasma.filters.functional import radial_gradient
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
            Linear(64, 12),
            Tanh()
        )
        # Constant buffers
        self.register_buffer("x_s", tensor(0.75))
        self.register_buffer("x_h", tensor(-0.8))
        self.register_buffer("v_r", tensor(2.5))
        self.register_buffer("selective_lum", zeros(1, 3, 1))
        self.register_buffer("basis", tensor([
            [1.0, 0.65, 0.0],   # orange
            [1.0, 1.0, 0.0],    # yellow
            [0.0, 1.0, 0.0]     # green
        ]))

    def forward (self, input: Tensor) -> Tensor:
        weights = self.coefficients(input)
        result = self.filter(input, weights)
        return result

    def coefficients (self, input: Tensor) -> Tensor:
        """
        Compute the editing coefficients for a given image.

        Parameters:
            input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].

        Returns:
            Tensor: Editing coefficients with shape (N,12) in range [-1., 1.].
        """
        input = interpolate(input, size=(512, 512), mode="bilinear", align_corners=False)
        weights = self.model(input)
        return weights

    def filter (self, input: Tensor, weights: Tensor) -> Tensor:
        """
        Apply editing forward model.

        Parameters:
            input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
            weights (Tensor): Editing coefficients with shape (M,3) in [-1., 1.].
        
        Returns:
            Tensor: Filtered image with shape (N,3,H,W) in range [-1., 1.].
        """
        batch, _, _, _ = input.shape
        linear_weights, selective_weights = weights[:,:6], weights[:,6:]
        # Fixed
        input = shadows(input, self.x_s)
        input = highlights(input, self.x_h)
        # Linear
        x_0, x_1, x_2, x_3, x_4, x_5 = linear_weights.split(1, dim=1)
        input = contrast(input, x_0)
        input = exposure(input, x_1)
        input = temperature(input, x_2)
        input = tint(input, x_3)
        input = contrast(input, x_4) # CHECK # Should this be fixed function??
        # Selective color
        x_selective = selective_weights.view(-1, 3, 2)              # Nx3x2
        x_selective_lum = self.selective_lum.repeat(batch, 1, 1)    # Nx3x1
        x_selective = cat([x_selective, x_selective_lum], dim=2)    # Nx3x3
        input = selective_color(input, self.basis, x_selective)
        # Vignette # INCOMPLETE
        # mask = 1. - radial_gradient(input, self.v_r)
        # input = exposure(input, mask * coefficients[:,11].view(-1, 1, 1, 1))
        return input

    
if __name__ == "__main__":
    model = DeepEdit()
    summary(model, (3, 1024, 1024), batch_size=8)