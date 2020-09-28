# 
#   Deep Edit
#   Copyright (c) 2020 Homedeck, LLC.
#

from plasma.filters import clarity, contrast, exposure, highlights, selective_color, shadows, temperature, tint
from plasma.filters.functional import radial_gradient
from torch import tensor, zeros_like, Tensor
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
        # Selective color
        basis = tensor([
            [255., 165., 0.],   # orange
            [255., 255., 0.],   # yellow
            [0., 255., 0.]      # green
        ]) / 255.
        self.register_buffer("basis", basis)

    def forward (self, input: Tensor) -> Tensor:
        weights = self.coefficients(input)
        result = self.edit(input, weights)
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

    def edit (self, input: Tensor, coefficients: Tensor) -> Tensor:
        """
        Apply editing forward model.

        Parameters:
            input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
            coefficients (Tensor): Editing coefficients with shape (N,3) in [-1., 1.].
        
        Returns:
            Tensor: Result image with shape (N,3,H,W) in range [-1., 1.].
        """
        batch, _, height, width = input.shape
        # Linear
        #input = shadows(input, 0.75)
        #input = highlights(input, -0.8)
        input = contrast(input, coefficients[:,0].view(-1, 1, 1, 1))
        input = exposure(input, coefficients[:,1].view(-1, 1, 1, 1))
        input = temperature(input, coefficients[:,2].view(-1, 1, 1, 1))
        input = tint(input, coefficients[:,3].view(-1, 1, 1, 1))
        input = clarity(input, coefficients[:,4].view(-1, 1, 1, 1))
        # Selective color
        sat_weight = coefficients[:,5:8].view(batch, 3, 1, 1).expand(-1, -1, height, width)
        lum_weight = coefficients[:,8:11].view(batch, 3, 1, 1).expand(-1, -1, height, width)
        exp_weight = zeros_like(sat_weight)
        input = selective_color(input, self.basis, exp_weight, sat_weight, lum_weight)
        # Vignette
        mask = 1. - radial_gradient(input, 2.5)
        input = exposure(input, mask * coefficients[:,11].view(-1, 1, 1, 1))
        return input

    
if __name__ == "__main__":
    model = DeepEdit()
    summary(model, (3, 1024, 1024), batch_size=8)