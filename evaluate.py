# 
#   Deep Edit
#   Copyright (c) 2020 Homedeck, LLC.
#

from argparse import ArgumentParser
from PIL import Image
from torch import cat, device as get_device, set_grad_enabled
from torch.cuda import is_available as cuda_available
from torch.jit import load
from torchvision.transforms import Compose, Normalize, Resize, ToPILImage, ToTensor
from model import DeepEdit
# Parse arguments
parser = ArgumentParser(description="Deep Color: Evaluate")
parser.add_argument("--model", type=str, default="deep_edit.pt", help="Path to trained model")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
args = parser.parse_args()

# Load model
device = get_device("cuda:0") if cuda_available() else get_device("cpu")
model = load(args.model, map_location=device).eval().to(device)
print(model)
set_grad_enabled(False)

# Load image
to_tensor = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input = Image.open(args.input)
input = to_tensor(input).unsqueeze(dim=0).to(device)

# Run forward
deepedit = DeepEdit()
weights = deepedit.weights(input)
result = model.filter(input, weights)

# Output
print("Weights:", weights.squeeze(dim=0))
to_image = Compose([
    Normalize(mean=[-1., -1., -1.], std=[2., 2., 2.]),
    ToPILImage()
])
to_image(result.cpu().squeeze(dim=0)).save(f"result.jpg")