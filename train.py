# 
#   Deep Edit
#   Copyright (c) 2020 Homedeck, LLC.
#

from argparse import ArgumentParser
from colorama import Fore, Style
from suya import set_suya_access_key
from suya.torch import PairedDataset
from torch import device as get_device, rand
from torch.cuda import is_available as cuda_available
from torch.jit import save, script
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.utils import make_grid
from torchsummary import summary
import tableprint

from highlight import HighlightLoss
from model import DeepEdit

# Parse arguments
parser = ArgumentParser(description="Deep Color: Training")
parser.add_argument("--tag", type=str, required=True, help="Dataset tag on Suya")
parser.add_argument("--suya-key", type=str, required=False, default=None, help="Suya access key")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="Nominal learning rate")
parser.add_argument("--epochs", type=int, default=10, help="Epochs")
parser.add_argument("--batch-size", type=int, default=12, help="Minibatch size")
parser.add_argument("--lambda-highlight", type=float, default=2e-1, help="Highlight regularization weight")
args = parser.parse_args()

# Create dataset
set_suya_access_key(args.suya_key)
transform = Compose([
    Resize(512),
    CenterCrop(512),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = PairedDataset(args.tag, transform=transform, size=2000)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, drop_last=True, pin_memory=True, shuffle=True)

# Create model
device = get_device("cuda:0") if cuda_available() else get_device("cpu")
model = DeepEdit().to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

# Create loss
l1_loss = L1Loss().to(device)
highlight_loss = HighlightLoss().to(device)

# Print
print("Preparing for training:")
summary(model, (3, 512, 512), batch_size=args.batch_size)

# Create summary writer
with SummaryWriter() as summary_writer:

    # Print table and graph
    HEADERS = ["Iteration", "Epoch", "Total"]
    print(tableprint.header(HEADERS))

    # Setup for training
    model.train(mode=True)
    iteration_index = 0
    last_loss = 1e+10

    # Train
    for epoch in range(args.epochs):

        # Iterate over all minibatches
        for input, target in dataloader:

            # Run forward pass
            input, target = input.to(device), target.to(device)
            prediction = model(input)

             # Compute losses
            loss_l1 = l1_loss(prediction, target)
            loss_highlight = highlight_loss(prediction, target)
            loss_total = loss_l1 + args.lambda_highlight * loss_highlight

            # Backpropagate
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Log
            summary_writer.add_scalar("Deep Edit/Total Loss", loss_total, iteration_index)
            summary_writer.add_scalar("Deep Edit/L1 Loss", loss_l1, iteration_index)
            summary_writer.add_scalar("Deep Edit/Highlight Loss", loss_highlight, iteration_index)
            LOG_DATA = [
                f"{iteration_index}",
                f"{epoch}",
                f"{Style.BRIGHT}{Fore.GREEN if loss_total < last_loss else Fore.RED}{loss_total:.4f}{Style.RESET_ALL}"
            ]
            print(tableprint.row(LOG_DATA))
            last_loss = loss_total
            iteration_index += 1

        # Log images
        to_grid = lambda mbatch: make_grid(mbatch.cpu(), range=(-1., 1.), nrow=args.batch_size, normalize=True)
        summary_writer.add_image("Input", to_grid(input), iteration_index)
        summary_writer.add_image("Prediction", to_grid(prediction), iteration_index)
        summary_writer.add_image("Target", to_grid(target), iteration_index)

        # Save model
        model.cpu()
        scripted_model = script(model)
        save(scripted_model, "deep_edit.pt")
        model.to(device)

    # Print
    print(tableprint.bottom(len(HEADERS)))