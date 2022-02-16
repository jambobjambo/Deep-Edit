# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import device
from torch.cuda import device_count

_d = None

def set_io_device (d: device):
    """
    Set a compute device for IO operations.
    Using a CUDA GPU can massively accelerate IO operations.

    Parameters:
        d (torch.device): Compute device.
    """
    global _d
    _d = d

def get_io_device () -> device:
    """
    Get the compute device for IO operations.
    
    Returns:
        torch.device: Compute device.
    """
    return _d or _get_default_device()

def _get_default_device () -> device:
    """
    Get the default compute device for IO operations.

    Returns:
        torch.device: Compute device.
    """
    available_devices = device_count()
    if available_devices == 0:
        return device("cpu")
    elif available_devices == 1:
        return device("cuda:0")
    else:
        return device("cuda:1")