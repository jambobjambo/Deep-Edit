# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pathlib import Path
from PIL import Image
from typing import List

def imread (*image_paths: str) -> Image.Image:
    """
    Load one or more raster images.

    Parameters:
        image_paths (str | list): Path(s) to image to be loaded.

    Returns:
        PIL.Image | list: Loaded image(s).
    """
    # Check
    if len(image_paths) == 0:
        return None
    # Load
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert(mode="RGB") if image.mode != "RGB" else image
        images.append(image)
    # Return
    return images if len(images) > 1 else images[0]

def is_raster_format (image_path: str) -> bool:
    """
    Is the file at the provided path a raster image?

    Parameters:
        image_path (str): Path to file.

    Returns:
        bool: Whether the file is a raster image.
    """
    RASTER_FORMATS = [".jpg", ".jpeg", ".tif", ".tiff"]
    format = Path(image_path).suffix.lower()
    return format in RASTER_FORMATS