# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from piexif import load as load_exif, dump as dump_exif, ImageIFD
from PIL import Image

def exifread (image_path: str) -> dict: # INCOMPLETE # Switch to whitelist
    """
    Read the EXIF dictionary from a file.

    Parameters:
        image_path (str): Path to image.

    Returns:
        dict: EXIF metadata dictionary.
    """
    # Load exif
    exif = load_exif(image_path)
    # Strip tags
    if "thumbnail" in exif:
        del exif["thumbnail"]
    if "0th" in exif and 700 in exif["0th"]:
        del exif["0th"][700]
    if "Interop" in exif:
        del exif["Interop"]
    if "Exif" in exif:
        if 37500 in exif["Exif"]:
            del exif["Exif"][37500]
        if 37510 in exif["Exif"]:
            del exif["Exif"][37510]
    # Add Homedeck tag
    exif["0th"][ImageIFD.Software] = "Homedeck AI"
    # Return
    return exif

def exifwrite (image: Image.Image, metadata: dict) -> Image.Image:
    """
    Write EXIF metadata to an image.

    Parameters:
        image (PIL.Image): Image to write metadata to.
        metadata (dict): Metadata dictionary.
    """
    image.info["exif"] = dump_exif(metadata)
    return image