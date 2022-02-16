# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from numpy import allclose, array
from pathlib import Path
from PIL import Image
from pkg_resources import resource_filename
from rawpy import imread, DemosaicAlgorithm, HighlightMode, Params
from rawpy.enhance import find_bad_pixels, repair_bad_pixels
from torch import cat, stack
from torchvision.transforms import ToTensor, ToPILImage

from ..conversion import rgb_to_yuv, yuv_to_rgb
from ..sampling import bilateral_filter_2d, color_sample_1d, lutread
from .device import get_io_device
from .metadata import exifread, exifwrite

def rawread (*image_paths: str) -> Image.Image:
    """
    Load one or more RAW images.

    Parameters:
        image_paths (str | list): Path(s) to image to be loaded.

    Returns:
        PIL.Image | list: Loaded image(s).
    """
    # Check
    if len(image_paths) == 0:
        return None
    # Find bad pixels
    bad_pixels = find_bad_pixels(image_paths)
    # Load
    exposures, metadatas = [], []
    for image_path in image_paths:
        with imread(image_path) as raw:
            # Repair bad pixels
            repair_bad_pixels(raw, bad_pixels, method="median")
            # Compute saturation level
            white_level = array(raw.camera_white_level_per_channel).min()
            saturation_level = raw.white_level if white_level == 0 else white_level
            # Demosaic
            params = Params(
                demosaic_algorithm=DemosaicAlgorithm.AHD,
                use_camera_wb=True,
                no_auto_bright=True,
                user_sat=saturation_level,
                output_bps=8,
                highlight_mode=HighlightMode.Clip,
                gamma=(1, 1)
            )
            rgb = raw.postprocess(params=params)
            exposure = Image.fromarray(rgb)
            exposures.append(exposure)
            # Load metadata
            metadata = exifread(image_path)
            metadatas.append(metadata)
    # Gamma correction
    device = get_io_device()
    exposures = [ToTensor()(exposure) for exposure in exposures]
    exposure_stack = stack(exposures, dim=0).to(device)
    exposure_stack = 2. * exposure_stack.pow(1. / 2.2) - 1.
    # Chromaticity noise reduction
    yuv = rgb_to_yuv(exposure_stack)
    y, u, v = yuv.split(1, dim=1)
    u = bilateral_filter_2d(u, (3, 5), grid_size=(8, 1024, 1024))
    v = bilateral_filter_2d(v, (3, 5), grid_size=(8, 1024, 1024))
    yuv = cat([y, u, v], dim=1)
    exposure_stack = yuv_to_rgb(yuv)
    # Tone curve
    tone_curve_path = resource_filename("plasma.io", "data/raw_standard_med.tif")
    tone_curve = lutread(tone_curve_path)
    exposure_stack = color_sample_1d(exposure_stack, tone_curve)
    exposure_stack = (exposure_stack + 1.) / 2.
    # Convert back to PIL
    exposure_stack = exposure_stack.cpu()
    exposures = exposure_stack.split(1, dim=0)
    exposures = [ToPILImage()(exposure.squeeze(dim=0)) for exposure in exposures]
    # Add EXIF metadata
    exposures = [exifwrite(exposure, metadata) for exposure, metadata in zip(exposures, metadatas)]
    return exposures if len(exposures) > 1 else exposures[0]

def is_raw_format (image_path: str) -> bool:
    """
    Is the file at the provided path a RAW image?

    Parameters:
        image_path (str): Path to file.

    Returns:
        bool: Whether the file is a RAW image.
    """
    RAW_FORMATS = [".arw", ".cr2", ".crw", ".dng", ".nef", ".srf"]
    format = Path(image_path).suffix.lower()
    return format in RAW_FORMATS