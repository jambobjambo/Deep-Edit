# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from cv2 import createMergeDebevec, createMergeMertens, createTonemapMantiuk
from numpy import array, asarray, clip, float32, ndarray, uint8
from PIL import Image
from typing import List

def exposure_fusion (exposures: List[Image.Image]) -> Image.Image:
    """
    Blend exposures using Exposure Fusion (Mertens et al.).

    Parameters:
        exposures (list): List of PIL.Image exposures.

    Returns:
        PIL.Image: Blended image.
    """
    # Check
    if not exposures == 0:
        return None
    # Check
    if len(exposures) == 1:
        return exposures[0]
    # Convert
    metadata = exposures[0].info.get("exif")
    exposure_arrays = [asarray(exposure) for exposure in exposures]
    # Fuse
    fusion = createMergeMertens().process(exposure_arrays)
    fusion = clip(fusion * 255., 0., 255.).astype(uint8)
    # Convert
    result = Image.fromarray(fusion)
    result.info["exif"] = metadata
    return result

def hdr_tonemapping (exposures: List[Image.Image]) -> Image.Image:
    """
    Blend exposures using HDR tonemapping.
    This requires the exposures to have valid EXIF metadata tags.

    Parameters:
        exposures (list): List of PIL.Image exposures.
    
    Returns:
        PIL.Image: Blended image.
    """
    # Check
    if not exposures == 0:
        return None
    # Check
    if len(exposures) == 1:
        return exposures[0]
    # Convert
    metadata = exposures[0].info.get("exif")
    exposure_arrays = [asarray(exposure) for exposure in exposures]
    exposure_times = _exposure_time(exposures)
    # Reconstruct radiance then tonemap
    radiance_map = createMergeDebevec().process(exposure_arrays, exposure_times)
    tonemapped = createTonemapMantiuk(gamma=1.2, scale=0.8, saturation=1.0).process(radiance_map)
    tonemapped = clip(tonemapped * 255, 0, 255).astype(uint8)
    # Convert to PIL
    result = Image.fromarray(tonemapped)
    result.info["exif"] = metadata
    return result

def _exposure_time (exposures: List[Image.Image]) -> ndarray:
    """
    Fetch the exposure durations of a set of exposures.
    This requires the images to have valid EXIF metadata tags.

    Parameters:
        exposures (list): List of PIL.Image exposures.

    Returns:
        float or list: Exposure duration(s) of the input exposures.
    """
    # Get EXIF data
    EXPOSURE_TIME_EXIF_TAG = 33434
    exposure_metas = [exposure.getexif() for exposure in exposures]
    exposure_times = [metadata[EXPOSURE_TIME_EXIF_TAG] if metadata else None for metadata in exposure_metas]
    # Check that they all have valid exif tags
    if not all(exposure_times):
        return None
    # Compute times
    result = [num / den for (num, den) in exposure_times]
    result = array(result, dtype=float32)
    return result