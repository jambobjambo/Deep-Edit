# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from dateutil.parser import parse as parse_datetime
from exifread import process_file
from io import BytesIO
from PIL import Image
from numpy import asarray, float64, interp, ndarray, uint8, unique
from rawpy import imread as rawread, HighlightMode, Params, ThumbFormat
from typing import Tuple

from ..metadata import exifread, exifwrite
from ..raster import is_raster_format
from ..raw import is_raw_format

def load_exposure (image_path: str) -> Image.Image:
    """
    Load an exposure into memory.

    For RAW files, this function will try to load the thumbnail.
    If no thumbnail is available, the RAW is fully demosaiced.
    All metadata is loaded with the file if present.
    
    Parameters:
        image_path (str): Path to exposure.
    
    Returns:
        PIL.Image: Loaded exposure.
    """
    # If raster format, load directly
    if is_raster_format(image_path):
        image = Image.open(image_path)
    # If RAW, check for thumbnail
    elif is_raw_format(image_path):
        with rawread(image_path) as raw:
            # Load thumbnail
            try:
                thumb = raw.extract_thumb()
                if thumb.format == ThumbFormat.JPEG:
                    thumb_data = BytesIO(thumb.data)
                    image = Image.open(thumb_data)
                elif thumb.format == ThumbFormat.BITMAP:
                    image = Image.fromarray(thumb.data)
            # Demosaic RAW
            except:
                params = Params(
                    half_size=True,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    user_sat=11000,
                    exp_shift=1.,
                    exp_preserve_highlights=1.,
                    highlight_mode=HighlightMode.Clip,
                )
                image = raw.postprocess(params=params)
                image = Image.fromarray(image)
        # Append metadata
        metadata = exifread(image_path)
        exifwrite(image, metadata)
    return image

def normalize_exposures (image_a: Image.Image, image_b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Normalize two exposures to have similar histograms.

    This method preserves metadata on the output images.
    
    Parameters:
        image_a (PIL.Image): First image.
        image_b (PIL.Image): Second image.
    
    Returns:
        tuple: Normalized exposures.
    """
    # Convert to array
    image_a_arr = asarray(image_a)
    image_b_arr = asarray(image_b)
    image_a_meta = image_a.info.get("exif")
    image_b_meta = image_b.info.get("exif")
    # Match histograms
    std_a = image_a_arr.std()
    std_b = image_b_arr.std()
    input, target = (image_a_arr, image_b_arr) if std_a < std_b else (image_b_arr, image_a_arr)
    matched = _match_histogram(input, target)
    result_a = Image.fromarray(matched)
    result_b = Image.fromarray(target)
    # Apply metadata
    result_a.info["exif"] = image_a_meta if std_a < std_b else image_b_meta
    result_b.info["exif"] = image_b_meta if std_a < std_b else image_a_meta
    return result_a, result_b

def _match_histogram (input: ndarray, target: ndarray) -> ndarray:
    """
    Match the histogram of an input image to that of a target image.
    
    Parameters:
        input (ndarray): Input image.
        target (ndarray): Target image.
    
    Returns:
        ndarray: Histogram-matched input image.
    """
    # Source: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    s_values, bin_idx, s_counts = unique(input.ravel(), return_inverse=True, return_counts=True)
    t_values, t_counts = unique(target.ravel(), return_counts=True)
    s_quantiles = s_counts.cumsum().astype(float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = t_counts.cumsum().astype(float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = interp(s_quantiles, t_quantiles, t_values)
    result = interp_t_values[bin_idx].reshape(input.shape).astype(uint8)
    return result