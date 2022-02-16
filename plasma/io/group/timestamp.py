# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from dateutil.parser import parse as parse_datetime
from PIL import Image
from typing import Callable

def timestamp_similarity (max_delta_time: float=4.) -> Callable[[Image.Image, Image.Image], bool]:
    """
    Create a timestamp-based similarity function.

    Parameters:
        max_delta_time (float): Maximum exposure time difference for images to be considered similar, in seconds.

    Returns:
        callable: Pairwise similarity function returning a boolean.
    """
    def similarity_fn (image_a: Image.Image, image_b: Image.Image) -> bool:
        timestamp_a = exposure_timestamp(image_a)
        timestamp_b = exposure_timestamp(image_b)
        delta_time = abs(timestamp_a - timestamp_b)
        return delta_time <= max_delta_time
    return similarity_fn

def exposure_timestamp (image: Image.Image) -> float:
    """
    Get the exposure timestamp from its EXIF metadata.

    If the required EXIF dictionary or tag is not present, `-1` will be returned.
    
    Parameters:
        image (PIL.Image): Exposure.
    
    Returns:
        float: Image timestamp.
    """
    DATETIME_ORIGINAL = 36867
    timestamp = image.getexif().get(DATETIME_ORIGINAL)
    if timestamp:
        timestamp = str(timestamp)
        datetime = parse_datetime(timestamp)
        return datetime.timestamp()
    else:
        return -1