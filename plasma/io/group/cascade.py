# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from PIL import Image
from typing import Callable

def cascade_similarity (*similarity_fns) -> Callable[[Image.Image, Image.Image], bool]:
    """
    Create a cascade of similarity functions.

    The similarity functions are applied in sequence until the images are successfully grouped.
    If no similarity function returns `True`, then the images will not be grouped.

    Parameters:
        similarity_fns (list): Individual similarity functions to apply in sequence.
    
    Returns:
        callable: Pairwise similarity function returning a boolean.
    """
    def similarity_fn (image_a: Image.Image, image_b: Image.Image) -> bool:
        return any([fn(image_a, image_b) for fn in similarity_fns])
    return similarity_fn