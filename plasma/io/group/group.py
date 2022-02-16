# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import Callable, List

from .timestamp import exposure_timestamp
from .utility import load_exposure

def group_exposures (exposure_paths: List[str], similarity_fn: Callable[[Image.Image, Image.Image], bool], workers: int=8) -> List[List[str]]:
    """
    Group a set of exposures using a similarity function.

    Parameters:
        exposure_paths (list): Paths to exposures to group.
        similarity_fn (callable): Pairwise similarity function returning a boolean.
        workers (int): Number of workers for IO.
    
    Returns:
        list: Groups of exposure paths.
    """
    # Check
    if not exposure_paths:
        return []
    # Trivial case
    if len(exposure_paths) == 1:
        return [exposure_paths]
    # Load all exposures into memory
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Sort by timestamp
        exposures_with_paths = executor.map(lambda path: (path, load_exposure(path)), exposure_paths)
        exposures_with_paths = sorted(exposures_with_paths, key=lambda pair: exposure_timestamp(pair[1]))
        exposure_paths, exposures = list(zip(*exposures_with_paths))
        # Compute pairwise similarity
        pairwise_similarities = executor.map(lambda pair: similarity_fn(*pair), [(exposures[i], exposures[i+1]) for i in range(len(exposures) - 1)])
        pairwise_similarities = list(pairwise_similarities) # So we can see it when debugging
    # Group
    groups = []
    current_group = [exposure_paths[0]]
    for i, similar in enumerate(pairwise_similarities):
        if not similar:
            groups.append(current_group)
            current_group = []
        current_group.append(exposure_paths[i+1])
    groups.append(current_group)
    return groups