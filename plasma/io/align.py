# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from cv2 import DescriptorMatcher_create, findHomography, ORB_create, warpPerspective, DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, RANSAC
from PIL import Image
from numpy import array, asarray
from typing import List, Union

def align_exposures (input: Union[Image.Image, List[Image.Image]], target: Image.Image, keypoints: int=1000) -> Union[Image.Image, List[Image.Image]]:
    """
    Align one or more images with a target image.

    Parameters:
        input (PIL.Image | list): Source image(s).
        target (PIL.Image): Target image.
        keypoints (int): Maximum number of keypoints to search for.

    Returns:
        PIL.Image | list: Aligned images.
    """
    # Check
    if not input:
        raise ValueError("Input is not a valid image")
    # Get source and target images
    input = input if isinstance(input, list) else [input]
    source = asarray(input[0])
    target = asarray(target)
    # Create detector
    height, width, _ = target.shape
    orb = ORB_create(keypoints)
    keypoints_a, descriptors_a = orb.detectAndCompute(source, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(target, None)
    # Match features
    matcher = DescriptorMatcher_create(DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_a, descriptors_b, None)
    matches = sorted(matches, key=lambda x: x.distance)
    top_k = int(len(matches) * 0.20)
    matches = matches[:top_k]
    # Extract correspondence points
    points_a = array([keypoints_a[match.queryIdx].pt for match in matches])
    points_b = array([keypoints_b[match.trainIdx].pt for match in matches])
    # Warp images
    H, mask = findHomography(points_a, points_b, RANSAC)
    results = []
    for image in input:
        image_array = asarray(image)
        warped_image = warpPerspective(image_array, H, (width, height))
        result = Image.fromarray(warped_image)
        result.info["exif"] = image.info.get("exif")
        results.append(result)
    # Return
    return results if len(results) > 1 else results[0]