# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from concurrent.futures import ThreadPoolExecutor
from cv2 import DescriptorMatcher_create, findHomography, ORB_create, DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, RANSAC
from numpy import array, asarray, ndarray, sqrt
from numpy.linalg import eig
from PIL import Image
from typing import Callable, List

from .utility import normalize_exposures

def feature_similarity (max_cost: float=1e-3) -> Callable[[Image.Image, Image.Image], bool]:
    """
    Create a feature-based similarity function.

    Parameters:
        max_cost (float). Maximum alignment cost for images to be considered similar.

    Returns:
        callable: Pairwise similarity function returning a boolean.
    """
    def similarity_fn (image_a: Image.Image, image_b: Image.Image) -> bool:
        # Compute matches
        image_a, image_b = normalize_exposures(image_a, image_b)
        keypoints_a, keypoints_b, matches = _compute_matches(image_a, image_b)
        # Compute alignment cost
        cost = _compute_alignment_cost(keypoints_a, keypoints_b, matches)
        return cost <= max_cost
    return similarity_fn

def _compute_matches (image_a: Image.Image, image_b: Image.Image):
    """
    Compute feature matches between two images.

    Parameters:
        image_a (PIL.Image): First image.
        image_b (PIL.Image): Second image.

    Returns:
        tuple: Keypoints from first image, keypoints from second image, and matches between them.
    """
    # Normalize
    image_a, image_b = asarray(image_a), asarray(image_b)
    # Detect features
    orb = ORB_create(1000)
    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)
    # Match features
    matcher = DescriptorMatcher_create(DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_a, descriptors_b, None)
    # Extract top k
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.15)]
    # Return
    return keypoints_a, keypoints_b, matches

def _compute_alignment_cost (keypoints_a: ndarray, keypoints_b: ndarray, matches: ndarray) -> float:
    """
    Compute the alignment cost between two sets of keypoints.

    Parameters:
        keypoints_a (ndarray): First set of keypoints.
        keypoints_b (ndarray): Second set of keypoints.
        matches (ndarray): Matches between keypoints.

    Returns:
        float: Alignment cost.
    """
    # Compute homography
    points_a = array([keypoints_a[match.queryIdx].pt for match in matches])
    points_b = array([keypoints_b[match.trainIdx].pt for match in matches])
    H, _ = findHomography(points_a, points_b, RANSAC)
    # Compute alignment cost
    singular_values, _ = eig(H.T * H)
    induced_norm = sqrt(singular_values.max())
    cost = abs(induced_norm - 1.)
    # Return
    return cost