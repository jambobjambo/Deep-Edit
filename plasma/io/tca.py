# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from cv2 import cornerHarris, findTransformECC, MOTION_TRANSLATION, TERM_CRITERIA_COUNT, TERM_CRITERIA_EPS
from numpy import array, argpartition, asarray, column_stack, concatenate, eye, float32, ndarray, ones, split, stack as stack_array, unravel_index
from numpy.linalg import norm
from PIL import Image
from sklearn.linear_model import LinearRegression
from torch import cat, linspace, meshgrid, stack, Tensor
from torch.nn.functional import grid_sample
from torchvision.transforms import ToPILImage, ToTensor
from typing import Tuple

from .device import get_io_device

def tca_correction (*images: Image.Image) -> Image.Image:
    """
    Appply transverse chromatic aberration correction on an image.

    Parameters:
        images (PIL.Image | list): Input image(s).
    
    Returns:
        PIL.Image | list: Corrected image(s).
    """
    # Check
    if len(images) == 0:
        return None
    # Save EXIF
    exifs = [image.info.get("exif") for image in images]
    # Create exposure stack tensor
    device = get_io_device()
    exposure_stack = stack([ToTensor()(image) for image in images], dim=0).to(device)
    # Correct
    red_coeffs, blue_coeffs = _compute_coefficients(images[0])
    result_stack = _tca_forward(exposure_stack, red_coeffs, blue_coeffs)
    # Convert back to images
    exposures = result_stack.split(1, dim=0)
    images = [ToPILImage()(exposure.squeeze(dim=0).cpu()) for exposure in exposures]
    # Add EXIF and return
    for image, exif in zip(images, exifs):
        image.info["exif"] = exif
    return images if len(images) > 1 else images[0]
    
def _compute_coefficients (image: Image.Image) -> Tuple[ndarray, ndarray]:
    """
    Compute TCA correction coefficients.

    We use a cubic lens distortion model.

    Parameters:
        image (PIL.Image): Input image.

    Returns:
        tuple: Red and blue channel correction coefficients, each with shape (4,)
    """
    # Compute displacements
    image_array = asarray(image)
    height, width, _ = image_array.shape
    # Extract patches to inspect
    corners = _find_corners(image_array, count=100)
    patches, centers = _extract_patches(image_array, corners, size=100)
    # Compute displacements
    displacements, mask = _compute_patch_displacements(patches)
    displacements_red, displacements_blue = displacements[:,0], displacements[:,1]
    # Compute radial field
    image_size = array([ width, height ])
    image_center = image_size / 2.
    patch_radii = norm((centers[mask] - image_center) / image_center, axis=1)
    displaced_radii_red = norm((centers[mask] - displacements_red - image_center) / image_center, axis=1)
    displaced_radii_blue = norm((centers[mask] - displacements_blue - image_center) / image_center, axis=1)
    # Compute coefficients
    regressor_red = LinearRegression(fit_intercept=False)
    regressor_blue = LinearRegression(fit_intercept=False)
    X = stack_array([patch_radii, patch_radii ** 2, patch_radii ** 3, patch_radii ** 4], axis=1)
    regressor_red.fit(X, displaced_radii_red)
    regressor_blue.fit(X, displaced_radii_blue)
    return regressor_red.coef_, regressor_blue.coef_

def _tca_forward (input: Tensor, red_coeffs: ndarray, blue_coeffs: ndarray) -> Tensor:
    """
    Apply the cubic TCA correction forward model.

    Parameters:
        input (Tensor): Image stack with shape (N,3,H,W).
        red_coeffs (ndarray): Red channel correction coefficients with shape (4,).
        blue_coeffs (ndarray): Blue channel correction coefficients with shape (4,).

    Returns:
        Tensor: Corrected image stack with shape (N,3,H,W).
    """
    # Construct sample grid
    batch, _, height, width = input.shape
    hg, wg = meshgrid(linspace(-1., 1., height), linspace(-1., 1., width))
    hg = hg.repeat(batch, 1, 1).unsqueeze(dim=3).to(input.device)
    wg = wg.repeat(batch, 1, 1).unsqueeze(dim=3).to(input.device)
    sample_field = cat([wg, hg], dim=3)
    r_dst = sample_field.norm(dim=3, keepdim=True)
    # Compute distortions
    r_a, r_b, r_c, r_d = red_coeffs
    b_a, b_b, b_c, b_d = blue_coeffs
    red_distortion = r_a + r_b * r_dst.pow(1) + r_c * r_dst.pow(2) + r_d * r_dst.pow(3)
    blue_distortion = b_a + b_b * r_dst.pow(1) + b_c * r_dst.pow(2) + b_d * r_dst.pow(3)
    # Compute sample grids
    red_grid = sample_field * red_distortion
    blue_grid = sample_field * blue_distortion
    # Sample
    red, green, blue = input.split(1, dim=1)
    red_shifted = grid_sample(red, red_grid, mode="bilinear", padding_mode="border", align_corners=False)
    blue_shifted = grid_sample(blue, blue_grid, mode="bilinear", padding_mode="border", align_corners=False)
    # Combine
    result = cat([red_shifted, green, blue_shifted], dim=1)
    return result

def _find_corners (input: ndarray, count: int=100) -> ndarray:
    """
    Find corners in an image.

    Parameters:
        input (ndarray): Input image with shape (H,W,3).
        count (int): Maximum number of corners to return.

    Returns:
        ndarray: Coordinates of corners with shape (N,2).
    """
    # Find corners in green channel
    _, g, _ = split(input, 3, axis=2)
    corners = cornerHarris(g.astype(float32), 2, 3, 0.04)
    # Get coordinates with max response
    corner_indices = argpartition(corners, -count, axis=None)[-count:]
    y_coords, x_coords = unravel_index(corner_indices, corners.shape)
    # Return
    coords = column_stack([x_coords, y_coords])
    return coords

def _extract_patches (input: ndarray, centers: ndarray, size: int=100) -> Tuple[ndarray, ndarray]:
    """
    Extract image patches centered around a set of patches.
    
    Note that the number of returned patches might be less than N, as patches that are not full-size are discarded.

    Parameters:
        input (ndarray): Input image with shape (H,W,3).
        centers (ndarray): Patch centers (x,y) coordinates with shape (N,2).
        size (int): Size in each dimension.

    Returns:
        tuple: Patch stack with shape (M,S,S,3) and patch centers with shape (M,2).
    """
    negatives = centers - size // 2
    patches = [input[y_min:y_max, x_min:x_max] for x_min, y_min, x_max, y_max in concatenate([negatives, negatives + size], axis=1)]
    patches = [(patch, center) for patch, center in zip(patches, centers) if patch.shape[0] == patch.shape[1] == size]
    patches, centers = zip(*patches)
    patches, centers = stack_array(patches), stack_array(centers)
    return patches, centers

def _compute_patch_displacements (patches: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Compute per-patch alignment displacements for N patches.

    Note that the number of returned displacements might be less than N.
    This happens when no suitable displacement can be computed for a given patch.

    Parameters:
        patches (ndarray): Patch stack with shape (N,S,S,3).

    Returns:
        tuple: Red and blue channel displacement vectors with shape (M,2,2), selection mask with shape (N,).
    """
    # Constants
    IDENTITY = eye(2, 3, dtype=float32)
    CRITERIA = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 100, 1e-4)
    # Compute
    displacements = []
    mask = ones(patches.shape[0]).astype(bool)
    for i, patch in enumerate(patches):
        try:
            patch_r, patch_g, patch_b = split(patch, 3, axis=2)
            _, warp_matrix_r = findTransformECC(patch_g, patch_r, IDENTITY.copy(), MOTION_TRANSLATION, CRITERIA, None, 5)
            _, warp_matrix_b = findTransformECC(patch_g, patch_b, IDENTITY.copy(), MOTION_TRANSLATION, CRITERIA, None, 5)
            displacement = -stack_array([warp_matrix_r[:,2], warp_matrix_b[:,2]], axis=0) # invert displacement
            displacements.append(displacement)
        except:
            mask[i] = False
    # Return
    displacements = stack_array(displacements)
    return displacements, mask