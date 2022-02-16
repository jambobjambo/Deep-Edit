# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from lensfunpy import Database, Modifier
from numpy import ndarray
from PIL import Image
from torch import from_numpy, stack
from torch.nn.functional import grid_sample
from torchvision.transforms import ToPILImage, ToTensor

from .device import get_io_device

def lens_correction (*images: Image.Image) -> Image.Image:
    """
    Appply lens distortion correction on an image.
    This requires the images to have valid EXIF metadata tags.

    Parameters:
        images (PIL.Image | list): Input image(s) with EXIF metadata.

    Returns:
        PIL.Image | list: Corrected image(s).
    """
    # Check
    if len(images) == 0:
        return None
    # Compute sample grid
    grid = _compute_sample_grid(images[0])
    if grid is None:
        return images if len(images) > 1 else images[0]
    # Save EXIF
    exifs = [image.info.get("exif") for image in images]
    # Create exposure stack tensor
    device = get_io_device()
    image_tensors = [ToTensor()(image) for image in images]
    exposure_stack = stack(image_tensors, dim=0).to(device)
    # Create sampling grid tensor
    grid = from_numpy(grid).unsqueeze(dim=0).to(device)
    grid = grid.repeat(len(images), 1, 1, 1)
    # Sample
    result_stack = grid_sample(exposure_stack, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # Convert back to images
    exposures = result_stack.split(1, dim=0)
    images = [ToPILImage()(exposure.squeeze(dim=0).cpu()) for exposure in exposures]
    # Add EXIF and return
    for image, exif in zip(images, exifs):
        image.info["exif"] = exif
    return images if len(images) > 1 else images[0]

def _compute_sample_grid (image: Image.Image) -> ndarray:
    """
    Create a modifier for the camera and lens used to capture a given image.
    This function relies on the image's EXIF metadata to function.

    Parameters:
        image (PIL.Image): Input image.
        
    Returns:
        Modifier: Modifier for the camera and lens, or `None` if there is insufficient metadata.
    """
    # EXIF tags
    CAMERA_MAKER_EXIF_TAG = 271
    CAMERA_MODEL_EXIF_TAG = 272
    LENS_MAKER_EXIF_TAG = 42035
    LENS_MODEL_EXIF_TAG = 42036
    FOCAL_LENGTH_EXIF_TAG = 37386
    F_NUMBER_EXIF_TAG = 33437
    # Get metadata
    metadata = image.getexif()
    camera_maker, camera_model = metadata.get(CAMERA_MAKER_EXIF_TAG), metadata.get(CAMERA_MODEL_EXIF_TAG)
    lens_maker, lens_model = metadata.get(LENS_MAKER_EXIF_TAG), metadata.get(LENS_MODEL_EXIF_TAG)
    # Check
    if not all([camera_maker, camera_model, lens_model]):
        return None
    # Find model
    database = Database()
    cameras = database.find_cameras(camera_maker, camera_model)
    if len(cameras) == 0:
        return None
    lenses = database.find_lenses(cameras[0], lens_maker, lens_model)
    if len(lenses) == 0:
        return None
    # Get focal length and f number
    focal_length = metadata.get(FOCAL_LENGTH_EXIF_TAG, 20)
    f_number = metadata.get(F_NUMBER_EXIF_TAG, 8)
    focal_length = focal_length[0] / focal_length[1] if isinstance(focal_length, tuple) else focal_length
    f_number = f_number[0] / f_number[1] if isinstance(f_number, tuple) else f_number
    # Create modifier
    modifier = Modifier(lenses[0], cameras[0].crop_factor, image.width, image.height)
    modifier.initialize(focal_length, f_number)
    # Compute sample grid
    sample_grid = modifier.apply_geometry_distortion() # (H,W,2)
    # Normalize
    sample_grid[:,:,0] = 2. * sample_grid[:,:,0] / image.width - 1.
    sample_grid[:,:,1] = 2. * sample_grid[:,:,1] / image.height - 1.
    return sample_grid