import numpy as np
from numpy.lib import stride_tricks
import color_space_test as color_space

def random_crop(img, size):
    """
    Perform a random crop of the given size on the image.

    Parameters:
    img (numpy.ndarray): Input image as a numpy array.
    size (int): Size of the crop.

    Returns:
    numpy.ndarray: Cropped image.
    """
    img_height, img_width = img.shape[:2]
    
    if size > min(img_height, img_width):
        raise ValueError("Crop size is larger than the smallest dimension of the image.")
    
    # Calculate the maximum possible starting positions for the crop
    max_start_y = img_height - size
    max_start_x = img_width - size
    
    # Generate random starting positions for the crop
    start_y = np.random.randint(0, max_start_y + 1)
    start_x = np.random.randint(0, max_start_x + 1)
    
    # Crop the image
    cropped_img = img[start_y:start_y + size, start_x:start_x + size]
    
    return cropped_img

def extract_patches(img, num_patches):
    """
    Extract n^2 non-overlapping patches from a square image.
    
    Parameters:
    img (numpy.ndarray): Input image as a numpy array.
    num_patches (int): Number of patches along one dimension.

    Returns:
    numpy.ndarray: Array of patches.
    """
    H, W = img.shape[:2]  # Assuming the input image is square
    patch_size = H // num_patches  # Calculate patch size

    if H % num_patches != 0 or W % num_patches != 0:
        raise ValueError("Image dimensions must be divisible by the number of patches.")
    
    if len(img.shape) == 3:
        # Color image (RGB)
        shape = (num_patches, num_patches, patch_size, patch_size, img.shape[2])
        strides = (patch_size * img.strides[0], patch_size * img.strides[1], *img.strides)
    else:
        # Grayscale image
        shape = (num_patches, num_patches, patch_size, patch_size)
        strides = (patch_size * img.strides[0], patch_size * img.strides[1], *img.strides)

    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    
    return patches.reshape(-1, patch_size, patch_size, img.shape[2]) if len(img.shape) == 3 else patches.reshape(-1, patch_size, patch_size)

import numpy as np

def resize_img(img, factor):
    """
    Resize an image using nearest neighbor interpolation.

    Parameters:
    img (numpy.ndarray): Input image as a numpy array.
    factor (float): Scale factor for resizing the image.

    Returns:
    numpy.ndarray: Resized image.
    """
    # Get the original dimensions
    H, W = img.shape[:2]

    # Calculate the new dimensions
    new_H, new_W = int(H * factor), int(W * factor)

    # Generate the grid of coordinates in the new image
    row_indices = (np.arange(new_H) / factor).astype(int)
    col_indices = (np.arange(new_W) / factor).astype(int)

    # Clip indices to ensure they are within the valid range
    row_indices = np.clip(row_indices, 0, H - 1)
    col_indices = np.clip(col_indices, 0, W - 1)

    # Use broadcasting to create a 2D grid of coordinates
    row_indices = row_indices[:, np.newaxis]
    col_indices = col_indices[np.newaxis, :]

    # Apply nearest neighbor interpolation
    if len(img.shape) == 3:
        new_img = img[row_indices, col_indices, :]
    else:
        new_img = img[row_indices, col_indices]

    return new_img


def color_jitter(img, hue, saturation, value):
    """
    Perturb the HSV values of an input image by a random amount up to the specified limits.

    Parameters:
    img (numpy.ndarray): Input image as a numpy array.
    hue (float): Maximum amount to perturb the hue channel.
    saturation (float): Maximum amount to perturb the saturation channel.
    value (float): Maximum amount to perturb the value channel.

    Returns:
    numpy.ndarray: Image with perturbed HSV values.
    """
    # Convert RGB image to HSV
    hsv_img = color_space.rgb2hsv(img)

    # Generate random perturbation values
    hue_perturbation = np.random.uniform(-hue, hue)
    saturation_perturbation = np.random.uniform(-saturation, saturation)
    value_perturbation = np.random.uniform(-value, value)

    # Modify the HSV image
    hsv_img[:, :, 0] = np.clip(hsv_img[:, :, 0] + hue_perturbation, 0, 1)
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] + saturation_perturbation, 0, 1)
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + value_perturbation, 0, 1)

    # Convert HSV image back to RGB
    modified_img = color_space.hsv2rgb(hsv_img)

    # Scale the RGB values back to [0, 255]
    modified_img = (modified_img * 255).astype(np.uint8)

    return modified_img



