import math
import logging
import cv2   as cv
import numpy as np
from PIL import Image

def convert_pil_to_cv(image):
    """
    Converts a PIL Image to a NumPy array compatible with OpenCV.

    Parameters:
        image (PIL.Image.Image): The input PIL Image.

    Returns:
        np.ndarray: The image in BGR format suitable for OpenCV.
    """
    if isinstance(image, Image.Image):
        # logging.info("Converting PIL Image to NumPy array.")
        image = np.array(image)
        
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        elif image.ndim == 2:
            logging.info("Image is already in grayscale format.")
        else:
            raise ValueError("Unsupported image format!")
    else:
        raise TypeError("Unsupported image type. Expected PIL Image.")
    
    return image

def scale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scales the input image by the given scale factor.

    Parameters:
        image (np.ndarray): The input image to be scaled.
        scale_factor (float): The factor by which to scale the image.

    Returns:
        np.ndarray: The scaled image.
    """
    # logging.info(f"Scaling image with factor of: {scale_factor}")
    #(width, height)
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))  
    scaled_image = cv.resize(image, new_size, interpolation=cv.INTER_AREA)
    return scaled_image

#function to convert euler angles to rotation matrix
def euler_to_matrix(roll, pitch, yaw, degrees=True):
    """Convert Euler angles (roll, pitch, yaw) to 3x3 rotation."""
    if degrees:
        roll  = math.radians(roll)
        pitch = math.radians(pitch)
        yaw   = math.radians(yaw)
    Rx = np.array([
        [1,           0,            0],
        [0,  math.cos(roll), -math.sin(roll)],
        [0,  math.sin(roll),  math.cos(roll)]
    ])
    Ry = np.array([
        [ math.cos(pitch),  0, math.sin(pitch)],
        [             0,    1,              0],
        [-math.sin(pitch),  0, math.cos(pitch)]
    ])
    Rz = np.array([
        [ math.cos(yaw), -math.sin(yaw), 0],
        [ math.sin(yaw),  math.cos(yaw), 0],
        [           0,               0, 1]
    ])
    # typical Rz * Ry * Rx
    return Rz @ Ry @ Rx
