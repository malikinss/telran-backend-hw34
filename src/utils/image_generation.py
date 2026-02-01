# ./src/utils/image_generation.py

"""
Utilities for generating synthetic images with simple geometric shapes.

The module follows DRY and KISS principles:
- image creation logic is centralized
- shape-specific functions only define drawing parameters
- file path construction is delegated to a path builder
"""

import cv2
import numpy as np
from .common_constants import IMAGE_HEIGHT, IMAGE_WIDTH
from .common_type_aliases import DrawProps, PathBuilder, DrawFunc


def make_image_template() -> np.ndarray:
    """
    Create an empty RGB image template.

    Returns:
        np.ndarray: Zero-filled RGB image of predefined size
                    (IMAGE_HEIGHT, IMAGE_WIDTH, 3).
    """
    return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)


def save_image(
    image: np.ndarray, filename: str, path_builder: PathBuilder
) -> None:
    """
    Save an image to disk using the provided path builder.

    Args:
        image (np.ndarray): Image array to save.
        filename (str): Image file name.
        path_builder (PathBuilder): Function that returns the full file path
                                    for saving.

    Returns:
        None
    """
    cv2.imwrite(path_builder(filename), image)


def make_image(
    draw_func: DrawFunc,
    draw_props: DrawProps,
    filename: str,
    path_builder: PathBuilder,
) -> None:
    """
    Create an image, apply a drawing function, and save to disk.

    Args:
        draw_func (DrawFunc): OpenCV drawing function
                              (e.g., cv2.circle or cv2.rectangle).
        draw_props (DrawProps): Keyword arguments for the drawing function.
        filename (str): Name of the file to save.
        path_builder (PathBuilder): Function that builds the full file path.

    Returns:
        None
    """
    image = make_image_template()
    draw_func(image, **draw_props)
    save_image(image, filename, path_builder)
