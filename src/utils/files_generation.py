# ./src/utils/files_generation.py

import cv2
from typing import Callable, Mapping
from .label_generation import calculate_center, make_label
from .image_generation import make_image
from .common_type_aliases import (
    ShapeType, DrawPropsFunc, LabelParamsFunc
)


# Config dictionaries
DRAW_FUNCTIONS: Mapping[ShapeType, Callable[..., None]] = {
    "circle": cv2.circle,
    "square": cv2.rectangle,
}

DRAW_PROPS: Mapping[ShapeType, DrawPropsFunc] = {
    "circle": lambda x, y, size: {
        "center": (x, y),
        "radius": size,
        "color": (0, 0, 255),
        "thickness": -1,
    },
    "square": lambda x, y, size: {
        "pt1": (x, y),
        "pt2": (x + size, y + size),
        "color": (0, 0, 255),
        "thickness": -1,
    },
}

CLASS_IDS: Mapping[ShapeType, str] = {
    "circle": "0",
    "square": "1",
}

LABEL_PARAMS: Mapping[ShapeType, LabelParamsFunc] = {
    "circle": lambda x, y, size: (x, y, size * 2),
    "square": lambda x, y, size: (
        calculate_center(x, size),
        calculate_center(y, size),
        size,
    ),
}


# Core functions
def make_image_object(
    shape: ShapeType,
    x: int,
    y: int,
    size: int,
    filename: str,
    path_builder: Callable[[str], str],
) -> None:
    """
    Generate an image with a filled shape (circle or square).

    Args:
        shape: Type of shape ("circle" or "square").
        x: X center (circle) or top-left X (square) in pixels.
        y: Y center (circle) or top-left Y (square) in pixels.
        size: Radius (circle) or side length (square) in pixels.
        filename: Image file name.
        path_builder: Function that builds full file path.
    """
    draw_func = DRAW_FUNCTIONS[shape]
    draw_props = DRAW_PROPS[shape](x, y, size)
    make_image(draw_func=draw_func, draw_props=draw_props,
               filename=filename, path_builder=path_builder)


def make_label_object(
    shape: ShapeType,
    x: int,
    y: int,
    size: int,
    filename: str,
    path_builder: Callable[[str], str],
) -> None:
    """
    Create YOLO label for a shape (circle or square).

    Args:
        shape: Type of shape ("circle" or "square").
        x: X center (circle) or top-left X (square) in pixels.
        y: Y center (circle) or top-left Y (square) in pixels.
        size: Radius (circle) or side length (square) in pixels.
        filename: Label file name.
        path_builder: Function that builds full file path.
    """
    class_id = CLASS_IDS[shape]
    x_center, y_center, label_size = LABEL_PARAMS[shape](x, y, size)

    make_label(
        x_center=x_center,
        y_center=y_center,
        size=label_size,
        filename=filename,
        class_id=class_id,
        path_builder=path_builder,
    )
