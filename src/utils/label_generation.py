# ./src/utils/label_generation.py

"""
Utilities for creating YOLO-compatible label files.

Includes functions for:
- normalizing bounding boxes
- formatting values
- writing label files to disk
"""

from typing import Sequence, Callable, Final
from .common_constants import IMAGE_HEIGHT, IMAGE_WIDTH


def float_to_str(value: float) -> str:
    """
    Format a float value for YOLO label files.

    Args:
        value: Floating-point value to format.

    Returns:
        String representation with 6 decimal places.
    """
    return f"{value:.6f}"


def calculate_center(start: float, size: float) -> float:
    """
    Calculate center coordinate from top-left position and size.

    Args:
        start: Starting coordinate (x or y).
        size: Object size (width or height).

    Returns:
        Center coordinate.
    """
    return start + size / 2


def normalize_bbox(
    x_center: float,
    y_center: float,
    size: float,
) -> list[float]:
    """
    Normalize bounding box values for YOLO format.

    Args:
        x_center: X center in pixels.
        y_center: Y center in pixels.
        size: Width/height in pixels (assumes square bounding box).

    Returns:
        List of normalized [x_center, y_center, width, height].
    """
    return [
        x_center / IMAGE_WIDTH,
        y_center / IMAGE_HEIGHT,
        size / IMAGE_WIDTH,
        size / IMAGE_HEIGHT,
    ]


def format_label_values(values: Sequence[float]) -> str:
    """
    Convert normalized values into YOLO-compatible string.

    Args:
        values: Sequence of normalized float values.

    Returns:
        Space-separated string of formatted values.
    """
    return " ".join(map(float_to_str, values))


def write_label_file(
    path_builder: Callable[[str], str],
    filename: str,
    class_id: str,
    label_data: str,
) -> None:
    """
    Write YOLO label data to a file.

    Args:
        path_builder: Function that builds full file path.
        filename: Label file name.
        class_id: YOLO class ID.
        label_data: Normalized label data string.

    Returns:
        None
    """
    full_path: Final[str] = path_builder(filename)
    with open(full_path, "w", encoding="utf-8") as file:
        file.write(f"{class_id} {label_data}")


def make_label(
    *,
    x_center: float,
    y_center: float,
    size: float,
    filename: str,
    class_id: str,
    path_builder: Callable[[str], str],
) -> None:
    """
    Create and write a YOLO label file.

    Args:
        x_center: X center in pixels.
        y_center: Y center in pixels.
        size: Object size in pixels.
        filename: Label file name.
        class_id: YOLO class ID.
        path_builder: Function that builds full file path.

    Returns:
        None
    """
    normalized_values: Final[list[float]] = normalize_bbox(
        x_center=x_center,
        y_center=y_center,
        size=size,
    )
    label_string: Final[str] = format_label_values(normalized_values)

    write_label_file(
        path_builder=path_builder,
        filename=filename,
        class_id=class_id,
        label_data=label_string,
    )
