# ./src/utils/common_constants.py

"""
Common constants used across dataset generation and image processing modules.
"""

from typing import Final, Tuple
from .common_type_aliases import DataType, SplitType, ShapeType

# Dataset structure
BASE_DIR: Final[str] = "datasets"
DATA_TYPES: Final[Tuple[DataType, ...]] = ("images", "labels")
SPLITS: Final[Tuple[SplitType, ...]] = ("train", "val")
SHAPES: Final[Tuple[ShapeType, ...]] = ("circle", "square")

# Image dimensions
IMAGE_WIDTH: Final[int] = 256
IMAGE_HEIGHT: Final[int] = 256
