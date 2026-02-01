# ./src/utils/training_structure.py

"""
Utilities for creating dataset directory structure and building
dataset file paths for images and labels.
"""

import os
from functools import partial
from typing import Iterable, Dict, Final
from .common_type_aliases import PathBuilder
from .common_constants import BASE_DIR, DATA_TYPES, SPLITS


# Core helpers
def dataset_file_path(*parts: str) -> str:
    """
    Build a full dataset file path relative to BASE_DIR.

    Args:
        *parts: Arbitrary path components (e.g. 'images', 'train', 'img1.jpg').

    Returns:
        Full path string.
    """
    return os.path.join(BASE_DIR, *parts)


def create_dataset_structure(
    data_types: Iterable[str],
    splits: Iterable[str],
) -> None:
    """
    Create dataset directories for all types and splits.

    Automatically creates directories like:
        BASE_DIR/images/train
        BASE_DIR/images/val
        BASE_DIR/labels/train
        BASE_DIR/labels/val

    Args:
        data_types: Dataset subdirectories (e.g. images, labels).
        splits: Dataset splits (e.g. train, val).

    Returns:
        None
    """
    for data_type in data_types:
        for split in splits:
            os.makedirs(dataset_file_path(data_type, split), exist_ok=True)


def build_all_paths() -> Dict[str, PathBuilder]:
    """
    Generate all pre-configured dataset path builders.

    Returns:
        A dictionary mapping keys like 'images_train', 'labels_val'
        to partial functions that accept a filename and return
        the full dataset path.
    """
    return {
        f"{data_type}_{split}": partial(dataset_file_path, data_type, split)
        for data_type in DATA_TYPES
        for split in SPLITS
    }


# Pre-configured path builders (DRY helpers)
paths: Final[Dict[str, PathBuilder]] = build_all_paths()
