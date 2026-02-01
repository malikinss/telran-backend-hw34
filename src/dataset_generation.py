# ./src/dataset_generation.py

import random
from typing import Tuple, List

from .utils.common_type_aliases import (
    ShapeType, SplitType,
    PathBuilder,
    Params, ParamsGenerator
)
from .utils.common_constants import (
    DATA_TYPES, SPLITS, IMAGE_WIDTH, IMAGE_HEIGHT
)
from .utils.training_structure import create_dataset_structure, paths
from .utils.files_generation import make_image_object, make_label_object


# Create base dataset directories
create_dataset_structure(DATA_TYPES, SPLITS)


# Random parameter generators
def random_circle_params() -> Params:
    """
    Generate random (x_center, y_center, radius) for a circle.
    """
    radius = random.randint(10, 50)
    x_center = random.randint(radius, IMAGE_WIDTH - radius)
    y_center = random.randint(radius, IMAGE_HEIGHT - radius)
    return x_center, y_center, radius


def random_square_params() -> Params:
    """
    Generate random (x, y, size) for a square.
    """
    size = random.randint(10, 50)
    x = random.randint(0, IMAGE_WIDTH - size)
    y = random.randint(0, IMAGE_HEIGHT - size)
    return x, y, size


RANDOM_PARAMS: dict[ShapeType, ParamsGenerator] = {
    "circle": random_circle_params,
    "square": random_square_params,
}

# Dataset generation configuration
DATA_CONFIG: List[Tuple[ShapeType, SplitType, int]] = [
    ("circle", "train", 30),
    ("square", "train", 30),
    ("circle", "val", 3),
    ("square", "val", 3),
]


def create_dataset():
    """
    Generate images and labels for all configured shapes and splits.
    """
    for shape, split, count in DATA_CONFIG:
        img_path_builder: PathBuilder = paths[f"images_{split}"]
        label_path_builder: PathBuilder = paths[f"labels_{split}"]
        param_generator = RANDOM_PARAMS[shape]

        for i in range(count):
            x, y, size = param_generator()
            filename_base = f"{shape}_{split}_{i}"

            make_image_object(
                shape=shape,
                x=x,
                y=y,
                size=size,
                filename=f"{filename_base}.jpg",
                path_builder=img_path_builder,
            )
            make_label_object(
                shape=shape,
                x=x,
                y=y,
                size=size,
                filename=f"{filename_base}.txt",
                path_builder=label_path_builder,
            )
