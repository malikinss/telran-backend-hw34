# ./src/training.py

from pathlib import Path
from ultralytics import YOLO    # type:ignore

from .utils.common_constants import BASE_DIR


DATA_YAML = Path(BASE_DIR) / "data.yaml"
MODEL_NAME = "yolov8m.pt"


def train_model(
    *,
    epochs: int = 20,
    img_size: int = 256,
    batch: int = 2,
    experiment_name: str = "circle_exp",
) -> None:
    """
    Train YOLOv8 model on generated dataset.

    Args:
        epochs: Number of training epochs.
        img_size: Input image size.
        batch: Batch size.
        experiment_name: Name of the experiment folder.

    Returns:
        None
    """
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name=experiment_name,
    )
