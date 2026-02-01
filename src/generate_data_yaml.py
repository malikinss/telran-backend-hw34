# ./src/utils/generate_data_yaml.py

import yaml
from pathlib import Path
from .utils.common_constants import BASE_DIR, DATA_TYPES, SPLITS, SHAPES


def build_data_yaml() -> dict:
    """
    Build YOLO-compatible data.yaml structure.
    """
    return {
        "path": BASE_DIR,
        "train": f"{DATA_TYPES[0]}/{SPLITS[0]}",
        "val": f"{DATA_TYPES[0]}/{SPLITS[1]}",
        "names": dict(enumerate(SHAPES)),
    }


def generate_data_yaml() -> None:
    """
    Generate data.yaml file inside BASE_DIR.
    """
    output_path = Path(BASE_DIR) / "data.yaml"

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            build_data_yaml(),
            f,
            sort_keys=False,
        )
