# ./src/utils/common_type_aliases.py

from typing import (
    Callable,
    Literal,
    Mapping,
    Any,
    Tuple,
)

# =========================
# Dataset domain types
# =========================

ShapeType = Literal["circle", "square"]
SplitType = Literal["train", "val"]
DataType = Literal["images", "labels"]

# =========================
# Image drawing types
# =========================

DrawProps = Mapping[str, Any]
DrawPropsFunc = Callable[[int, int, int], DrawProps]
DrawFunc = Callable[..., Any]

# =========================
# Label generation types
# =========================

LabelParamsFunc = Callable[[int, int, int], Tuple[float, float, int]]

# =========================
# Filesystem helpers
# =========================

PathBuilder = Callable[[str], str]

# =========================
# Dataset generation helpers
# =========================

Params = Tuple[int, int, int]
ParamsGenerator = Callable[[], Params]
