"""
phase0 - reusable zero-shot eval framework for surgical hand pose.

Supports HaMeR, WiLoR, and WildHands against:
- harvardsil/aria-surgical-hand-pose (fitted-MANO eval GT)
- harvardsil/pov-surgery (native MANO GT)

All predictions and GT are reordered to OpenPose joint convention for
consistent cross-model comparison.
"""
from .config import (
    MANO_TO_OPENPOSE,
    ARIA_MPS_TO_OPENPOSE,
    OP_VALID,
    DEFAULT_DATASETS,
)
from .geometry import (
    joints2d_to_bbox,
    crop_resize,
    procrustes_align,
    metrics_mm,
)
from .mano_fit import MANOFitter, FitResult
from .models.base import HandModel, BatchPrediction
from .eval_loop import eval_stream, EvalConfig, EvalResult
from .viz import draw_hand_overlay, save_overlay_png

__all__ = [
    "MANO_TO_OPENPOSE",
    "ARIA_MPS_TO_OPENPOSE",
    "OP_VALID",
    "DEFAULT_DATASETS",
    "joints2d_to_bbox",
    "crop_resize",
    "procrustes_align",
    "metrics_mm",
    "MANOFitter",
    "FitResult",
    "HandModel",
    "BatchPrediction",
    "eval_stream",
    "EvalConfig",
    "EvalResult",
    "draw_hand_overlay",
    "save_overlay_png",
]

