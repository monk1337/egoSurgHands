"""
Qualitative visualization: draw GT and predicted hand keypoints on images.

Used by the eval loop to produce debug overlays for N random samples per
(model, dataset). Outputs PNG files with:
- Blue dots = GT 2D keypoints
- Red dots = Predicted 2D keypoints (in image pixel coords)
- Lines connecting finger joints (OpenPose skeleton)
- Title text with metric values
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np

# OpenPose hand skeleton edges (21 joints)
_FINGER_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
]


def draw_hand_overlay(
    img_rgb: np.ndarray,
    gt_2d: Optional[np.ndarray] = None,
    pred_2d: Optional[np.ndarray] = None,
    gt_v2_2d: Optional[np.ndarray] = None,
    title: str = "",
    radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Overlay GT and predicted hand keypoints on an image.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image
        gt_2d: (21, 2) ground truth 2D keypoints (OpenPose order) or None
        pred_2d: (21, 2) predicted 2D keypoints (OpenPose order) or None
        title: text to overlay on top
        radius: keypoint dot radius in pixels
        line_thickness: skeleton edge thickness

    Returns:
        out_rgb: (H, W, 3) uint8 RGB image with overlay
    """
    import cv2

    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    def _draw_skel(points, color_joint, color_edge):
        if points is None:
            return
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape != (21, 2):
            return
        # Edges
        for a, b in _FINGER_EDGES:
            p0 = pts[a]
            p1 = pts[b]
            if np.isnan(p0).any() or np.isnan(p1).any():
                continue
            cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     color_edge, line_thickness, cv2.LINE_AA)
        # Joints
        for i in range(21):
            p = pts[i]
            if np.isnan(p).any():
                continue
            cv2.circle(out, (int(p[0]), int(p[1])), radius, color_joint, -1, cv2.LINE_AA)

    # GT (stored) in blue, v2 MPS GT in green, pred in red
    _draw_skel(gt_2d, (255, 50, 50), (200, 40, 40))     # BGR: B=high → blue
    _draw_skel(gt_v2_2d, (50, 200, 50), (30, 160, 30))   # BGR: G=high → green
    _draw_skel(pred_2d, (40, 40, 255), (30, 30, 200))   # BGR: R=high → red

    if title:
        cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def save_overlay_png(out_dir, filename, img_rgb, gt_2d=None, pred_2d=None,
                     gt_v2_2d=None, title=""):
    """Save a single overlay to disk. Blue=stored GT, Green=v2 MPS GT, Red=pred."""
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_hand_overlay(img_rgb, gt_2d=gt_2d, pred_2d=pred_2d,
                                gt_v2_2d=gt_v2_2d, title=title)
    path = out_dir / filename
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return path
