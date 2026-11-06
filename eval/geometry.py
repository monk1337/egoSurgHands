"""Geometry helpers: bbox derivation, crop/resize, Procrustes, metrics."""
from __future__ import annotations
import numpy as np
from PIL import Image as PILImage


def joints2d_to_bbox(joints_2d, image_w: int, image_h: int, pad_ratio: float = 0.3, min_in_bounds: int = 5):
    """
    Derive a square hand bounding box from 2D joints.

    Args:
        joints_2d: (21, 2) array of predicted 2D joint positions
        image_w, image_h: full image dims
        pad_ratio: fractional padding around the tight joint bbox
        min_in_bounds: minimum number of in-bounds joints required

    Returns:
        np.ndarray [x0, y0, x1, y1] or None if too few joints are valid.
    """
    j = np.asarray(joints_2d, dtype=np.float32).reshape(21, 2)
    in_bounds = (j[:, 0] >= 0) & (j[:, 0] < image_w) & (j[:, 1] >= 0) & (j[:, 1] < image_h)
    if in_bounds.sum() < min_in_bounds:
        return None
    j_in = j[in_bounds]
    x0, y0 = j_in.min(axis=0)
    x1, y1 = j_in.max(axis=0)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + pad_ratio * 2)
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)


def crop_resize(img_pil: PILImage.Image, bbox, size: int = 256):
    """
    Crop and resize a region from a PIL image for model input.

    Returns:
        (arr, (cx, cy, scale)) where arr is uint8 (size, size, 3) and scale is in HaMeR units (/200).
        Returns (None, None) if the crop is degenerate.
    """
    W, H = img_pil.size
    x0, y0, x1, y1 = bbox
    x0c, y0c = max(0, int(x0)), max(0, int(y0))
    x1c, y1c = min(W, int(x1)), min(H, int(y1))
    if x1c <= x0c or y1c <= y0c:
        return None, None
    crop = img_pil.crop((x0c, y0c, x1c, y1c)).resize((size, size), PILImage.Resampling.BILINEAR)
    arr = np.array(crop, dtype=np.uint8)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    scale = max(x1 - x0, y1 - y0) / 200.0
    return arr, (cx, cy, scale)


def procrustes_align(pred, gt):
    """
    Full Procrustes alignment: rotation + uniform scale + translation.

    Args:
        pred: (N, 3) predicted 3D points
        gt: (N, 3) ground truth 3D points (same order as pred)

    Returns:
        aligned: (N, 3) rigidly aligned pred, same units as inputs
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mu_p = pred.mean(0)
    mu_g = gt.mean(0)
    p = pred - mu_p
    g = gt - mu_g
    M = p.T @ g
    U, S, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        S = S.copy()
        S[-1] *= -1
        R = Vt.T @ U.T
    var_p = (p ** 2).sum()
    scale = S.sum() / var_p if var_p > 1e-12 else 1.0
    return (scale * (p @ R.T) + mu_g).astype(np.float32)


def metrics_mm(pred_3d, gt_3d):
    """
    Compute MPJPE and PA-MPJPE (Procrustes-aligned) in millimeters.

    Args:
        pred_3d: (N, 3) predicted joints, in meters, root-centered recommended
        gt_3d: (N, 3) ground-truth joints, same frame/units

    Returns:
        (mpjpe_mm, pa_mpjpe_mm) as floats
    """
    pred = np.asarray(pred_3d, dtype=np.float32)
    gt = np.asarray(gt_3d, dtype=np.float32)
    mpjpe = float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000)
    aligned = procrustes_align(pred, gt)
    pa = float(np.linalg.norm(aligned - gt, axis=-1).mean() * 1000)
    return mpjpe, pa


def mrrpe_mm(pred_root, gt_root):
    """Mean Root-Relative Position Error - L2 distance between predicted and GT
    wrist positions in camera frame (mm). Measures absolute 3D placement quality.

    Args:
        pred_root: (3,) predicted wrist position in camera frame (meters)
        gt_root: (3,) GT wrist position in camera frame (meters)

    Returns:
        mrrpe_mm: float
    """
    return float(np.linalg.norm(np.asarray(pred_root) - np.asarray(gt_root)) * 1000)
