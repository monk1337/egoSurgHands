"""MeshGraphormer wrapper - uniform interface over microsoft/MeshGraphormer.

Output convention: 21 joints in OpenPose order, root-aligned (wrist at 0), meters.
Adapted from bmif702/AIM_2_Project/src/eval/eval_meshgraphormer.py.
"""
from __future__ import annotations
from typing import List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from .base import HandModel, BatchPrediction


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _mgfm_crop(img_np: np.ndarray, bbox_xyxy):
    """224x224 affine crop matching MGFM's reference inference recipe."""
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bbox_size = max(x1 - x0, y1 - y0)
    src = np.array([
        [cx - bbox_size / 2, cy - bbox_size / 2],
        [cx + bbox_size / 2, cy - bbox_size / 2],
        [cx - bbox_size / 2, cy + bbox_size / 2],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [224, 0], [0, 224]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(
        img_np, M, (224, 224),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    img_t = crop.astype(np.float32) / 255.0
    img_t = (img_t - _IMAGENET_MEAN) / _IMAGENET_STD
    img_t = np.transpose(img_t, (2, 0, 1))
    return img_t


class MeshGraphormerModel(HandModel):
    use_autocast = False
    """MGFM hand mesh wrapper - flips left hands to the right-hand canonical frame."""

    name = "meshgraphormer"

    def __init__(self, model, mano, mesh_sampler, device):
        self.model = model
        self.mano = mano
        self.mesh_sampler = mesh_sampler
        self.device = device

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[PILImage.Image],
        bboxes: Sequence,
        hand_sides: Sequence[str],
    ) -> BatchPrediction:
        crops, valid_idx, flip_flags = [], [], []
        for i, (img, bbox, hs) in enumerate(zip(images, bboxes, hand_sides)):
            if bbox is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            if x1 <= x0 or y1 <= y0:
                continue
            img_np = np.asarray(img.convert("RGB"))
            flip = (hs == "left")
            if flip:
                img_np = img_np[:, ::-1].copy()
                W = img_np.shape[1]
                # mirror bbox horizontally
                x0n = W - x1
                x1n = W - x0
                bbox_use = (x0n, y0, x1n, y1)
            else:
                bbox_use = (x0, y0, x1, y1)
            patch = _mgfm_crop(img_np, bbox_use)
            crops.append(patch)
            flip_flags.append(flip)
            valid_idx.append(i)

        if not crops:
            return BatchPrediction(
                joints_3d=np.zeros((0, 21, 3), dtype=np.float32),
                valid_idx=[],
            )

        x = torch.from_numpy(np.stack(crops)).to(self.device, dtype=torch.float32)
        outs = self.model(x, self.mano, self.mesh_sampler)
        # MGFM forward returns (pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, ...)
        pred_3d_joints = outs[1].cpu().float().numpy()  # (B, 21, 3) OP order
        pred_vertices = outs[3].cpu().float().numpy()   # (B, 778, 3)

        # Root-align (subtract wrist)
        wrist = pred_3d_joints[:, 0:1, :]
        pred_3d = pred_3d_joints - wrist
        pred_v = pred_vertices - wrist

        # Un-flip the left-hand outputs back to image-frame x
        for k, flip in enumerate(flip_flags):
            if flip:
                pred_3d[k, :, 0] = -pred_3d[k, :, 0]
                pred_v[k, :, 0] = -pred_v[k, :, 0]

        return BatchPrediction(
            joints_3d=pred_3d.astype(np.float32),
            vertices_3d=pred_v.astype(np.float32),
            joints_2d_px=None,
            valid_idx=valid_idx,
        )
