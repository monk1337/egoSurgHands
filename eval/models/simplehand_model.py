"""SimpleHand (Zhou et al. CVPR 2024) wrapper - uniform HandModel interface.

Wraps patienceFromZhou/simpleHand HandNet (fastvit_ma36 backbone, HANDS23 winner).

Key notes:
- Input: 224x224 RGB, scaled internally as (image/255 - 0.5) - no ImageNet norm.
- Output: uv (B, 42) in [0,1] crop coords, joints (B, 21, 3) in meters (OpenPose
  order, ROOT-RELATIVE - no root translation head), vertices (B, 778, 3).
- Joint order after simpleHand's remap_joints_and_fingertip is OpenPose-21:
  [wrist, thumb(CMC/MCP/IP/TIP), index(MCP/PIP/DIP/TIP), middle×4, ring×4, pinky×4]
- Left-hand input: horizontally flip image, run MANO-right model, mirror output.
- Absolute MPJPE and root_err are meaningless; only PA-MPJPE / PA-PVE / PA-MPVPE
  and P2D are valid.
"""
from __future__ import annotations
from typing import List, Optional, Sequence

import sys, os
import numpy as np
import torch
from PIL import Image as PILImage

# Ensure simpleHand is importable
_SH_PATH = "/content/simpleHand"
if _SH_PATH not in sys.path:
    sys.path.insert(0, _SH_PATH)

from ..geometry import crop_resize
from .base import HandModel, BatchPrediction


class SimpleHandModel(HandModel):
    """HandNet wrapper with batched predict_batch()."""

    name = "simplehand"
    use_autocast = False  # fastvit backbone + hand_net have fp32 assumptions


    def __init__(self, model, device, crop_size: int = 224):
        self.model = model.eval().to(device)
        self.device = device
        self.crop_size = crop_size
        self._intrinsics_cache: Optional[List[np.ndarray]] = None

    def set_intrinsics(self, intrinsics_list):
        self._intrinsics_cache = intrinsics_list

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[PILImage.Image],
        bboxes: Sequence,
        hand_sides: Sequence[str],
    ) -> BatchPrediction:
        B = len(images)

        crops, sides, valid_idx, box_params = [], [], [], []
        for i, (img, bbox, hs) in enumerate(zip(images, bboxes, hand_sides)):
            if bbox is None:
                continue
            arr, params = crop_resize(img, bbox, size=self.crop_size)
            if arr is None:
                continue
            crops.append(arr)
            sides.append(hs)
            box_params.append(params)
            valid_idx.append(i)

        if not crops:
            return BatchPrediction(
                joints_3d=np.zeros((0, 21, 3), dtype=np.float32),
                vertices_3d=np.zeros((0, 778, 3), dtype=np.float32),
                joints_2d_px=np.zeros((0, 21, 2), dtype=np.float32),
                valid_idx=[],
            )

        # arr from crop_resize is (H,W,3) uint8-like or float 0..1 per repo. Inspect at runtime.
        crops_np = np.stack(crops, axis=0)  # (N, H, W, 3)
        # Normalize to [0, 255] - SimpleHand does (image/255 - 0.5) internally
        if crops_np.dtype != np.uint8 and crops_np.max() <= 1.5:
            crops_np = (crops_np * 255.0).astype(np.float32)
        else:
            crops_np = crops_np.astype(np.float32)

        x = torch.from_numpy(crops_np).permute(0, 3, 1, 2).contiguous().to(self.device)

        # Left-hand: horizontal flip, run right-hand model, mirror output
        N = x.shape[0]
        is_left = torch.tensor([hs == "left" for hs in sides], device=self.device)
        if is_left.any():
            idx = torch.where(is_left)[0]
            x[idx] = torch.flip(x[idx], dims=[3])  # flip along W

        out = self.model(x)
        joints = out["joints"].reshape(N, 21, 3)          # meters, root-relative, OpenPose order
        vertices = out["vertices"].reshape(N, 778, 3)
        uv = out["uv"].reshape(N, 21, 2) * float(self.crop_size)  # crop pixel coords

        # Mirror outputs back for left hands
        if is_left.any():
            idx = torch.where(is_left)[0]
            joints[idx, :, 0] = -joints[idx, :, 0]
            vertices[idx, :, 0] = -vertices[idx, :, 0]
            uv[idx, :, 0] = float(self.crop_size) - uv[idx, :, 0]

        joints_np = joints.cpu().numpy().astype(np.float32)
        vertices_np = vertices.cpu().numpy().astype(np.float32)
        uv_np = uv.cpu().numpy().astype(np.float32)

        # Map crop-space uv -> full-image px using box_params from crop_resize
        # box_params: (cx, cy, side) - center + side length in full image
        full_uv = np.zeros_like(uv_np)
        for k, p in enumerate(box_params):
            cx, cy, side = p
            # uv in [0, crop_size]; map to [cx-side/2, cx+side/2]
            x_full = (uv_np[k, :, 0] / self.crop_size) * side + (cx - side / 2.0)
            y_full = (uv_np[k, :, 1] / self.crop_size) * side + (cy - side / 2.0)
            full_uv[k, :, 0] = x_full
            full_uv[k, :, 1] = y_full

        return BatchPrediction(
            joints_3d=joints_np,
            vertices_3d=vertices_np,
            joints_2d_px=full_uv,
            valid_idx=valid_idx,
        )
