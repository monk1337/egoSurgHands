"""WiLoR model wrapper - uniform interface over rolpotamias/WiLoR."""
from __future__ import annotations
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image as PILImage

from ..config import MANO_TO_OPENPOSE
from ..geometry import crop_resize
from .base import HandModel, BatchPrediction


class WiLoRModel(HandModel):
    """
    Wrap a loaded WiLoR model + cfg for batched zero-shot inference.

    Returns 21 joints (MANO→OpenPose), 778 vertices, and 2D keypoints
    projected to full image pixels.
    """

    name = "wilor"

    def __init__(self, model, cfg, device, crop_size: int = 256):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.crop_size = crop_size

        from wilor.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
        self.mean = torch.tensor(DEFAULT_MEAN, dtype=torch.float32).view(3, 1, 1) / 255.0
        self.std = torch.tensor(DEFAULT_STD, dtype=torch.float32).view(3, 1, 1) / 255.0
        self.mano_to_op = np.asarray(MANO_TO_OPENPOSE, dtype=np.int64)

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[PILImage.Image],
        bboxes: Sequence,
        hand_sides: Sequence[str],
    ) -> BatchPrediction:
        crops, box_centers, box_sizes, img_sizes, rights, valid_idx, is_left_flags = \
            [], [], [], [], [], [], []
        for i, (img, bbox, hs) in enumerate(zip(images, bboxes, hand_sides)):
            if bbox is None:
                continue
            c, params = crop_resize(img, bbox, size=self.crop_size)
            if c is None:
                continue
            cx, cy, scale = params
            if hs == "left":
                c = c[:, ::-1, :].copy()
            crops.append(c)
            box_centers.append([cx, cy])
            box_sizes.append(scale * 200)
            img_sizes.append([img.size[0], img.size[1]])
            rights.append(1.0 if hs == "right" else 0.0)
            is_left_flags.append(hs == "left")
            valid_idx.append(i)

        if not crops:
            return BatchPrediction(joints_3d=np.zeros((0, 21, 3), dtype=np.float32), valid_idx=[])

        crops_t = torch.from_numpy(np.stack(crops)).float().permute(0, 3, 1, 2) / 255.0
        crops_t = (crops_t - self.mean) / self.std
        box_centers_t = torch.tensor(box_centers, dtype=torch.float32, device=self.device)
        box_sizes_t = torch.tensor(box_sizes, dtype=torch.float32, device=self.device)

        batch = {
            "img": crops_t.to(self.device),
            "box_center": box_centers_t,
            "box_size": box_sizes_t,
            "img_size": torch.tensor(img_sizes, dtype=torch.float32, device=self.device),
            "right": torch.tensor(rights, dtype=torch.float32, device=self.device),
        }
        out = self.model(batch)

        # 3D joints - WiLoR's pred_keypoints_3d is ALREADY in OpenPose order
        # (verified 2026-05-06: WiLoR/wilor/models/mano_wrapper.py:35 applies
        # joint_map = mano_to_openpose internally, so no remap is needed here.
        # The prior `pred_kp3d[:, self.mano_to_op]` was a double-remap that scrambled joints.
        # Same fix as HaMeR's wrapper, propagated 2026-05-06.)
        pred_kp3d = out["pred_keypoints_3d"].cpu().numpy()
        pred_kp3d_op = pred_kp3d

        pred_vertices = None
        if "pred_vertices" in out:
            pred_vertices = out["pred_vertices"].cpu().numpy()

        pred_kp2d_px = None
        if "pred_keypoints_2d" in out:
            pred_kp2d_norm = out["pred_keypoints_2d"].cpu().numpy()
            box_sizes_np = np.asarray(box_sizes, dtype=np.float32)
            box_centers_np = np.asarray(box_centers, dtype=np.float32)
            pred_kp2d_unmirrored = pred_kp2d_norm.copy()
            for k, is_left in enumerate(is_left_flags):
                if is_left:
                    pred_kp2d_unmirrored[k, :, 0] = -pred_kp2d_unmirrored[k, :, 0]
            # 2D - same fix as 3D: pred_keypoints_2d is already in OpenPose order
            pred_kp2d_px = (
                pred_kp2d_unmirrored * box_sizes_np[:, None, None]
                + box_centers_np[:, None, :]
            )

        return BatchPrediction(
            joints_3d=pred_kp3d_op,
            vertices_3d=pred_vertices,
            joints_2d_px=pred_kp2d_px,
            valid_idx=valid_idx,
        )
