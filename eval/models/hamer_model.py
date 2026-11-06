"""HaMeR model wrapper - uniform interface over geopavlakos/hamer."""
from __future__ import annotations
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image as PILImage

from ..config import MANO_TO_OPENPOSE
from ..geometry import crop_resize
from .base import HandModel, BatchPrediction


class HaMeRModel(HandModel):
    """
    Wrap a loaded HaMeR model + cfg for batched zero-shot inference.

    Returns 21-joint MANO reordered to OpenPose, 778 MANO vertices, and
    predicted 2D keypoints in *full image pixel coordinates* (via
    crop→full mapping from HaMeR demo).
    """

    name = "hamer"

    def __init__(self, model, cfg, device, crop_size: int = 256):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.crop_size = crop_size

        from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
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

        # 3D joints - HaMeR's pred_keypoints_3d is ALREADY in OpenPose order
        # (empirically verified 2026-04-16: the upstream hamer/mano_wrapper.py returns
        # joints in OpenPose order, so no MANO_TO_OP remap is needed here. The previous
        # remap was silently canceled by a matching bug in eval_loop.py that also applied
        # MANO_TO_OP to aria_stored GT; fixing both unmasks a 24mm WildHands improvement.)
        pred_kp3d = out["pred_keypoints_3d"].cpu().numpy()  # (K, 21, 3) OpenPose order
        pred_kp3d_op = pred_kp3d

        # Vertices
        pred_vertices = None
        if "pred_vertices" in out:
            pred_vertices = out["pred_vertices"].cpu().numpy()  # (K, 778, 3)

        # 2D keypoints: HaMeR outputs pred_keypoints_2d in normalized crop coords [-0.5, 0.5]
        pred_kp2d_px = None
        if "pred_keypoints_2d" in out:
            pred_kp2d_norm = out["pred_keypoints_2d"].cpu().numpy()  # (K, 21, 2)
            # Un-mirror for left hands (flip X in normalized coords)
            box_sizes_np = np.asarray(box_sizes, dtype=np.float32)
            box_centers_np = np.asarray(box_centers, dtype=np.float32)
            pred_kp2d_unmirrored = pred_kp2d_norm.copy()
            for k, is_left in enumerate(is_left_flags):
                if is_left:
                    pred_kp2d_unmirrored[k, :, 0] = -pred_kp2d_unmirrored[k, :, 0]
            # Normalized → crop pixel → full image pixel
            # pred_kp2d_norm * box_size + box_center
            pred_kp2d_px_mano = (
                pred_kp2d_unmirrored * box_sizes_np[:, None, None]
                + box_centers_np[:, None, :]
            )
            pred_kp2d_px = pred_kp2d_px_mano  # already OpenPose order (see 3D comment above)

        return BatchPrediction(
            joints_3d=pred_kp3d_op,
            vertices_3d=pred_vertices,
            joints_2d_px=pred_kp2d_px,
            valid_idx=valid_idx,
        )
