"""JointTransformer model wrapper - uniform HandModel interface.

Wraps kabouzeid/JointTransformer's TransformerSF (ARCTIC p2 egocentric, ViT-G).

Key notes:
- Model is bimanual: one forward returns both .r and .l outputs. For zero-shot
  per-hand eval we run the full image crop once per sample and pick the
  appropriate branch based on hand_side.
- Input: 224x224 ImageNet-normalized.
- Output: 16 MANO kinematic joints per hand. Extended to 21-joint OpenPose
  convention via to_openpose_21 in-module.
- Per-sample intrinsics passed via set_intrinsics().
- `query_names` placeholder - the ARCTIC object head is stubbed.
"""
from __future__ import annotations
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image as PILImage

from ..geometry import crop_resize
from .base import HandModel, BatchPrediction
from .jointtransformer_setup import to_openpose_21


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class JointTransformerModel(HandModel):
    """TransformerSF wrapper with batched predict_batch()."""

    name = "jointtransformer"
    use_autocast = False  # JT MANO heads hard-assert FloatTensor (no fp16 in transforms)


    def __init__(self, model, device, img_res: int = 224, crop_size: int = 224):
        self.model = model.eval().to(device)
        self.device = device
        self.img_res = img_res
        self.crop_size = crop_size
        self._mean = IMAGENET_MEAN.to(device)
        self._std = IMAGENET_STD.to(device)
        self._intrinsics_cache: Optional[List[np.ndarray]] = None

    def set_intrinsics(self, intrinsics_list: List[np.ndarray]) -> None:
        """Pass per-sample (3,3) intrinsics for the next predict_batch call."""
        self._intrinsics_cache = intrinsics_list

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[PILImage.Image],
        bboxes: Sequence,
        hand_sides: Sequence[str],
    ) -> BatchPrediction:
        B = len(images)
        intr = self._intrinsics_cache
        assert intr is not None and len(intr) == B, (
            "JointTransformerModel requires set_intrinsics(list of (3,3) K) "
            "before predict_batch"
        )

        crops, K_list, sides, valid_idx, box_params = [], [], [], [], []
        for i, (img, bbox, hs) in enumerate(zip(images, bboxes, hand_sides)):
            if bbox is None:
                continue
            arr, params = crop_resize(img, bbox, size=self.crop_size)
            if arr is None:
                continue
            crops.append(arr)
            K_list.append(intr[i])
            sides.append(hs)
            box_params.append(params)
            valid_idx.append(i)

        if not crops:
            return BatchPrediction(joints_3d=np.zeros((0, 21, 3), dtype=np.float32),
                                   vertices_3d=np.zeros((0, 778, 3), dtype=np.float32),
                                   joints_2d_px=np.zeros((0, 21, 2), dtype=np.float32),
                                   valid_idx=[])

        x = torch.from_numpy(np.stack(crops, axis=0)).float().to(self.device) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = (x - self._mean) / self._std

        # Rescale full-image K to crop-space K
        K_crop = []
        for K_full, (cx, cy, scale) in zip(K_list, box_params):
            side_px = scale * 200.0
            if side_px <= 0:
                side_px = self.crop_size
            s = self.crop_size / side_px
            K_c = np.array([
                [K_full[0, 0] * s, 0.0,               (K_full[0, 2] - cx) * s + self.crop_size / 2.0],
                [0.0,               K_full[1, 1] * s, (K_full[1, 2] - cy) * s + self.crop_size / 2.0],
                [0.0,               0.0,               1.0],
            ], dtype=np.float32)
            K_crop.append(K_c)
        K_t = torch.from_numpy(np.stack(K_crop, axis=0)).float().to(self.device)

        meta_info = {"intrinsics": K_t, "query_names": ["capsulemachine"] * len(crops)}

        out = self.model({"img": x}, meta_info)

        j16_r = out["mano.joints3d.r"]
        j16_l = out["mano.joints3d.l"]
        v_r   = out["mano.vertices.r"]
        v_l   = out["mano.vertices.l"]
        cam_t_r = out["mano.cam_t.r"]
        cam_t_l = out["mano.cam_t.l"]

        side_is_left = torch.tensor([hs == "left" for hs in sides], device=self.device)
        j16 = torch.where(side_is_left.view(-1, 1, 1), j16_l, j16_r)
        v   = torch.where(side_is_left.view(-1, 1, 1), v_l,   v_r)
        cam_t = torch.where(side_is_left.view(-1, 1),   cam_t_l, cam_t_r)

        j21 = to_openpose_21(j16, v)

        j21_np   = j21.detach().cpu().numpy().astype(np.float32)
        v_np     = v.detach().cpu().numpy().astype(np.float32)
        cam_t_np = cam_t.detach().cpu().numpy().astype(np.float32)

        joints_2d_px = np.zeros((j21_np.shape[0], 21, 2), dtype=np.float32)
        for k, (K_full, _) in enumerate(zip(K_list, box_params)):
            j_cam = j21_np[k] + cam_t_np[k][None, :]
            z = np.clip(j_cam[:, 2:3], 1e-6, None)
            u = K_full[0, 0] * (j_cam[:, 0:1] / z) + K_full[0, 2]
            vp = K_full[1, 1] * (j_cam[:, 1:2] / z) + K_full[1, 2]
            joints_2d_px[k] = np.concatenate([u, vp], axis=1)

        return BatchPrediction(
            joints_3d=j21_np,
            vertices_3d=v_np,
            joints_2d_px=joints_2d_px,
            valid_idx=valid_idx,
        )
