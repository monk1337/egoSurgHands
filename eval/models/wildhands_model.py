"""WildHands model wrapper - uniform interface over ap229997/hands (demo branch)."""
from __future__ import annotations
import os
from typing import List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from ..config import MANO_TO_OPENPOSE
from .base import HandModel, BatchPrediction


class _MultiWildHandsDataset(torch.utils.data.Dataset):
    """Wraps N (image, bbox, side, K, focal) tuples and delegates per-item
    preprocessing to WildHandsDataset. Enables true batched forward passes."""

    def __init__(self, WildHandsDataset, cfg, images, boxes, rights, intrinsics, focals, rescale_factor):
        self.WildHandsDataset = WildHandsDataset
        self.cfg = cfg
        self.images = images
        self.boxes = boxes
        self.rights = rights
        self.intrinsics = intrinsics
        self.focals = focals
        self.rescale_factor = rescale_factor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        inner = self.WildHandsDataset(
            self.cfg,
            self.images[idx],
            np.array([self.boxes[idx]], dtype=np.float32),
            np.array([self.rights[idx]], dtype=np.float32),
            focal_length=self.focals[idx],
            rescale_factor=self.rescale_factor,
            intrx=self.intrinsics[idx].astype(np.float32).copy(),
        )
        return inner[0]


class WildHandsModel(HandModel):
    """
    Wrap a loaded WildHands model for zero-shot inference.

    Note: `use_autocast = False` because wildhands/common/transforms.py asserts
    `isinstance(x, FloatTensor)` which fails under fp16 autocast.
    """

    name = "wildhands"
    use_autocast = False

    def __init__(self, model, cfg, device, render_res: int = 840, rescale_factor: float = 1.75):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.render_res = render_res
        self.rescale_factor = rescale_factor
        self.mano_to_op = np.asarray(MANO_TO_OPENPOSE, dtype=np.int64)
        self._intrinsics_cache: List[np.ndarray] | None = None

        from wildhands.datasets.dataset import WildHandsDataset
        import wildhands.common.data_utils as wh_data_utils
        from hamer.utils import recursive_to
        self._WildHandsDataset = WildHandsDataset
        self._wh_data_utils = wh_data_utils
        self._recursive_to = recursive_to

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
        if self._intrinsics_cache is None:
            raise RuntimeError(
                "WildHandsModel requires per-sample intrinsics. "
                "Call .set_intrinsics([K_0, K_1, ...]) before predict_batch()."
            )
        if len(self._intrinsics_cache) != len(images):
            raise ValueError(
                f"Intrinsics cache length {len(self._intrinsics_cache)} != images {len(images)}"
            )

        # Build per-sample list, skipping any with bbox=None.
        valid_idx = []
        imgs_cv, boxes_v, rights_v, Ks_v, focals_v, sides_v, maxWH_v = [], [], [], [], [], [], []
        for i, (img, bbox, hs, K) in enumerate(zip(images, bboxes, hand_sides, self._intrinsics_cache)):
            if bbox is None:
                continue
            cv_img = np.array(img)
            H, W = cv_img.shape[:2]
            imgs_cv.append(cv_img)
            boxes_v.append(list(bbox))
            rights_v.append(1.0 if hs == "right" else 0.0)
            sides_v.append(hs)
            Ks_v.append(K)
            focals_v.append(float((K[0, 0] + K[1, 1]) / 2.0))
            maxWH_v.append(float(max(W, H)))
            valid_idx.append(i)
        self._intrinsics_cache = None

        if not imgs_cv:
            return BatchPrediction(joints_3d=np.zeros((0, 21, 3), dtype=np.float32), valid_idx=[])

        ds = _MultiWildHandsDataset(
            self._WildHandsDataset, self.cfg,
            imgs_cv, boxes_v, rights_v, Ks_v, focals_v,
            rescale_factor=self.rescale_factor,
        )
        # Batch the whole group in a single forward if memory allows; otherwise chunk.
        bs = int(os.environ.get("WILDHANDS_BS", "32"))
        nw = int(os.environ.get("WILDHANDS_NW", "4"))
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

        joints_list, verts_list, j2d_list = [], [], []
        cursor = 0
        for batch in loader:
            batch = self._recursive_to(batch, self.device)
            out = self.model(batch)
            # DataLoader collates our (inputs, meta_info) tuple into [inputs_dict, meta_dict]
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            bsz = inputs["img"].shape[0]
            for k in range(bsz):
                hs = sides_v[cursor + k]
                is_right = hs == "right"
                j_key = "pred.j3d.cam.r" if is_right else "pred.j3d.cam.l"
                v_key = "pred.v3d.cam.r" if is_right else "pred.v3d.cam.l"
                j2d_key = "pred.j2d.norm.r" if is_right else "pred.j2d.norm.l"
                joints_list.append(out[j_key][k].cpu().numpy())
                verts_list.append(out[v_key][k].cpu().numpy() if v_key in out else None)
                if j2d_key in out:
                    j2d_norm = out[j2d_key][k].cpu().numpy()
                    # normalize_kp2d maps [0, img_res] -> [-1, 1] on the rgb_processed
                    # whole-image space (max(W,H) rescaled to img_res). Invert to original px:
                    #   px_orig = (j2d_norm + 1) * 0.5 * max(W, H)
                    j2d_list.append(((j2d_norm + 1.0) * 0.5 * maxWH_v[cursor + k]).astype(np.float32))
                else:
                    j2d_list.append(None)
            cursor += bsz

        reorder = os.environ.get("WILDHANDS_REORDER", "mano_to_op").lower()
        joints_raw = np.stack(joints_list)
        j2d_raw = np.stack(j2d_list) if all(j is not None for j in j2d_list) else None
        if not getattr(self, "_logged_reorder", False):
            print(f"  [wildhands] reorder mode: {reorder!r} | batch_size={bs}")
            self._logged_reorder = True

        if reorder == "mano_to_op":
            joints_arr = joints_raw[:, self.mano_to_op]
            j2d_arr = j2d_raw[:, self.mano_to_op] if j2d_raw is not None else None
        elif reorder == "op_to_mano":
            inv = np.argsort(self.mano_to_op)
            joints_arr = joints_raw[:, inv]
            j2d_arr = j2d_raw[:, inv] if j2d_raw is not None else None
        else:
            joints_arr = joints_raw
            j2d_arr = j2d_raw

        verts_arr = None
        if all(v is not None for v in verts_list):
            verts_arr = np.stack(verts_list)

        return BatchPrediction(
            joints_3d=joints_arr,
            vertices_3d=verts_arr,
            joints_2d_px=j2d_arr,
            valid_idx=valid_idx,
        )
