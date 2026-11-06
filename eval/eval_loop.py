"""
Streaming eval loop with full metric suite + per-sample caching + viz dump.

Computes for each sample:
    - MPJPE        (3D joint error, mm)
    - PA-MPJPE     (Procrustes-aligned 3D joint error, mm)
    - PVE          (3D mesh vertex error, mm) - if pred/gt vertices present
    - PA-PVE       (Procrustes-aligned 3D mesh vertex error, mm)
    - P2D          (2D pixel joint error, px) - if pred/gt 2D present

Aggregates:
    - Overall mean/std per metric
    - Per-sequence breakdown
    - Per-hand-side breakdown

Saves:
    - {out_dir}/{model}_{dataset}_metrics.json     - summary stats
    - {out_dir}/{model}_{dataset}_per_sample.pkl   - per-sample cache (pred+gt+metrics)
    - {out_dir}/viz/{model}_{dataset}/*.png        - N random overlays
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Optional
import json
import pickle
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from .config import MANO_TO_OPENPOSE, OP_VALID
from .geometry import joints2d_to_bbox, metrics_mm, mrrpe_mm, procrustes_align
from .mano_fit import MANOFitter
from .models.base import HandModel
from .viz import save_overlay_png


@dataclass
class EvalConfig:
    batch_size: int = 32
    n_per_sequence: Optional[int] = None   # cap samples per sequence (None = unlimited)
    max_total: Optional[int] = None         # hard cap on total samples
    log_every: int = 10
    show_progress: bool = True
    hf_token: Optional[str] = None
    # Output
    out_dir: Optional[str] = None            # where to save metrics/cache/viz
    save_per_sample: bool = True             # save pickled per-sample cache
    viz_count: int = 20                      # N random samples to save as overlay PNGs (-1 = ALL)
    viz_seed: int = 0
    save_pred_vertices: bool = False         # include verts in per-sample pkl (10x size)
    viz_dual_gt: bool = False                # draw both stored (blue) and v2 (green) GT on overlays


@dataclass
class EvalResult:
    model_name: str
    dataset_name: str
    n_samples: int
    metrics: dict = field(default_factory=dict)                 # overall mean/std per metric
    per_sequence: dict = field(default_factory=dict)            # seq → metric dict
    per_hand_side: dict = field(default_factory=dict)           # 'left'/'right' → metric dict
    mano_fit_residual_mean: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "n_samples": self.n_samples,
            "overall": self.metrics,
            "per_sequence": self.per_sequence,
            "per_hand_side": self.per_hand_side,
            "mano_fit_residual_mm": self.mano_fit_residual_mean,
        }


def _load_dataset_stream(dataset_spec: dict, cfg: EvalConfig):
    from datasets import load_dataset
    if dataset_spec.get("local_parquet_glob"):
        import glob as _glob
        _gp = dataset_spec["local_parquet_glob"]
        _files = sorted(_glob.glob(_gp))
        print("[v2-loader] " + str(len(_files)) + " local parquet shards from " + _gp)
        return load_dataset("parquet", data_files=_files, split="train", streaming=True)
    
    indices = dataset_spec.get("sample_indices")
    if indices is not None:
        # Non-streaming + select: needed for smart sampling
        ds = load_dataset(
            dataset_spec["repo_id"],
            split=dataset_spec["split"],
            token=cfg.hf_token,
        )
        return ds.select(list(indices))
    return load_dataset(
        dataset_spec["repo_id"],
        split=dataset_spec["split"],
        token=cfg.hf_token,
        streaming=True,
    )


def _compute_pve(pred_vertices, gt_vertices):
    """Return (PVE_mm, PA-PVE_mm) for a single sample."""
    pv = np.asarray(pred_vertices, dtype=np.float32)
    gv = np.asarray(gt_vertices, dtype=np.float32)
    pve = float(np.linalg.norm(pv - gv, axis=-1).mean() * 1000)
    aligned = procrustes_align(pv, gv)
    pa_pve = float(np.linalg.norm(aligned - gv, axis=-1).mean() * 1000)
    return pve, pa_pve


def _compute_p2d(pred_2d_px, gt_2d_px):
    """Return mean pixel distance between pred and GT 2D keypoints (21 joints)."""
    pp = np.asarray(pred_2d_px, dtype=np.float32)
    gp = np.asarray(gt_2d_px, dtype=np.float32)
    return float(np.linalg.norm(pp - gp, axis=-1).mean())


def _agg(records: list, key: str):
    """Aggregate mean + std for a metric key over records where it's not None."""
    vals = [r[key] for r in records if r.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def _compute_paper_mrrpe(records: list) -> dict:
    """Paper (WildHands/ARCTIC) MRRPE: pair L+R hands per (seq, frame), align
    left roots, measure residual right-root error.

        MRRPE_frame = ||(pred_R - pred_L) - (gt_R - gt_L)||

    Only defined on frames where both L and R hands were predicted successfully.
    """
    from collections import defaultdict
    frames = defaultdict(dict)
    for r in records:
        key = (r.get("sequence_name"), r.get("frame_id"))
        hs = r.get("hand_side")
        if hs in ("left", "right"):
            frames[key][hs] = r
    vals = []
    for key, hands in frames.items():
        if "left" in hands and "right" in hands:
            pL = np.asarray(hands["left"]["pred_joints_3d"])[0]
            pR = np.asarray(hands["right"]["pred_joints_3d"])[0]
            gL = np.asarray(hands["left"]["gt_joints_3d"])[0]
            gR = np.asarray(hands["right"]["gt_joints_3d"])[0]
            pred_rel = pR - pL
            gt_rel = gR - gL
            vals.append(float(np.linalg.norm(pred_rel - gt_rel) * 1000))
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def _aggregate(records: list) -> dict:
    out = {
        # Primary (focus) - vs primary GT (stored for Aria dual_v2)
        "mpjpe_mm":       _agg(records, "mpjpe_mm"),       # root-relative (paper MPJPE)
        "p2d_px":         _agg(records, "p2d_px"),         # 2D L2 (paper)
        # Secondary (focus)
        "pa_mpjpe_mm":    _agg(records, "pa_mpjpe_mm"),    # Procrustes-aligned
        "root_err_mm":    _agg(records, "root_err_mm"),    # single-hand root L2 err
        # Tertiary (focus, only frames with both L+R)
        "mrrpe_paper_mm": _compute_paper_mrrpe(records),   # cross-hand relative root err
        # Keep but don't focus
        "mpjpe_abs_mm":   _agg(records, "mpjpe_abs_mm"),   # absolute (no root subtract)
        "pve_mm":         _agg(records, "pve_mm"),
        "pa_pve_mm":      _agg(records, "pa_pve_mm"),
    }
    # v2 MPS GT metrics (dual-GT mode: aria_val_v2). All suffixed _v2.
    # These are only populated for samples where v2 lookup succeeded.
    for k in ("mpjpe_v2_mm", "pa_mpjpe_v2_mm", "root_err_v2_mm",
              "mpjpe_abs_v2_mm", "p2d_v2_px"):
        out[k] = _agg(records, k)
    return out


def eval_stream(
    model: HandModel,
    dataset_spec: dict,
    dataset_name: str,
    device: torch.device,
    config: EvalConfig,
    mano_fitter: Optional[MANOFitter] = None,
) -> EvalResult:
    """
    Full eval pass. See module docstring for what gets computed and saved.
    """
    _device = device if hasattr(device, "type") else torch.device(device)
    needs_fit = dataset_spec.get("needs_mano_fit", False)
    if needs_fit and mano_fitter is None:
        raise ValueError(
            f"Dataset {dataset_spec['repo_id']} needs MANO fitting - pass mano_fitter."
        )

    gt_field = dataset_spec["gt_field"]
    gt_order = dataset_spec["gt_order"]
    sample_filter = dataset_spec.get("filter") or (lambda s: True)
    mano_to_op_np = np.asarray(MANO_TO_OPENPOSE, dtype=np.int64)
    op_valid_np = np.asarray(OP_VALID, dtype=np.int64)  # 20 OP indices with MPS analog

    # v2 MPS lookup (for aria_dual_v2). Pull from HF dataset repo once.
    # Keyed by f"{PR}/{frame_id}/{hand_side}"; values have j3d_v2_cam, j2d_v2_disp,
    # confidence, dt_ms, rotation. j3d_v2_cam is already in OpenPose-21 order (the
    # MPS_TO_MANO permutation is actually MPS→OP - index 1 holds MPS palm_center as
    # a placeholder for OP thumb_CMC which has no MPS analog).
    v2_lookup: dict = {}
    v2_inline = bool(dataset_spec.get("v2_inline", False))
    if gt_order == "aria_dual_v2" and not v2_inline:
        v2_repo = dataset_spec.get("v2_lookup_repo")
        v2_file = dataset_spec.get("v2_lookup_file")
        if v2_repo and v2_file:
            from huggingface_hub import hf_hub_download
            _v2_path = hf_hub_download(
                repo_id=v2_repo, filename=v2_file, repo_type="dataset",
                token=config.hf_token,
            )
            with open(_v2_path) as _f:
                v2_lookup = json.load(_f)
            print(f"  [v2-lookup] {len(v2_lookup)} entries from {v2_repo}/{v2_file}")
        else:
            print(f"  [v2-lookup] WARNING: aria_dual_v2 but no lookup repo/file configured")
    elif v2_inline:
        print(f"  [v2-lookup] inline mode - reading joints_3d_v2 from each row")

    ds_iter = _load_dataset_stream(dataset_spec, config)

    buf: list = []
    seq_count: dict = defaultdict(int)
    per_sample_records: list = []
    viz_samples: list = []                # keep a reservoir of candidates for viz
    fit_residuals: list = []
    total = 0
    batch_count = 0
    t_start = time.time()
    rng = random.Random(config.viz_seed)

    # Empirically-determined rotations to convert MPS device frame → Aria camera frame.
    # Right hand: (x, z, -y) = +90° about -X axis. Validated 99.7% in-image on PR81-85.
    # Left hand: identity. The MANOFitter's mirror/un-mirror sequence leaves left hand
    # joints in a frame that does NOT need the device→camera rotation.
    # See paper/06_results/phase0_image_analysis.md for the empirical analysis.
    _R_DEV_TO_CAM_RIGHT = np.array(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
    )
    _R_DEV_TO_CAM_LEFT = np.eye(3, dtype=np.float32)

    def _device_to_camera(joints_3d, hand_side):
        """Rotate (21, 3) joints from MPS device frame to Aria camera frame."""
        R = _R_DEV_TO_CAM_RIGHT if hand_side == "right" else _R_DEV_TO_CAM_LEFT
        return (R @ joints_3d.T).T.astype(np.float32)

    def _project_camera_to_image(joints_cam_3d, K):
        """Project (21, 3) camera-frame joints through intrinsics K → (21, 2) pixels."""
        proj = (K @ joints_cam_3d.T).T  # (21, 3)
        return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

    def _process(buf_):
        nonlocal batch_count
        batch_count += 1

        # v1 dataset stores image as raw bytes (Value("binary")); older datasets use HF Image() (auto-PIL).
        # Handle both: convert bytes → PIL inline.
        from PIL import Image as _PILImage
        import io as _io
        def _to_pil(_x):
            if hasattr(_x, "size"): return _x
            if isinstance(_x, dict) and "bytes" in _x: _x = _x["bytes"]
            if isinstance(_x, (bytes, bytearray)):
                try:
                    return _PILImage.open(_io.BytesIO(_x)).convert("RGB")
                except Exception as _e:
                    return _PILImage.new("RGB", (1408, 1408), color=(128,128,128))
            return _x
        images = [_to_pil(s["image"]) for s in buf_]
        sides = [s["hand_side"] for s in buf_]
        seqs = [s["sequence_name"] for s in buf_]
        fids = [s["frame_id"] for s in buf_]
        gt_raw_joints = [np.asarray(s[gt_field], dtype=np.float32).reshape(21, 3) for s in buf_]
        gt_raw_verts = [
            np.asarray(s["vertices"], dtype=np.float32).reshape(778, 3) if "vertices" in s else None
            for s in buf_
        ]
        K_list = [np.asarray(s["intrinsics"], dtype=np.float32).reshape(3, 3) for s in buf_]

        # Resolve GT joints (+ vertices for PVE) to OpenPose order / fitted-MANO
        # gt_joints_cam: (21,3) in Aria CAMERA frame (rotation applied for Aria; identity for POV)
        # gt_verts_cam: (778,3) in same frame
        gt_joints_cam: list
        gt_verts_cam: list
        if gt_order == "mano":
            # POV-Surgery: native MANO GT - already in camera frame after MANO forward
            gt_joints_cam = [g[mano_to_op_np] for g in gt_raw_joints]
            gt_verts_cam = list(gt_raw_verts)
        elif gt_order in ("aria_stored", "aria_dual_v2"):
            # Use stored joints_3d (from SAM2+HaMeR pipeline) - already OpenPose-ordered
            # (empirically verified 2026-04-16 by bone-length analysis: e.g. joint[1] dist
            #  from wrist ≈ 39mm which matches OpenPose thumb_CMC, not MANO index_MCP).
            # Previously reordered with MANO_TO_OP, which silently canceled HaMeR's
            # matching wrapper bug but inflated WildHands PA-MPJPE by ~24mm.
            # This is a PSEUDO-GT from the same model family as HaMeR/WiLoR, so comparing
            # HaMeR to itself is circular. BUT comparing WiLoR or WildHands to it is
            # a meaningful cross-model agreement test.
            stored_j3d = [
                np.asarray(s["joints_3d"], dtype=np.float32).reshape(21, 3) for s in buf_
            ]
            gt_joints_cam = list(stored_j3d)
            # vertices are also stored
            stored_verts = [
                np.asarray(s["vertices"], dtype=np.float32).reshape(778, 3) if "vertices" in s and s["vertices"] else None
                for s in buf_
            ]
            gt_verts_cam = stored_verts
        elif gt_order == "aria_mps":
            # Aria val: fit MANO to MPS landmarks (which are in MPS device frame)
            # KNOWN ISSUE: produces ~40mm offset from HaMeR-consistent joints due to
            # joint-convention differences between MPS and MANO. See phase0_image_analysis.md.
            assert mano_fitter is not None
            mps_t = torch.tensor(np.stack(gt_raw_joints), device=device)
            fit = mano_fitter.fit_batch(mps_t, sides, verbose=False)
            fit_residuals.extend(fit.residual_mm.tolist())
            gt_joints_cam = [
                _device_to_camera(fit.joints_21_op[k], sides[k]) for k in range(len(buf_))
            ]
            gt_verts_cam = [
                _device_to_camera(fit.vertices[k], sides[k]) for k in range(len(buf_))
            ]
        elif gt_order == "aria_manofit":
            # Aria val: precomputed MANO fits (S6 pipeline) loaded from local Drive dir.
            # GT joints already OP-21 ordered; vertices available for PVE.
            # These fits are in the MPS device frame (un-mirrored), so we still need
            # the _device_to_camera rotation, same as the aria_mps branch.
            fit_j_raw = [
                np.asarray(s["mano_fit_joints_21_op"], dtype=np.float32).reshape(21, 3)
                for s in buf_
            ]
            fit_v_raw = [
                np.asarray(s["mano_fit_vertices"], dtype=np.float32).reshape(778, 3)
                for s in buf_
            ]
            gt_joints_cam = [
                _device_to_camera(fit_j_raw[k], sides[k]) for k in range(len(buf_))
            ]
            gt_verts_cam = [
                _device_to_camera(fit_v_raw[k], sides[k]) for k in range(len(buf_))
            ]
            for s in buf_:
                r = s.get("mano_fit_residual_mm")
                if r is not None:
                    fit_residuals.append(float(r))
        else:
            raise ValueError(f"Unknown gt_order: {gt_order}")

        # GT 2D for P2D metric + viz.
        # - aria_stored: stored_j3d is in crop-local frame, cannot project through K;
        #   use the stored joints_2d field directly (pre-projected by HaMeR pipeline).
        # - otherwise: project camera-frame GT through K.
        if gt_order in ("aria_stored", "aria_dual_v2"):
            # stored joints_2d is already OpenPose-ordered (same pipeline as joints_3d)
            gt_2d_proj = [
                np.asarray(s["joints_2d"], dtype=np.float32).reshape(21, 2)
                for s in buf_
            ]
        else:
            gt_2d_proj = [
                _project_camera_to_image(gt_joints_cam[k], K_list[k]) for k in range(len(buf_))
            ]
        # aria_manofit uses the same central bbox as aria_mps/aria_stored
        if gt_order == "aria_manofit":
            pass  # bbox selection handled below alongside aria_mps/aria_stored
        # For Aria, the stored joints_2d and the device→camera rotation+projection
        # both point to the wrong place. Empirically, HaMeR with a large central
        # bbox (300..1200 in a 1408 image) reliably detects the actual hand.
        # We use that central bbox for Aria and the stored joints_2d-derived bbox for POV.
        # bbox_override: dict {(seq, frame_id, hand_side): np.array([x0,y0,x1,y1])} - used
        # when a prior model's 2D predictions give a tighter hand bbox than the default.
        bbox_override = dataset_spec.get("bbox_override") or {}
        if gt_order in ("aria_mps", "aria_stored", "aria_manofit", "aria_dual_v2"):
            bboxes = []
            for k in range(len(buf_)):
                key = (seqs[k], fids[k], sides[k])
                if key in bbox_override:
                    bboxes.append(np.asarray(bbox_override[key], dtype=np.float32))
                else:
                    bboxes.append(np.array([300.0, 300.0, 1200.0, 1200.0], dtype=np.float32))
        else:
            stored_j2ds = [
                np.asarray(s["joints_2d"], dtype=np.float32).reshape(21, 2) for s in buf_
            ]
            bboxes = [
                joints2d_to_bbox(stored_j2ds[k], images[k].size[0], images[k].size[1])
                for k in range(len(buf_))
            ]
        # Also stash j2ds as the recomputed projections for viz / P2D metric below.
        j2ds = gt_2d_proj
        # Alias for backward compatibility with downstream code:
        gt_joints_op = gt_joints_cam
        gt_verts = gt_verts_cam

        # WildHands needs per-sample intrinsics (already extracted above as K_list)
        if hasattr(model, "set_intrinsics"):
            model.set_intrinsics(K_list)

        _use_ac = getattr(model, "use_autocast", True) and (_device.type == "cuda")
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=_use_ac):
            pred = model.predict_batch(images, bboxes, sides)
        if pred.joints_3d.shape[0] == 0:
            return

        for k, idx in enumerate(pred.valid_idx):
            # 3D joint metrics
            pred_j = pred.joints_3d[k]
            gt_j = gt_joints_op[idx]
            pred_j_rel = pred_j - pred_j[0]
            gt_j_rel = gt_j - gt_j[0]
            mpjpe, pa_mpjpe = metrics_mm(pred_j_rel, gt_j_rel)   # root-relative (paper MPJPE)
            mpjpe_abs, _ = metrics_mm(pred_j, gt_j)              # absolute (no root subtract)

            # Single-hand root error (camera frame, mm). NOT paper MRRPE.
            root_err = mrrpe_mm(pred_j[0], gt_j[0])

            # Vertex metrics (if available)
            pve = pa_pve = None
            if pred.vertices_3d is not None and gt_verts[idx] is not None:
                pred_v = pred.vertices_3d[k]
                gt_v = gt_verts[idx]
                # Root-center vertices at wrist-equivalent
                pred_v_rel = pred_v - pred_j[0]
                gt_v_rel = gt_v - gt_j[0]
                pve, pa_pve = _compute_pve(pred_v_rel, gt_v_rel)

            # 2D pixel metric (if pred 2D present)
            p2d = None
            if pred.joints_2d_px is not None:
                pred_2d = pred.joints_2d_px[k]
                # gt_2d_proj is already in OpenPose order (projected from gt_joints_cam)
                gt_2d = j2ds[idx]
                p2d = _compute_p2d(pred_2d, gt_2d)

            pred_2d_save = None
            if pred.joints_2d_px is not None:
                pred_2d_save = pred.joints_2d_px[k]
            record = {
                "sequence_name": seqs[idx],
                "frame_id": fids[idx],
                "hand_side": sides[idx],
                "mpjpe_mm": mpjpe,               # root-relative (paper MPJPE)
                "pa_mpjpe_mm": pa_mpjpe,
                "root_err_mm": root_err,         # single-hand root L2 err (NOT paper MRRPE)
                "mpjpe_abs_mm": mpjpe_abs,       # absolute (no root subtract)
                "pve_mm": pve,
                "pa_pve_mm": pa_pve,
                "p2d_px": p2d,
                "pred_joints_3d": pred_j,        # (21, 3) OpenPose order
                "gt_joints_3d": gt_j,             # (21, 3) OpenPose order
                "pred_joints_2d_px": pred_2d_save,  # (21, 2) px OR None
            }
            if config.save_pred_vertices and pred.vertices_3d is not None:
                record["pred_vertices"] = pred.vertices_3d[k]
                record["gt_vertices"] = gt_verts[idx]

            # ── v2 MPS secondary GT (aria_dual_v2) ──────────────────────────
            # v2 joints are in OpenPose-21 order (MPS_TO_OP perm, palm_center at
            # OP[1]=THUMB_CMC placeholder). To avoid the ~40mm noise at that one
            # placeholder joint, metrics are computed on OP_VALID (20 indices
            # excluding index 1) only. Save raw v2 arrays to per_sample for
            # downstream analysis.
            gt_v2_2d_op_full = None
            entry = None
            if v2_inline:
                _row = buf_[idx]
                _v2_j3d = _row.get("joints_3d_v2")
                if _v2_j3d is not None and len(_v2_j3d) == 63:
                    entry = {
                        "j3d_v2_cam":  _v2_j3d,
                        "j2d_v2_disp": _row.get("joints_2d_v2") or [0.0]*42,
                        "confidence":  _row.get("confidence"),
                        "dt_ms":       None,
                        "rotation":    None,
                    }
            elif v2_lookup:
                v2_key = f"{seqs[idx]}/{fids[idx]}/{sides[idx]}"
                entry = v2_lookup.get(v2_key)
            if v2_inline or v2_lookup:
                if entry is None:
                    record["v2_missing"] = True
                else:
                    gt_v2_j3d = np.asarray(entry["j3d_v2_cam"], dtype=np.float32).reshape(21, 3)
                    gt_v2_j2d = np.asarray(entry["j2d_v2_disp"], dtype=np.float32).reshape(21, 2)
                    gt_v2_2d_op_full = gt_v2_j2d
                    # OP_VALID subset for fair comparison (skip placeholder at OP[1])
                    pred_j_v = pred_j[op_valid_np]
                    gt_v2_v = gt_v2_j3d[op_valid_np]
                    pred_j_v_rel = pred_j_v - pred_j_v[0]
                    gt_v2_v_rel = gt_v2_v - gt_v2_v[0]
                    mpjpe_v2, pa_mpjpe_v2 = metrics_mm(pred_j_v_rel, gt_v2_v_rel)
                    mpjpe_abs_v2, _ = metrics_mm(pred_j_v, gt_v2_v)
                    root_err_v2 = mrrpe_mm(pred_j_v[0], gt_v2_v[0])
                    p2d_v2 = None
                    if pred.joints_2d_px is not None:
                        pred_2d_v = pred.joints_2d_px[k][op_valid_np]
                        gt_v2_2d_v = gt_v2_j2d[op_valid_np]
                        p2d_v2 = _compute_p2d(pred_2d_v, gt_v2_2d_v)
                    record["mpjpe_v2_mm"]     = mpjpe_v2
                    record["pa_mpjpe_v2_mm"]  = pa_mpjpe_v2
                    record["root_err_v2_mm"]  = root_err_v2
                    record["mpjpe_abs_v2_mm"] = mpjpe_abs_v2
                    record["p2d_v2_px"]       = p2d_v2
                    record["gt_v2_joints_3d"] = gt_v2_j3d
                    record["gt_v2_joints_2d"] = gt_v2_j2d
                    record["v2_confidence"]   = entry.get("confidence")
                    record["v2_dt_ms"]        = entry.get("dt_ms")
                    record["v2_rotation"]     = entry.get("rotation")

            per_sample_records.append(record)

            # Viz sample (image + pred 2D + GT 2D). If config.viz_count == -1,
            # save ALL samples; otherwise reservoir-sample to the given cap.
            _vs = {
                "image": np.asarray(images[idx]),
                "gt_2d": j2ds[idx],
                "gt_v2_2d": gt_v2_2d_op_full,
                "pred_2d": pred.joints_2d_px[k] if pred.joints_2d_px is not None else None,
                "seq": seqs[idx],
                "frame_id": fids[idx],
                "hand_side": sides[idx],
                "pa_mpjpe": pa_mpjpe,
                "p2d": p2d,
                "pa_mpjpe_v2": record.get("pa_mpjpe_v2_mm"),
                "p2d_v2": record.get("p2d_v2_px"),
            }
            if config.viz_count < 0 or len(viz_samples) < config.viz_count:
                viz_samples.append(_vs)
            else:
                r = rng.randint(0, len(per_sample_records) - 1)
                if r < config.viz_count:
                    viz_samples[r] = _vs

        if config.show_progress and batch_count % config.log_every == 0:
            cur_pa_pf = np.mean([r["pa_mpjpe_mm"] for r in per_sample_records if np.isfinite(r["pa_mpjpe_mm"])]) if per_sample_records else float("nan")
            try:
                _iter.set_postfix_str(f"n={total} PA-MPJPE={cur_pa_pf:.2f}mm")
            except Exception:
                pass
            elapsed = time.time() - t_start
            cur_pa = np.mean([r["pa_mpjpe_mm"] for r in per_sample_records]) if per_sample_records else float("nan")
            extra = f", fit_res={np.mean(fit_residuals):.2f}mm" if fit_residuals else ""
            print(
                f"  [{dataset_name}/{model.name}] batch {batch_count}: "
                f"{total} samples, {elapsed:.0f}s, PA-MPJPE={cur_pa:.2f}mm{extra}"
            )

    # ── aria_manofit: local Drive lookup for precomputed fits ──────────────
    _local_fit_dir = dataset_spec.get("local_mano_fit_dir")
    _manofit_missing = 0

    def _attach_manofit(s: dict):
        nonlocal _manofit_missing
        import os as _os
        import pickle as _pickle
        seq = s["sequence_name"]
        fid = int(s["frame_id"])
        side = s["hand_side"]
        # S6 file convention: dual-hand frames use `_left`/`_right` suffix;
        # single-hand frames have no suffix (side stored in pkl's `hand_side` field).
        cands = [
            _os.path.join(_local_fit_dir, seq, f"{fid:05d}_{side}.pkl"),
            _os.path.join(_local_fit_dir, seq, f"{fid:05d}.pkl"),
        ]
        fp = None
        for c in cands:
            if _os.path.exists(c):
                fp = c
                break
        if fp is None:
            _manofit_missing += 1
            return None
        with open(fp, "rb") as _f:
            d = _pickle.load(_f)
        if "mano_fit_joints_21_op" not in d or "mano_fit_vertices" not in d:
            _manofit_missing += 1
            return None
        # For side-less pkl, sanity check the recorded side matches HF's side
        if fp.endswith(f"{fid:05d}.pkl") and d.get("hand_side") != side:
            _manofit_missing += 1
            return None
        s["mano_fit_joints_21_op"] = d["mano_fit_joints_21_op"]
        s["mano_fit_vertices"]     = d["mano_fit_vertices"]
        s["mano_fit_residual_mm"]  = d.get("mano_fit_residual_mm")
        return s

    _pbar_total = config.max_total if config.max_total else None
    _iter = tqdm(ds_iter, desc=f"[{dataset_name}/{model.name}]",
                 total=_pbar_total, unit="sample", dynamic_ncols=True)
    for sample in _iter:
        if not sample_filter(sample):
            continue
        if _local_fit_dir is not None:
            s2 = _attach_manofit(sample)
            if s2 is None:
                continue
            sample = s2
        if config.n_per_sequence is not None:
            if seq_count[sample["sequence_name"]] >= config.n_per_sequence:
                continue
            seq_count[sample["sequence_name"]] += 1
        buf.append(sample)
        total += 1
        if len(buf) >= config.batch_size:
            _process(buf)
            buf = []
        if config.max_total is not None and total >= config.max_total:
            break

    if buf:
        _process(buf)

    if _local_fit_dir is not None:
        print(f"  [{dataset_name}/{model.name}] skipped {_manofit_missing} missing manofit pkls")

    # Overall aggregation
    overall = _aggregate(per_sample_records)

    # Per-sequence
    by_seq = defaultdict(list)
    for r in per_sample_records:
        by_seq[r["sequence_name"]].append(r)
    per_seq_agg = {seq: _aggregate(records) for seq, records in by_seq.items()}

    # Per-hand-side
    by_hand = defaultdict(list)
    for r in per_sample_records:
        by_hand[r["hand_side"]].append(r)
    per_hand_agg = {hs: _aggregate(records) for hs, records in by_hand.items()}

    result = EvalResult(
        model_name=model.name,
        dataset_name=dataset_name,
        n_samples=len(per_sample_records),
        metrics=overall,
        per_sequence=per_seq_agg,
        per_hand_side=per_hand_agg,
        mano_fit_residual_mean=(float(np.mean(fit_residuals)) if fit_residuals else None),
    )

    # Persist to disk
    if config.out_dir:
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / f"{model.name}_{dataset_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  → wrote {metrics_path}")

        if config.save_per_sample:
            per_sample_path = out_dir / f"{model.name}_{dataset_name}_per_sample.pkl"
            # Strip large fields for light cache
            light = [
                {k: v for k, v in r.items()
                 if k not in ("pred_vertices", "gt_vertices")
                 or config.save_pred_vertices}
                for r in per_sample_records
            ]
            with open(per_sample_path, "wb") as f:
                pickle.dump(light, f)
            print(f"  → wrote {per_sample_path} ({len(light)} samples)")

        if (config.viz_count > 0 or config.viz_count < 0) and viz_samples:
            viz_dir = out_dir / "viz" / f"{model.name}_{dataset_name}"
            viz_dir.mkdir(parents=True, exist_ok=True)
            for i, vs in enumerate(viz_samples):
                title = f"{vs['seq']}/{vs['frame_id']}/{vs['hand_side']} PA={vs['pa_mpjpe']:.1f}mm"
                if vs["p2d"] is not None:
                    title += f" P2D={vs['p2d']:.0f}px"
                if vs.get("pa_mpjpe_v2") is not None:
                    title += f" | v2 PA={vs['pa_mpjpe_v2']:.1f}mm"
                if vs.get("p2d_v2") is not None:
                    title += f" P2D={vs['p2d_v2']:.0f}px"
                fname = f"{i:04d}_{vs['seq']}_{vs['frame_id']:05d}_{vs['hand_side']}.png"
                save_overlay_png(viz_dir, fname, vs["image"],
                                 gt_2d=vs["gt_2d"], pred_2d=vs["pred_2d"],
                                 gt_v2_2d=vs.get("gt_v2_2d"), title=title)
            print(f"  → wrote {len(viz_samples)} overlays to {viz_dir}")

    return result
