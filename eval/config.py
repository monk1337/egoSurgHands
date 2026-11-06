"""Constants: joint conventions, HF repo IDs, default dataset configs."""

# ─────────────────────────────────────────────────────────────────────────────
# Joint convention mappings
# ─────────────────────────────────────────────────────────────────────────────
#
# MANO output from smplx/HaMeR/WiLoR/WildHands is in MANO internal order:
#   0: wrist
#   1-3: index (mcp, pip, dip)
#   4-6: middle (mcp, pip, dip)
#   7-9: pinky (mcp, pip, dip)
#   10-12: ring (mcp, pip, dip)
#   13-15: thumb (cmc, mcp, ip)
#   16-20: fingertips (thumb, index, middle, ring, pinky)
#
# OpenPose / MediaPipe hand convention:
#   0: wrist
#   1-4: thumb (cmc, mcp, ip, tip)
#   5-8: index (mcp, pip, dip, tip)
#   9-12: middle (mcp, pip, dip, tip)
#   13-16: ring (mcp, pip, dip, tip)
#   17-20: pinky (mcp, pip, dip, tip)
#
# Use: joints_op = joints_mano[MANO_TO_OPENPOSE]  (21 → 21 reorder)
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

# Aria MPS hand landmark convention (per projectaria_tools):
#   0-4: fingertips (thumb, index, middle, ring, pinky)
#   5: wrist
#   6-7: thumb intermediate + distal
#   8-10: index (proximal, intermediate, distal)
#   11-13: middle
#   14-16: ring
#   17-19: pinky
#   20: palm center (no OpenPose analog - skipped)
#
# Aria MPS does NOT have THUMB_CMC, so OpenPose index 1 (THUMB_CMC) is not
# filled. Use OP_VALID = [0, 2..20] (20 indices) for fitting/comparison.
ARIA_MPS_TO_OPENPOSE = {
    5: 0,    # WRIST
    6: 2,    # THUMB_INTERMEDIATE → THUMB_MCP
    7: 3,    # THUMB_DISTAL → THUMB_IP
    0: 4,    # THUMB_FINGERTIP → THUMB_TIP
    8: 5,    # INDEX_PROXIMAL → INDEX_MCP
    9: 6,    # INDEX_INTERMEDIATE → INDEX_PIP
    10: 7,   # INDEX_DISTAL → INDEX_DIP
    1: 8,    # INDEX_FINGERTIP → INDEX_TIP
    11: 9,   # MIDDLE_PROXIMAL → MIDDLE_MCP
    12: 10,
    13: 11,
    2: 12,
    14: 13,  # RING_PROXIMAL
    15: 14,
    16: 15,
    3: 16,
    17: 17,  # PINKY_PROXIMAL
    18: 18,
    19: 19,
    4: 20,
}
OP_VALID = sorted(ARIA_MPS_TO_OPENPOSE.values())  # 20 OpenPose indices with an Aria analog

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace datasets
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASETS = {
    "aria_val": {
        # Primary: stored SAM2+HaMeR pseudo-GT (MANO-consistent, no MANOFitter offset)
        "repo_id": "harvardsil/aria-surgical-hand-pose",
        "split": "validation",
        "gt_field": "joints_3d",
        "gt_order": "aria_stored",
        "filter": lambda s: s.get("has_eval_gt", False),
        "needs_mano_fit": False,
    },
    "aria_val_mps": {
        # Secondary: MPS landmarks via MANOFitter (has ~40mm joint-convention offset)
        "repo_id": "harvardsil/aria-surgical-hand-pose",
        "split": "validation",
        "gt_field": "eval_joints_3d",
        "gt_order": "aria_mps",
        "filter": lambda s: s.get("has_eval_gt", False),
        "needs_mano_fit": True,
    },
    "pov_test": {
        "repo_id": "harvardsil/pov-surgery",
        "split": "test",
        "gt_field": "joints_3d",          # native MANO synthetic GT
        "gt_order": "mano",               # reorder to OpenPose directly
        "filter": None,
        "needs_mano_fit": False,
    },
    "aria_val_manofit": {
        # PRIMARY eval target: precomputed MANO fits to Aria MPS landmarks (S6 pipeline).
        # GT is already in OpenPose-21 order, includes vertices for PVE.
        # Fitter residual ~5mm noise floor (report alongside model numbers).
        # Per-sample lookup from local Drive dir; HF is only used for images.
        "repo_id": "harvardsil/aria-surgical-hand-pose",
        "split": "validation",
        "gt_field": "mano_fit_joints_21_op",
        "gt_order": "aria_manofit",
        "filter": lambda s: s.get("has_eval_gt", False),
        "needs_mano_fit": False,
        "local_mano_fit_dir": "/content/drive/MyDrive/phase0/mano_fitted/s6_out",
    },
    "aria_val_v2": {
        # DUAL-GT eval: report metrics against BOTH
        #   (a) stored SAM2+HaMeR joints_3d (existing "aria_stored" behavior), AND
        #   (b) v2 MPS rebuild  (2026-04-19: per-PR Δt + per-PR rotation +
        #       MPS→MANO permutation).
        # Per-sample v2 fields are loaded from an HF-hosted lookup JSON keyed by
        # "PR/frame_id/side". v2 j3d is in the Aria CAMERA frame, MANO-ordered;
        # v2 j2d is in DISPLAY-frame pixels that match the stored JPEGs.
        # See paper/03_data_processing/aria_mps_projection_history.md.
        "repo_id": "harvardsil/aria-surgical-hand-pose",
        "split": "validation",
        "gt_field": "joints_3d",
        "gt_order": "aria_dual_v2",
        "filter": lambda s: s.get("has_eval_gt", False),
        "needs_mano_fit": False,
        "v2_lookup_repo": "aaditya/phase0-artifacts",
        "v2_lookup_file": "mps_v2_val_20260419.json",
    },
    "aria_v1_locked": {
        # NEW IAA-CURATED HEADLINE BENCHMARK (v1 dataset, May 2026).
        # Filter: in_locked_eval=True (PR108-185, per-hand IAA-verified).
        # 10,420 rows, 100% dual-GT (joints_3d HaMeR + joints_3d_v2 MPS inline).
        # No sidecar - v2 read directly from row.
        "repo_id": "harvardsil/aria-surgical-hand-pose-v1",
        "split": "validation",
        "gt_field": "joints_3d",
        "gt_order": "aria_dual_v2",
        "filter": lambda s: s.get("in_locked_eval", False),
        "needs_mano_fit": False,
        "v2_inline": True,  # read joints_3d_v2 from row, not sidecar
    },
    "aria_v1_legacy_val": {
        # LEGACY VAL (v1 dataset, May 2026): PR81-85 only, 3,259 rows, 100% dual-GT.
        # Comparable to aria_val_v2_clean used in FT matrix winner (19.56/15.88).
        "repo_id": "harvardsil/aria-surgical-hand-pose-v1",
        "split": "validation",
        "gt_field": "joints_3d",
        "gt_order": "aria_dual_v2",
        "filter": lambda s: s.get("source_tag") == "oldtrain" and s.get("pr", "").startswith("PR8"),
        "needs_mano_fit": False,
        "v2_inline": True,
    },
    "aria_v2_train_mps": {
        "local_parquet_glob": "/workspace/v2_clean_v2/train/*.parquet",
        "split": "train",
        "gt_field": "joints_3d",
        "gt_order": "aria_dual_v2",
        "filter": lambda s: s.get("has_eval_gt", False),
        "needs_mano_fit": False,
        "v2_inline": True,
    },
    "aria_v2_val": {
        "local_parquet_glob": "/workspace/v2_clean_v2/validation/*.parquet",
        "split": "validation",
        "gt_field": "joints_3d",
        "gt_order": "aria_dual_v2",
        "filter": lambda s: s.get("has_eval_gt", False),
        "needs_mano_fit": False,
        "v2_inline": True,
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Model registry (HF repo → local paths after setup.sh)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINTS = {
    "hamer": {
        "repo": "https://github.com/geopavlakos/hamer.git",
        "ckpt": "_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
        "vitpose_ckpt": "_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth",
    },
    "wilor": {
        "repo": "https://github.com/rolpotamias/WiLoR.git",
        "ckpt": "pretrained_models/wilor_final.ckpt",
        "cfg": "pretrained_models/model_config.yaml",
    },
    "wildhands": {
        "repo": "https://github.com/ap229997/hands.git",
        "branch": "demo",
        "ckpt": "downloads/wildhands/wildhands.ckpt",
        "intrx": "downloads/wildhands/intrx.pkl",
    },
}
