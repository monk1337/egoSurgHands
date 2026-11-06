"""Atomic run helper: run one model/dataset combo, stage artifacts, upload to HF.

Extracted from the notebook prototype (scratchpad session 2026-04-17/18) and
promoted here so reruns don't require re-executing notebook cells.

Usage:
    from phase0.runner import run_atomic
    metrics = run_atomic(hamer_wrapped, 'aria_val_manofit',
                         tag='full_b256_v3', batch_size=256)
"""
from __future__ import annotations
import os
import json
import time
import glob
import shutil
import subprocess
from typing import Optional

import torch

from .config import DEFAULT_DATASETS
from .eval_loop import EvalConfig, eval_stream

DRIVE_RUNS = "/content/drive/MyDrive/phase0/runs"
TAR_STAGE = "/content/_tarstage"
HF_REPO_DEFAULT = "aaditya/phase0-artifacts"


def run_atomic(
    model_wrap,
    dataset_key: str,
    *,
    tag: str,
    batch_size: int = 48,
    n_per_seq: Optional[int] = None,
    max_total: Optional[int] = None,
    viz_count: int = 20,
    mano_fitter=None,
    hf_repo: str = HF_REPO_DEFAULT,
    drive_runs: str = DRIVE_RUNS,
    tar_stage: str = TAR_STAGE,
    device=None,
    upload: bool = True,
) -> dict:
    """Run one eval, save to Drive, tar locally, upload to HF.

    Returns the enriched metrics dict (also written to disk).
    """
    os.makedirs(drive_runs, exist_ok=True)
    os.makedirs(tar_stage, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M")
    run_name = f"{model_wrap.name}__{dataset_key}__{tag}_{ts}"
    out_dir = f"{drive_runs}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = EvalConfig(
        batch_size=batch_size,
        n_per_sequence=n_per_seq,
        max_total=max_total,
        hf_token=os.environ.get("HF_TOKEN"),
        out_dir=out_dir,
        save_per_sample=True,
        viz_count=viz_count,
        save_pred_vertices=False,
    )

    ds_spec = DEFAULT_DATASETS[dataset_key]
    fitter = mano_fitter if ds_spec.get("needs_mano_fit") else None

    print(f'\n{"=" * 70}\nRUN {run_name}\n  out={out_dir}\n  '
          f'B={batch_size}  n_per_seq={n_per_seq}  max_total={max_total}\n{"=" * 70}\n')
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = eval_stream(model_wrap, ds_spec, dataset_key, device, cfg, mano_fitter=fitter)
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    metrics = r.to_dict()
    metrics.update({
        "_run_name": run_name,
        "_dataset_key": dataset_key,
        "_model": model_wrap.name,
        "_tag": tag,
        "_elapsed_s": elapsed,
        "_peak_vram_gb": peak_gb,
        "_batch_size": batch_size,
        "_n_per_seq": n_per_seq,
        "_max_total": max_total,
        "_fp16": getattr(model_wrap, "use_autocast", True),
    })

    jpath = next(iter(glob.glob(f"{out_dir}/*_metrics.json")),
                 f"{out_dir}/{model_wrap.name}_{dataset_key}_metrics.json")
    with open(jpath, "w") as f:
        json.dump(metrics, f, indent=2)

    # Stage locally, then tar (avoid Drive fuse tar bug).
    stage = f"{tar_stage}/{run_name}"
    if os.path.exists(stage):
        shutil.rmtree(stage)
    shutil.copytree(out_dir, stage)
    tar_path = f"/content/{run_name}.tar.gz"
    subprocess.run(["tar", "-czf", tar_path, "-C", tar_stage, run_name], check=True)
    size_mb = os.path.getsize(tar_path) / 1e6
    print(f"\n  tar: {tar_path}  ({size_mb:.1f} MB)  peak VRAM: {peak_gb:.1f} GB")

    if upload and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.upload_file(
                path_or_fileobj=tar_path,
                path_in_repo=f"runs/{run_name}.tar.gz",
                repo_id=hf_repo, repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj=jpath,
                path_in_repo=f"metrics/{run_name}.json",
                repo_id=hf_repo, repo_type="dataset",
            )
            print(f"  HF: runs/{run_name}.tar.gz + metrics/{run_name}.json")
        except Exception as e:
            print(f"  HF upload err: {e}")
    try:
        os.remove(tar_path)
    except OSError:
        pass
    try:
        shutil.rmtree(stage)
    except OSError:
        pass

    ov = metrics.get("overall", {})
    def g(k):
        v = ov.get(k, {})
        return v.get("mean", float("nan")) if isinstance(v, dict) else float("nan")

    print(f"\n  === {run_name} ===")
    print(f"  n_samples       : {metrics.get('n_samples')}")
    print(f"  PA-MPJPE        : {g('pa_mpjpe_mm'):.2f} mm  (vs primary GT)")
    print(f"  MPJPE           : {g('mpjpe_mm'):.2f} mm")
    print(f"  PA-PVE          : {g('pa_pve_mm'):.2f} mm")
    print(f"  PVE             : {g('pve_mm'):.2f} mm")
    print(f"  P2D             : {g('p2d_px'):.1f} px  (vs stored joints_2d)")
    print(f"  root_err        : {g('root_err_mm'):.2f} mm")
    # v2 MPS secondary GT (aria_dual_v2)
    pa_v2 = g("pa_mpjpe_v2_mm")
    if pa_v2 == pa_v2:  # NaN check
        print(f"  PA-MPJPE v2     : {pa_v2:.2f} mm  (vs v2 MPS GT, OP_VALID 20)")
        print(f"  MPJPE    v2     : {g('mpjpe_v2_mm'):.2f} mm")
        print(f"  P2D      v2     : {g('p2d_v2_px'):.1f} px  (vs eval_joints_2d_v2)")
        print(f"  root_err v2     : {g('root_err_v2_mm'):.2f} mm")
    fr = metrics.get("mano_fit_residual_mm")
    if fr and isinstance(fr, dict):
        print(f"  fitter residual : median={fr.get('median', float('nan')):.2f} mm (noise floor)")
    print(f"  elapsed         : {elapsed:.1f} s "
          f"({(metrics.get('n_samples') or 0) / max(elapsed, 1):.1f} samples/s)")
    return metrics
