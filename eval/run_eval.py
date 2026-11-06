"""Runpod 5-model zero-shot eval on aria_v1_locked (NEW IAA-curated v1 dataset).

Adapted from eval_arena_run.py (which used aria_val_v2_clean / sidecar v2 lookup).
Uses inline v2 GT mode - no sidecar needed.

Output:
  /workspace/eval_v1_locked_out/
    {model}_aria_v1_locked_metrics.json
    {model}_aria_v1_locked_per_sample.pkl
    SUMMARY.json
"""
from __future__ import annotations
import os, sys, time, json, shutil, gc
from pathlib import Path

REPRO = "/workspace/repro"
HAMER_REPO = f"{REPRO}/hamer"
WILOR_REPO = f"{REPRO}/WiLoR"
WH_REPO    = f"{REPRO}/WildHands"
JT_REPO    = f"{REPRO}/JointTransformer"
SH_REPO    = f"{REPRO}/simpleHand"
PHASE0_REPO = f"{REPRO}/phase0-eval"
MANO_DIR    = f"{REPRO}/weights/mano"

# HF_TOKEN
if not os.environ.get("HF_TOKEN"):
    for cand in ["/workspace/root_backup/env.sh", "/root/env.sh"]:
        if os.path.exists(cand):
            for line in open(cand):
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k in ("HF_TOKEN", "export HF_TOKEN"):
                        os.environ.setdefault("HF_TOKEN", v.strip().strip('"').strip("'").replace("export ", ""))
                    elif k.startswith("export "):
                        os.environ.setdefault(k[7:], v.strip().strip('"').strip("'"))
                    else:
                        os.environ.setdefault(k, v.strip().strip('"').strip("'"))
            break
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing - source /workspace/root_backup/env.sh"

# Set up output dir based on dataset choice
import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--dataset", default="aria_v1_locked", choices=["aria_v1_locked", "aria_v1_legacy_val", "pov_test", "aria_v2_train_mps", "aria_v2_val"])
_ap.add_argument("--models", nargs="+", default=None)
_ap.add_argument("--max", type=int, default=None)
_ap.add_argument("--batch", type=int, default=32)
_ap.add_argument("--viz", type=int, default=0)
_args, _ = _ap.parse_known_args()

OUT_DIR = f"/workspace/eval_v1_out/{_args.dataset}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[setup] dataset={_args.dataset}  out={OUT_DIR}")

sys.path.insert(0, PHASE0_REPO)
for p in [HAMER_REPO, WILOR_REPO, WH_REPO, SH_REPO]:
    if p not in sys.path:
        sys.path.insert(0, p)

import phase0.models.simplehand_model as _shm
_shm._SH_PATH = SH_REPO
if SH_REPO not in sys.path:
    sys.path.insert(0, SH_REPO)

from phase0.pytorch3d_stub import install as _install_p3d_stub
_install_p3d_stub()

import torch
import numpy as np
from phase0 import DEFAULT_DATASETS
from phase0.eval_loop import EvalConfig, eval_stream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[setup] device={device}")

SPEC_NAME = _args.dataset
SPEC = DEFAULT_DATASETS[SPEC_NAME]
print(f"[reg] using {SPEC_NAME}: filter applied at iteration time")


def _clear_module(prefix):
    for k in list(sys.modules):
        if k == prefix or k.startswith(prefix + "."):
            del sys.modules[k]


def load_hamer_model():
    cwd = os.getcwd()
    os.chdir(REPRO)
    try:
        from hamer.models import load_hamer as _lh
        from phase0.models.hamer_model import HaMeRModel
        # Path adapted: ckpt is at weights/hamer_ckpts/hamer.ckpt on this pod
        ckpt = f"{REPRO}/weights/hamer_ckpts/hamer.ckpt"
        if not os.path.exists(ckpt):
            ckpt = f"{REPRO}/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
        m, cfg = _lh(ckpt)
        m = m.to(device).eval()
        return HaMeRModel(m, cfg, device)
    finally:
        os.chdir(cwd)


def load_wilor_model():
    cwd = os.getcwd()
    os.chdir(WILOR_REPO)
    try:
        _clear_module("wilor")
        from wilor.models import load_wilor as _lw
        from phase0.models.wilor_model import WiLoRModel
        m, cfg = _lw(
            checkpoint_path=f"{REPRO}/weights/wilor/wilor_final.ckpt",
            cfg_path=f"{REPRO}/weights/model_config.yaml",
        )
        m = m.to(device).eval()
        return WiLoRModel(m, cfg, device)
    finally:
        os.chdir(cwd)


def load_wildhands_model():
    _clear_module("wildhands")
    os.environ.setdefault("WILDHANDS_MANO_DIR", MANO_DIR)
    cwd = os.getcwd()
    os.chdir(WH_REPO)
    try:
        from wildhands.configs.parser import construct_args
        from wildhands.models.wrapper import WildHandsWrapper as Wrapper
        from phase0.models.wildhands_model import WildHandsModel
        cfg = construct_args()
        model = Wrapper(cfg)
        ckpt = torch.load(f"{REPRO}/weights/wildhands.ckpt", map_location="cpu", weights_only=False)
        redundant = ("head_o", "arti_head", "grasp_classifier")
        state = {k: v for k, v in ckpt["state_dict"].items()
                 if not any(r in k for r in redundant)}
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()
        return WildHandsModel(model, cfg, device)
    finally:
        os.chdir(cwd)


def load_jt_model():
    _clear_module("src")
    from phase0.models.jointtransformer_setup import bootstrap, load_transformer_sf
    from phase0.models.jointtransformer_model import JointTransformerModel
    bootstrap(jt_repo=JT_REPO, mano_src_dir=MANO_DIR)
    m, hp = load_transformer_sf(
        ckpt_path=f"{REPRO}/weights/jointtransformer/final_p2_462fc683e.ckpt",
        device=device,
    )
    return JointTransformerModel(m, device, img_res=hp["img_res"], crop_size=hp["img_res"])


def load_simplehand_model():
    cwd = os.getcwd()
    os.chdir(SH_REPO)
    try:
        _clear_module("hand_net")
        _clear_module("cfg")
        from cfg import _CONFIG
        from hand_net import HandNet
        from phase0.models.simplehand_model import SimpleHandModel
        m = HandNet(_CONFIG, pretrained=False)
        ck = torch.load(
            f"{REPRO}/weights/simplehand/simpleHand_epoch_200.pt",
            map_location="cpu", weights_only=False,
        )
        sd = ck.get("state_dict", ck)
        m.load_state_dict(sd, strict=True)
        m = m.to(device).eval()
        return SimpleHandModel(m, device, crop_size=224)
    finally:
        os.chdir(cwd)


LOADERS = [
    ("hamer", load_hamer_model),
    ("wilor", load_wilor_model),
    ("wildhands", load_wildhands_model),
    ("jointtransformer", load_jt_model),
    ("simplehand", load_simplehand_model),
]


def run():
    summary = {}
    models_to_run = LOADERS if _args.models is None else [(n, f) for n, f in LOADERS if n in _args.models]
    for name, loader in models_to_run:
        print(f"\n{'='*72}\n[{name}] LOADING\n{'='*72}")
        t_load = time.time()
        try:
            wrapper = loader()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{name}] LOAD FAILED: {e}")
            continue
        print(f"[{name}] loaded in {time.time()-t_load:.1f}s")

        cfg = EvalConfig(
            batch_size=_args.batch,
            max_total=_args.max,
            log_every=5,
            show_progress=True,
            hf_token=os.environ["HF_TOKEN"],
            out_dir=OUT_DIR,
            save_per_sample=True,
            viz_count=_args.viz,
            save_pred_vertices=False,
        )
        print(f"[{name}] EVAL on {SPEC_NAME}")
        t_eval = time.time()
        try:
            r = eval_stream(wrapper, SPEC, SPEC_NAME, device, cfg, mano_fitter=None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{name}] EVAL FAILED: {e}")
            del wrapper; gc.collect(); torch.cuda.empty_cache()
            continue
        dt_eval = time.time() - t_eval
        summary[name] = {
            "n_samples": r.n_samples,
            "elapsed_s": dt_eval,
            "metrics": r.metrics,
        }
        print(f"[{name}] DONE n={r.n_samples} in {dt_eval:.0f}s")
        for metric, vals in r.metrics.items():
            if vals and isinstance(vals, dict) and vals.get("mean") is not None:
                print(f"  {metric:20s}: {vals['mean']:7.3f}  (n={vals['n']})")
        del wrapper; gc.collect(); torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n==== {SPEC_NAME} | primary GT (HaMeR-derived joints_3d) ====")
    cols = ["mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "pa_pve_mm", "p2d_px", "mpjpe_abs_mm", "mrrpe_paper_mm"]
    print("| model            | " + " | ".join(f"{c:>14s}" for c in cols) + " |    n |")
    print("|" + "-"*150 + "|")
    for name, d in summary.items():
        cells = []
        for c in cols:
            v = d["metrics"].get(c, {})
            v = v.get("mean") if isinstance(v, dict) else None
            cells.append(f"{v:14.3f}" if v is not None else "             --")
        print(f"| {name:16s} | " + " | ".join(cells) + f" | {d['n_samples']:4d} |")

    print(f"\n==== {SPEC_NAME} | secondary GT (Aria MPS joints_3d_v2) ====")
    cols_v2 = ["mpjpe_v2_mm", "pa_mpjpe_v2_mm", "mpjpe_abs_v2_mm", "p2d_v2_px"]
    print("| model            | " + " | ".join(f"{c:>14s}" for c in cols_v2) + " |    n |")
    print("|" + "-"*100 + "|")
    for name, d in summary.items():
        cells = []
        for c in cols_v2:
            v = d["metrics"].get(c, {})
            v = v.get("mean") if isinstance(v, dict) else None
            cells.append(f"{v:14.3f}" if v is not None else "             --")
        print(f"| {name:16s} | " + " | ".join(cells) + f" | {d['n_samples']:4d} |")


if __name__ == "__main__":
    run()
