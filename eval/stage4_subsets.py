"""Post-process WH ARCTIC-parity eval output for Aria val into the
4-GT × 3-subset reporting matrix.

Inputs (produced by scripts/run_wh_aria_arctic.py, one set per variant):
    wh_aria_arctic_hamer_metrics.json  + _preds.pkl
    wh_aria_arctic_mps_metrics.json    + _preds.pkl
    wh_aria_arctic_mid_metrics.json    + _preds.pkl

Plus sidecar (from build_aria_val_arctic_npy.py):
    aria_val_arctic_sidecar.json      - vidx → (seq, frame_id, has_right, has_left)

Plus pre-registered keysets (artifacts/aria_val_agreement_keysets_125.json):
    clean_hand_keys / bad_hand_keys / all_hand_keys    (set of "seq/fid/side")

Output: a 4-column × 3-row × K-metric matrix
    columns: hamer, mps, mid, min
    rows:    clean, bad, all
    metrics: mpjpe.ra, mpjpe.pa.ra, mrrpe.rl, pix_err

The "min" column is per-sample min(err_hamer, err_mps), computed post-hoc.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


# WildHands metric_dict keys (per-hand, per-sample).
# See task #148 repro:
#   metric.mpjpe/ra/h         per-hand RA-MPJPE  (mean over R+L)
#   metric.mpjpe/pa/ra/h      per-hand PA-RA MPJPE
#   metric.mrrpe/r/l          R→L root positional error (pair-only)
#   metric.pix_err/h          per-hand 2D pixel error in crop space
PRIMARY_METRIC_KEYS = (
    "metric.mpjpe/ra/h",
    "metric.mpjpe/pa/ra/h",
    "metric.mrrpe/r/l",
    "metric.pix_err/h",
)


def load_keysets(path: str | Path) -> dict:
    with open(path) as f:
        k = json.load(f)
    return {
        "clean": set(k["clean_hand_keys"]),
        "bad":   set(k["bad_hand_keys"]),
        "all":   set(k["all_hand_keys"]),
        "threshold_px": k["threshold_px"],
        "n_clean_frames": k["n_clean_frames"],
        "n_bad_frames":   k["n_bad_frames"],
        "n_clean_hands": k.get("n_clean_hands", len(k["clean_hand_keys"])),
        "n_bad_hands":   k.get("n_bad_hands",   len(k["bad_hand_keys"])),
    }


def load_sidecar(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _key_for(entry: dict, side: str) -> str:
    return f"{entry['seq']}/{entry['frame_id']}/{side}"


def _per_sample_mask_by_subset(sidecar: dict, keysets: dict):
    """Return dict[subset] → np.bool_ arrays of length N_slots, one per side.

    The WH metric arrays are shaped (N_slots,) per-hand metric (mean of R+L),
    or (N_slots, 21) if per-joint is saved. For subset splitting we need
    per-*hand* masks, but the wrapper already reduces R+L into a single
    per-hand value via `/h`. To get per-side membership we instead split into
    right-only and left-only masks and combine per-side keyset membership.

    Returns:
        {
            "right": {subset: bool array (N,)},
            "left":  {subset: bool array (N,)},
        }
    For subset membership on a per-frame R-only/L-only basis, we check
    whether (seq, fid, side) is in each subset's key set AND whether that side
    was actually present in the fabricated GT (`has_right` / `has_left`).
    """
    entries = sidecar["entries"]
    N = len(entries)
    r_mask = {s: np.zeros(N, dtype=bool) for s in ("clean", "bad", "all")}
    l_mask = {s: np.zeros(N, dtype=bool) for s in ("clean", "bad", "all")}
    for i, e in enumerate(entries):
        if e["has_right"]:
            key = _key_for(e, "right")
            for s in ("clean", "bad", "all"):
                if key in keysets[s]:
                    r_mask[s][i] = True
        if e["has_left"]:
            key = _key_for(e, "left")
            for s in ("clean", "bad", "all"):
                if key in keysets[s]:
                    l_mask[s][i] = True
    return {"right": r_mask, "left": l_mask}


def _load_variant(out_dir: Path, variant: str):
    m_path = out_dir / f"wh_aria_arctic_{variant}_metrics.json"
    p_path = out_dir / f"wh_aria_arctic_{variant}_preds.pkl"
    with open(m_path) as f:
        metrics = json.load(f)
    with open(p_path, "rb") as f:
        preds = pickle.load(f)
    # Convert JSON-ed arrays back to np
    out = {k: np.asarray(v, dtype=np.float32) for k, v in metrics.items()
           if isinstance(v, list)}
    return out, preds


def _agg(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"mean": None, "median": None, "std": None, "n": 0, "finite": 0.0}
    sub = arr[finite]
    return {
        "mean":   float(sub.mean()),
        "median": float(np.median(sub)),
        "std":    float(sub.std()),
        "n":      int(sub.size),
        "finite": float(finite.mean()),
    }


def _per_side_metrics(variant_metrics: dict) -> dict:
    """Some metrics are per-hand aggregates (/h). Also look for /r and /l variants.
    Return {metric_base: {'r': array, 'l': array, 'h': array}}.
    """
    grouped: dict = {}
    for k, arr in variant_metrics.items():
        if not k.startswith("metric."):
            continue
        # Split off the per-hand suffix
        # e.g. metric.mpjpe/ra/h  → base=metric.mpjpe/ra, side=h
        parts = k.rsplit("/", 1)
        if len(parts) == 2 and parts[1] in ("r", "l", "h"):
            base, side = parts[0], parts[1]
        else:
            base, side = k, "h"  # default: treat as per-hand aggregate
        grouped.setdefault(base, {})[side] = arr
    return grouped


def build_subset_matrix(out_dir: str | Path,
                        sidecar_path: str | Path,
                        keysets_path: str | Path,
                        variants: Iterable[str] = ("hamer", "mps", "mid"),
                        metric_bases: Iterable[str] = (
                            "metric.mpjpe/ra",
                            "metric.mpjpe/pa/ra",
                            "metric.pix_err",
                        )) -> dict:
    out_dir = Path(out_dir)
    sidecar = load_sidecar(sidecar_path)
    keysets = load_keysets(keysets_path)
    masks = _per_sample_mask_by_subset(sidecar, keysets)

    # Per-variant metric grouping
    variant_data = {}
    for v in variants:
        m, preds = _load_variant(out_dir, v)
        grouped = _per_side_metrics(m)
        variant_data[v] = {"grouped": grouped, "preds": preds, "raw": m}

    matrix = {
        "threshold_px": keysets["threshold_px"],
        "n_clean_frames": keysets["n_clean_frames"],
        "n_bad_frames":   keysets["n_bad_frames"],
        "n_clean_hands":  keysets["n_clean_hands"],
        "n_bad_hands":    keysets["n_bad_hands"],
        "columns": list(variants) + ["min"],
        "rows": ["clean", "bad", "all"],
        "metrics": {},
    }

    # Per-metric × per-variant × per-subset aggregation.
    for base in metric_bases:
        matrix["metrics"][base] = {}
        for col in list(variants) + ["min"]:
            matrix["metrics"][base][col] = {}
            for s in ("clean", "bad", "all"):
                vals = []
                # Prefer per-side /r /l splits when present (finer-grained)
                has_rl = any(
                    variant_data[v]["grouped"].get(base, {}).get("r") is not None
                    and variant_data[v]["grouped"].get(base, {}).get("l") is not None
                    for v in variants
                )
                if has_rl:
                    for side in ("r", "l"):
                        side_key = "right" if side == "r" else "left"
                        subset_mask = masks[side_key][s]
                        if col == "min":
                            a = variant_data["hamer"]["grouped"].get(base, {}).get(side)
                            b = variant_data["mps"]["grouped"].get(base, {}).get(side)
                            if a is None or b is None:
                                continue
                            arr = np.minimum(a, b)
                        else:
                            arr = variant_data[col]["grouped"].get(base, {}).get(side)
                            if arr is None:
                                continue
                        if arr.ndim > 1:
                            arr = np.nanmean(arr, axis=tuple(range(1, arr.ndim)))
                        vals.append(arr[subset_mask])
                else:
                    # Fall back to /h (per-hand aggregate, one value per frame).
                    # Frame is "in subset" if either side-slot is in that subset's keys.
                    frame_mask = masks["right"][s] | masks["left"][s]
                    if col == "min":
                        a = variant_data["hamer"]["grouped"].get(base, {}).get("h")
                        b = variant_data["mps"]["grouped"].get(base, {}).get("h")
                        if a is None or b is None:
                            arr = None
                        else:
                            arr = np.minimum(a, b)
                    else:
                        arr = variant_data[col]["grouped"].get(base, {}).get("h")
                    if arr is not None:
                        if arr.ndim > 1:
                            arr = np.nanmean(arr, axis=tuple(range(1, arr.ndim)))
                        vals.append(arr[frame_mask])
                matrix["metrics"][base][col][s] = _agg(
                    np.concatenate(vals) if vals else np.asarray([], dtype=np.float64)
                )

    # MRRPE - paired metric, pair-level (not per-side). Handle separately:
    # any frame where BOTH R and L are in the same subset contributes.
    mrrpe_base = "metric.mrrpe"
    matrix["metrics"][f"{mrrpe_base}/r_l"] = {}
    entries = sidecar["entries"]
    N = len(entries)
    pair_subset: dict = {s: np.zeros(N, dtype=bool) for s in ("clean", "bad", "all")}
    for i, e in enumerate(entries):
        if e["has_right"] and e["has_left"]:
            rk = _key_for(e, "right")
            lk = _key_for(e, "left")
            for s in ("clean", "bad", "all"):
                if rk in keysets[s] and lk in keysets[s]:
                    pair_subset[s][i] = True

    for col in list(variants) + ["min"]:
        matrix["metrics"][f"{mrrpe_base}/r_l"][col] = {}
        if col == "min":
            a = variant_data["hamer"]["raw"].get("metric.mrrpe/r/l")
            b = variant_data["mps"]["raw"].get("metric.mrrpe/r/l")
            arr = None if (a is None or b is None) else np.minimum(
                np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            )
        else:
            arr = variant_data[col]["raw"].get("metric.mrrpe/r/l")
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float32)
        for s in ("clean", "bad", "all"):
            if arr is None:
                matrix["metrics"][f"{mrrpe_base}/r_l"][col][s] = {
                    "mean": None, "median": None, "std": None, "n": 0, "finite": 0.0
                }
            else:
                matrix["metrics"][f"{mrrpe_base}/r_l"][col][s] = _agg(arr[pair_subset[s]])

    return matrix


def print_matrix(matrix: dict) -> None:
    print(f"\n=== WH-Aria Stage 4 matrix  (threshold={matrix['threshold_px']} px) ===")
    print(f"  clean_frames={matrix['n_clean_frames']} "
          f"bad_frames={matrix['n_bad_frames']}  "
          f"clean_hands={matrix['n_clean_hands']}  "
          f"bad_hands={matrix['n_bad_hands']}")
    cols = matrix["columns"]
    rows = matrix["rows"]
    for metric, per_col in matrix["metrics"].items():
        print(f"\n  metric = {metric}")
        hdr = f"    {'subset':<8s}"
        for c in cols:
            hdr += f" {c:>12s}"
        print(hdr)
        for s in rows:
            row = f"    {s:<8s}"
            for c in cols:
                cell = per_col.get(c, {}).get(s, {})
                m = cell.get("mean")
                n = cell.get("n", 0)
                row += f" {('--' if m is None else f'{m:7.2f}') + f' n={n:>4d}':>12s}"
            print(row)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--keysets", required=True)
    ap.add_argument("--save_json", default=None)
    a = ap.parse_args()
    matrix = build_subset_matrix(a.out_dir, a.sidecar, a.keysets)
    print_matrix(matrix)
    if a.save_json:
        with open(a.save_json, "w") as f:
            json.dump(matrix, f, indent=2)
        print(f"\n[saved] {a.save_json}")
