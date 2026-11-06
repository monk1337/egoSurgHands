# EgoSurgHands - Reproduction Code

Minimal pipeline to reproduce all numbers in the EgoSurgHands NeurIPS 2026 paper.

- **Paper:** *EgoSurgHands: An Egocentric 3D Hand Pose Dataset & Benchmark for Open-Surgery Training*, NeurIPS 2026, Evaluations & Datasets Track.
- **Dataset:** https://huggingface.co/datasets/aaditya/EgoSurgHands (gated: request access on the page, then `huggingface-cli login`).
- **Project page and leaderboard:** https://egosurghands.github.io

> **Headline:** WildHands+JRN reaches 10.40 mm PA-MPJPE-MPS and 75.8 mm absolute
> MPJPE on the v2 evaluation partition (9,476 hand instances, 13 PRs) - a 49.2% reduction
> in PA-MPJPE and an 85.8% reduction in absolute MPJPE vs WildHands ZS. Two
> ego-pretrained backbones converge at ~10.5 mm PA-MPJPE and ~77 mm absolute
> MPJPE with overlapping bootstrap CIs.

## What's in this repo

```
EgoSurgHands/
├── README.md              ← this file
├── setup.sh               ← downloads dataset + clones 6 backbone repos + ckpts
├── requirements.txt       ← Python deps (torch + datasets + pyarrow + matplotlib)
├── reproduce.sh           ← one-shot: setup → infer all 6 backbones → JRN → figures
│
├── eval/                  ← zero-shot inference
│   ├── run_eval.py        ← runs any of {hamer, wilor, wildhands, jt, simplehand}
│   ├── run_meshgraphormer.py  ← MeshGraphormer (different model construction)
│   ├── eval_loop.py       ← shared inference loop (from phase0-eval)
│   ├── geometry.py        ← Umeyama Procrustes (the literature standard)
│   ├── config.py          ← dataset specs (aria_v2_train_mps, aria_v2_val)
│   └── models/            ← 6 backbone wrappers
│       ├── base.py
│       ├── hamer_model.py
│       ├── wilor_model.py     ← post-fix wrapper (mano_to_op double-remap removed)
│       ├── wildhands_model.py
│       ├── jointtransformer_model.py
│       ├── simplehand_model.py
│       └── meshgraphormer_model.py
│
├── jrn/                   ← Joint-Residual Network (58K-param two-headed MLP: pose + translation)
│   ├── train_jrn.py       ← train+eval JRN per backbone (canonical headline)
│   └── pa_p2d.py          ← PA-P2D (Procrustes-aligned 2D pixel error)
│
├── analysis/              ← per-PR / per-task / per-strata
│   ├── per_pr.py
│   ├── per_task.py
│   └── per_strata.py
│
├── figures/               ← regenerate F2-F8 from saved summaries
│   └── make_figures.py
│
└── examples/              ← reference outputs (frozen numbers)
    └── headline.json
```

## Hardware

Realistic VRAM use: peak ~16 GB during the JointTransformer (ViT-G) zero-shot inference; everything else under 12 GB. JRN training itself is trivial (~50 MB, 6 sec on any GPU).

| GPU | VRAM | Wallclock (full reproduce) | Notes |
|---|---:|---:|---|
| RTX 4090 / RTX 3090 / A10G | 24 GB | ~4-5 hr | **best $/result**; JT/HaMeR/WiLoR may need batch=16 |
| **A100 40 GB / A40 48 GB**       | 40 GB | ~3 hr | **recommended sweet spot**; batch=32 for all 6 backbones |
| A100 80 GB / H100 80 GB | 80 GB | ~2.5 hr | overkill; only useful for cranking batch sizes |

- ~50 GB disk for backbone checkpoints + dataset
- Tested on RunPod with the `pytorch:2.4.0-py3.11-cuda12.4.1` image

## Quick start (one shot)

```bash
git clone https://github.com/egosurghands/EgoSurgHands.git
cd EgoSurgHands
export HF_TOKEN=your_hf_token  # or huggingface-cli login

bash setup.sh        # ~30 min: clones backbone repos, downloads ckpts, downloads dataset
bash reproduce.sh    # ~1 hour: runs ZS inference for all 6 backbones, trains JRN, computes PA-P2D

# Headline numbers land in:
ls outputs/headline_pa_mpjpe.json outputs/headline_pa_p2d.json
```

## Frozen reference numbers

If `outputs/headline_pa_mpjpe.json` matches `examples/headline.json` to ~0.1 mm, reproduction succeeded.

| Backbone | ZS PA-MPJPE [95% CI] | +JRN PA-MPJPE [95% CI] |
|---|---:|---:|
| **JointTransformer** ★ | 14.95 [14.39, 15.46] | **10.70** [10.21, 11.19] |
| **WildHands** ★ | 20.48 [19.72, 21.06] | **10.40** [9.91, 10.94] |
| HaMeR | 24.99 [23.77, 25.89] | 16.11 [14.16, 17.69] |
| MeshGraphormer | 26.22 [24.66, 27.77] | 15.12 [14.43, 15.69] |
| SimpleHand | 28.59 [27.67, 29.73] | 21.10 [20.01, 22.01] |
| WiLoR | 25.34 [24.10, 26.30] | 15.91 [14.13, 17.84] |

★ = ego-pretrained.

## Protocol

- **Train partition:** `aria_v2_train_mps` (41,909 hand-instance rows, 55 procedure recordings).
- **Val partition:** `aria_v2_val` (9,476 hand-instance rows, 13 procedure recordings; IAA-validated).
- **Train ∩ val PRs = ∅** by construction (asserted at runtime in `train_jrn.py`).
- **Ground truth:** Aria MPS (Machine Perception Services) sensor-fused landmarks, 21 joints.
- **Metric:** PA-MPJPE-MPS (Procrustes-aligned mean per-joint position error against MPS), Umeyama Procrustes (rotation + uniform scale + translation), OP_VALID 20-joint subset (drops OpenPose joint 1 = thumb_CMC, no Aria MPS analog), bootstrap 95% CI from 1,000 PR-keyed resamples.
- **JRN architecture:** Two-headed MLP, hidden=128, GELU, zero-init heads. **Pose head** (63 -> 128 -> 128 -> 63, 32,831 params) predicts a root-relative joint residual; **translation head** (66 -> 128 -> 128 -> 3, 25,475 params) predicts a wrist correction broadcast to all 21 joints. Total: **58,306 parameters**. The two heads are orthogonal: PA-MPJPE / MPJPE-RR are bit-identical (verified to 0.000000 mm) whether or not the translation head is applied, while root err and absolute MPJPE drop 70-86%.
- **JRN training:** AdamW lr=1e-3 wd=1e-4, batch=512, 60 epochs, seed=42, loss = MSE(pose) + MSE(trans). ~8 sec on A100.

## Per-step reproduction (if `reproduce.sh` fails)

### 1. Zero-shot inference (5 backbones via shared runner)
```bash
for bb in hamer wilor wildhands jointtransformer simplehand; do
  python eval/run_eval.py --dataset aria_v2_train_mps --models $bb
  python eval/run_eval.py --dataset aria_v2_val --models $bb
done
```

### 2. MeshGraphormer (separate runner due to non-standard model construction)
```bash
python eval/run_meshgraphormer.py --dataset aria_v2_train_mps
python eval/run_meshgraphormer.py --dataset aria_v2_val
```

### 3. JRN training + evaluation (per backbone)
```bash
for bb in hamer wilor wildhands jointtransformer simplehand meshgraphormer; do
  python jrn/train_jrn.py --backbone $bb --device cuda
done
```

### 4. PA-P2D (per backbone)
```bash
for bb in hamer wilor wildhands jointtransformer simplehand meshgraphormer; do
  python jrn/pa_p2d.py --backbone $bb --device cuda
done
```

### 5. Generate figures
```bash
python figures/make_figures.py
```

## WiLoR wrapper fix (one-time, applied during setup)

WiLoR's upstream `wilor/models/mano_wrapper.py:35` already permutes joints to OpenPose order; the eval-side wrapper used to apply that permutation a second time, scrambling joint identity. `setup.sh` applies the fix automatically. The active wrapper is `eval/models/wilor_model.py` in this repo.

Pre-fix vs post-fix bone[wrist→thumb_CMC]: 84.9 mm → 37.96 mm (anatomically corrected). Pre-fix vs post-fix WiLoR ZS: 36.98 → 25.34 mm.

## Citing

```bibtex
@inproceedings{pal2026egosurghands,
  title     = {{EgoSurgHands}: An Egocentric {3D} Hand Pose Dataset \& Benchmark for Open-Surgery Training},
  author    = {Pal, Ankit and Kugener, Guillaume and Volpi, Sofia and Schwartz, Joshua and Idarraga, Alex and Rajpurkar, Pranav and Brat, Gabriel A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

## Acknowledgements

Backbones: [HaMeR](https://github.com/geopavlakos/hamer) (Pavlakos et al. 2024), [WiLoR](https://github.com/rolpotamias/WiLoR) (Potamias et al. 2024), [WildHands](https://github.com/ap229997/hands) (Prakash et al. 2025), [JointTransformer](https://github.com/) (Patel et al. 2023), [SimpleHand](https://github.com/) (Zhou et al. 2024), [MeshGraphormer](https://github.com/microsoft/MeshGraphormer) (Lin et al. 2021).
Hardware: Project Aria (Meta).
