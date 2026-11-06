<div align="center">

# EgoSurgHands

### An Egocentric 3D Hand Pose Dataset & Benchmark for Open-Surgery Training

**NeurIPS 2026 · Evaluations & Datasets Track**

Ankit Pal<sup>1</sup> · Guillaume Kugener<sup>2</sup> · Sofia Volpi<sup>2</sup> · Joshua Schwartz<sup>2</sup> · Alex Idarraga<sup>2</sup> · Pranav Rajpurkar<sup>1,†</sup> · Gabriel A. Brat<sup>1,2,†</sup>

<sup>1</sup>Department of Biomedical Informatics, Harvard Medical School · <sup>2</sup>Department of Surgery, Beth Israel Deaconess Medical Center · <sup>†</sup>Equal senior authorship

[![Project page](https://img.shields.io/badge/Project-Page-e91e63)](https://egosurghands.github.io)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-aaditya%2FEgoSurgHands-ffcc4d)](https://huggingface.co/datasets/aaditya/EgoSurgHands)
[![Data license](https://img.shields.io/badge/Data-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

</div>

<p align="center">
  <img src="https://egosurghands.github.io/assets/qualitative.webp" width="88%" alt="WildHands zero-shot, WildHands with JRN, and the MPS label on three EgoSurgHands frames">
  <br>
  <em>WildHands zero-shot (left), WildHands + JRN (middle) and the Aria MPS label (right) on frames from three wearers and glove colors.</em>
</p>

## Overview

EgoSurgHands is, to our knowledge, the first benchmark for real-captured egocentric 3D hand pose estimation in open-surgery training. Surgical trainees wore Meta Project Aria glasses while suturing, tying knots, loading needles and cutting on synthetic-skin training pads.

- **41,909** training and **9,476** human-validated evaluation hand instances, from 55 and 13 procedure recordings.
- **4 trainees** (first-year medical student to third-year resident), **13 tasks** in 8 task families, **3 glove colors**.
- **Labels:** sensor-fused, human-validated pseudo-ground truth from Aria Machine Perception Services (MPS). Every evaluation frame passes expert review with examiner cross-review and senior-surgeon adjudication, and an external multi-view triangulated reference checks the labels.
- **Protocols:** a procedure-disjoint split plus a leave-one-wearer-out protocol that tests unseen trainees.

This repository reproduces the paper's zero-shot benchmark for the six MANO-based backbones and the JRN diagnostic baseline, a 58K-parameter residual corrector on frozen backbones.

## Highlights

- **Egocentric pretraining transfers well.** The ego-pretrained JointTransformer is best zero-shot at **14.95 mm** PA-MPJPE, 40% below the non-egocentric HaMeR (24.99 mm); WildHands is 18% below it.
- **A clear surgical-domain gap remains.** Zero-shot aligned errors span 15 to 30 mm, and the widely deployed MediaPipe Hands detects a hand in only **24.1%** of evaluation rows.
- **JRN** brings WildHands to 10.40 mm and JointTransformer to 10.70 mm on the procedure-disjoint split, and to 10.96 to 14.85 mm on held-out wearers. These corrected values reach the benchmark's label resolution scale of about 1 cm, so methods are compared on the zero-shot table.

## Leaderboard

Zero-shot PA-MPJPE in mm (lower is better) on the evaluation split (9,476 hand instances, 13 procedure recordings), on the 20 `OP_VALID` joints against the Aria MPS labels. Brackets give 95% CIs from 1,000 bootstrap resamples over procedure recordings.

| Rank | Method | Ego-pretrained | Zero-shot PA-MPJPE | + JRN |
|:---:|---|:---:|---:|---:|
| 1 | [JointTransformer](https://github.com/karimknaebel/JointTransformer) | ✓ | **14.95** [14.39, 15.46] | 10.70 |
| 2 | [IntagHand](https://github.com/Dw1010/IntagHand) |  | 20.28 | 13.94 |
| 3 | [WildHands](https://github.com/ap229997/hands) | ✓ | 20.48 [19.72, 21.06] | 10.40 |
| 4 | [HaMeR](https://github.com/geopavlakos/hamer) |  | 24.99 [23.77, 25.89] | 16.11 |
| 5 | [WiLoR](https://github.com/rolpotamias/WiLoR) |  | 25.34 [24.10, 26.30] | 15.91 |
| 6 | [MeshGraphormer](https://github.com/microsoft/MeshGraphormer) |  | 26.22 [24.66, 27.77] | 15.12 |
| 7 | [SimpleHand](https://github.com/patienceFromZhou/simpleHand) |  | 28.59 [27.67, 29.73] | 21.10 |
| - | [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe)\* |  | 30.45 | 18.05 |

\* MediaPipe Hands detects a hand in 2,283 of 9,476 evaluation rows (24.1%), so its errors cover those rows only and it is not ranked.
+ JRN values reach the label resolution scale and are not ranked. All five metrics (PA-MPJPE, MPJPE-RR, root error, MPJPE-abs, PA-P2D) are in Table 2 of the paper. To add a method, open an issue with your zero-shot results on the evaluation split.

## What's in this repository

```
egoSurgHands/
├── README.md
├── setup.sh               # environment, backbone repos and checkpoints, dataset download
├── requirements.txt       # Python dependencies (torch, datasets, pyarrow, matplotlib)
├── reproduce.sh           # one shot: zero-shot inference for 6 backbones, JRN, PA-P2D, figures
│
├── eval/                  # zero-shot inference
│   ├── run_eval.py        # HaMeR, WiLoR, WildHands, JointTransformer, SimpleHand
│   ├── run_meshgraphormer.py  # MeshGraphormer (separate model construction)
│   ├── eval_loop.py       # shared inference loop
│   ├── geometry.py        # Umeyama Procrustes alignment
│   ├── config.py          # dataset specifications
│   └── models/            # one wrapper per backbone
│
├── jrn/                   # Joint-Residual Network
│   ├── train_jrn.py       # JRN training and evaluation per backbone
│   └── pa_p2d.py          # PA-P2D (Procrustes-aligned 2D error)
│
├── figures/
│   └── make_figures.py    # paper figures from saved summaries
│
└── examples/
    └── headline.json      # frozen reference numbers
```

## Getting started

### Hardware

Peak memory is about 16 GB, during JointTransformer (ViT-G) inference; everything else stays under 12 GB. JRN training takes seconds on any GPU.

| GPU | VRAM | Full reproduction | Notes |
|---|---:|---:|---|
| RTX 4090 / RTX 3090 / A10G | 24 GB | about 4 to 5 h | best cost per result; JointTransformer, HaMeR and WiLoR may need batch 16 |
| A100 40 GB / A40 48 GB | 40 GB | about 3 h | recommended; batch 32 for all six backbones |
| A100 80 GB / H100 80 GB | 80 GB | about 2.5 h | more than needed |

About 50 GB of disk holds the backbone checkpoints and the dataset. Tested with the `pytorch:2.4.0-py3.11-cuda12.4.1` image.

### Data access

The dataset is gated. Request access at [huggingface.co/datasets/aaditya/EgoSurgHands](https://huggingface.co/datasets/aaditya/EgoSurgHands), then log in with `huggingface-cli login` or set `HF_TOKEN`. The MANO model files (`MANO_RIGHT.pkl`, `MANO_LEFT.pkl`) come from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) after registration.

### Quick start

```bash
git clone https://github.com/monk1337/egoSurgHands.git
cd egoSurgHands
export HF_TOKEN=your_hf_token   # or: huggingface-cli login

bash setup.sh       # about 30 min: backbone repos, checkpoints, dataset
bash reproduce.sh   # zero-shot inference for 6 backbones, JRN training, PA-P2D

ls outputs/headline_pa_mpjpe.json outputs/headline_pa_p2d.json
```

Reproduction succeeded if `outputs/headline_pa_mpjpe.json` matches `examples/headline.json` to about 0.1 mm.

### Step by step

```bash
# 1. Zero-shot inference (5 backbones through the shared runner)
for bb in hamer wilor wildhands jointtransformer simplehand; do
  python eval/run_eval.py --dataset aria_v2_train_mps --models $bb
  python eval/run_eval.py --dataset aria_v2_val --models $bb
done

# 2. MeshGraphormer (separate runner)
python eval/run_meshgraphormer.py --dataset aria_v2_train_mps
python eval/run_meshgraphormer.py --dataset aria_v2_val

# 3. JRN training and evaluation, then PA-P2D
for bb in hamer wilor wildhands jointtransformer simplehand meshgraphormer; do
  python jrn/train_jrn.py --backbone $bb --device cuda
  python jrn/pa_p2d.py --backbone $bb --device cuda
done

# 4. Figures
python figures/make_figures.py
```

## Benchmark protocol

- **Splits:** 41,909 training and 9,476 evaluation hand instances from 55 and 13 procedure recordings. The recordings are disjoint, and `train_jrn.py` asserts this at runtime.
- **Labels:** Aria MPS sensor-fused hand landmarks, 21 joints in the RGB camera frame.
- **Joints:** metrics use the 20 `OP_VALID` joints. Slot 1 holds the MPS palm center, because MPS has no thumb CMC.
- **Metric:** PA-MPJPE after Umeyama Procrustes alignment (rotation, uniform scale and translation). 95% CIs come from 1,000 bootstrap resamples over procedure recordings.
- **JRN:** a two-headed MLP (hidden size 128, GELU, zero-initialized heads) with 58,306 parameters.
  - The **pose head** (63 → 128 → 128 → 63, 32,831 parameters) predicts a root-relative joint residual.
  - The **translation head** (66 → 128 → 128 → 3, 25,475 parameters) predicts a wrist correction applied to all 21 joints.
  - The translation head leaves PA-MPJPE and MPJPE-RR unchanged (verified to 0.0000 mm), and full JRN cuts MPJPE-abs by 69% to 86% across the six backbones.
- **JRN training:** AdamW (lr 1e-3, weight decay 1e-4), batch 512, 60 epochs, seed 42, loss MSE(pose) + MSE(translation). About 8 s on an A100.

## WiLoR wrapper fix

WiLoR's `wilor/models/mano_wrapper.py` already permutes joints to OpenPose order, and the evaluation wrapper used to apply the permutation a second time. `setup.sh` applies the fix, and the active wrapper is `eval/models/wilor_model.py`. With the fix, WiLoR's zero-shot PA-MPJPE is 25.34 mm (36.98 mm before).

## Coming next

- IntagHand and MediaPipe Hands evaluation scripts
- Leave-one-wearer-out protocol scripts
- JRN checkpoints
- The multi-rater annotation tool
- Loading the public Hugging Face release directly in `setup.sh` and the evaluation loader

## Citation

```bibtex
@inproceedings{pal2026egosurghands,
  title     = {{EgoSurgHands}: An Egocentric {3D} Hand Pose Dataset \& Benchmark for Open-Surgery Training},
  author    = {Pal, Ankit and Kugener, Guillaume and Volpi, Sofia and Schwartz, Joshua and Idarraga, Alex and Rajpurkar, Pranav and Brat, Gabriel A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

## Acknowledgments

We thank the surgical trainees who wore the Aria glasses, and the annotators and surgeons who reviewed, corrected and adjudicated the evaluation frames. Recordings use Project Aria glasses and Machine Perception Services from Meta Reality Labs Research. GPU compute was provided by Nebius.

The benchmark builds on the public releases of [HaMeR](https://github.com/geopavlakos/hamer) (Pavlakos et al., CVPR 2024), [WiLoR](https://github.com/rolpotamias/WiLoR) (Potamias et al., CVPR 2025), [WildHands](https://github.com/ap229997/hands) (Prakash et al., ECCV 2024), [JointTransformer](https://github.com/karimknaebel/JointTransformer) (Knaebel, HANDS 2023 ARCTIC challenge winner), [MeshGraphormer](https://github.com/microsoft/MeshGraphormer) (Lin et al., ICCV 2021), [SimpleHand](https://github.com/patienceFromZhou/simpleHand) (Zhou et al., CVPR 2024), [IntagHand](https://github.com/Dw1010/IntagHand) (Li et al., CVPR 2022) and [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe) (Zhang et al., 2020).
