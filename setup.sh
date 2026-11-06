#!/bin/bash
# setup.sh - one-shot environment + dataset + checkpoint download
# Tested on RunPod with pytorch:2.4.0-py3.11-cuda12.4.1
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-/workspace}"
mkdir -p "$WORK"
cd "$WORK"

echo "=========================================================="
echo "[1/5] Python deps"
echo "=========================================================="
pip install -r "$REPO_ROOT/requirements.txt"

echo "=========================================================="
echo "[2/5] HF dataset (aria-surgical-hand-pose-v2)"
echo "=========================================================="
: "${HF_TOKEN:?Set HF_TOKEN env var or run: huggingface-cli login}"
mkdir -p "$WORK/v2_clean_v2"
hf download harvardsil/aria-surgical-hand-pose-v2 --repo-type dataset --local-dir "$WORK/v2_clean_v2"

echo "=========================================================="
echo "[3/5] Backbone repos (clone + branch)"
echo "=========================================================="
mkdir -p "$WORK/repos" && cd "$WORK/repos"

# HaMeR
[ -d hamer ] || git clone https://github.com/geopavlakos/hamer.git
# WiLoR
[ -d WiLoR ] || git clone https://github.com/rolpotamias/WiLoR.git
# WildHands (demo branch - hand reconstruction)
[ -d WildHands ] || git clone -b demo https://github.com/ap229997/hands.git WildHands
# JointTransformer (we use TransformerSF, ARCTIC p2 protocol)
[ -d JointTransformer ] || git clone https://github.com/zc-alexfan/jointtransformer.git JointTransformer
# SimpleHand
[ -d simpleHand ] || git clone https://github.com/patienceFromZhou/simpleHand.git
# MeshGraphormer
[ -d MeshGraphormer ] || git clone https://github.com/microsoft/MeshGraphormer.git

echo "=========================================================="
echo "[4/5] Apply WiLoR wrapper fix (mano_to_op double-remap removed)"
echo "=========================================================="
# Our patched wrapper lives in eval/models/wilor_model.py - copy onto the
# checkout so the framework imports the post-fix version.
mkdir -p "$REPO_ROOT/.cache_patches"
cp "$REPO_ROOT/eval/models/wilor_model.py" "$REPO_ROOT/.cache_patches/wilor_model_post_fix.py"
echo "WiLoR post-fix wrapper available at eval/models/wilor_model.py - eval/run_eval.py imports from this repo."

echo "=========================================================="
echo "[5/5] Backbone checkpoints"
echo "=========================================================="
mkdir -p "$WORK/weights"

# HaMeR (~2.7GB)
mkdir -p "$WORK/weights/hamer_ckpts/checkpoints"
[ -f "$WORK/weights/hamer_ckpts/checkpoints/hamer.ckpt" ] || \
  hf download geopavlakos/hamer hamer.ckpt --local-dir "$WORK/weights/hamer_ckpts/checkpoints"

# WiLoR (~2.5GB)
mkdir -p "$WORK/weights/wilor"
[ -f "$WORK/weights/wilor/wilor_final.ckpt" ] || \
  hf download rolpotamias/WiLoR wilor_final.ckpt --local-dir "$WORK/weights/wilor"

# WildHands (~855MB)
mkdir -p "$WORK/weights/wildhands"
[ -f "$WORK/weights/wildhands/wildhands.ckpt" ] || \
  hf download ap229997/wildhands wildhands.ckpt --local-dir "$WORK/weights/wildhands"

# JointTransformer (~5.2GB)
mkdir -p "$WORK/weights/jointtransformer"
[ -f "$WORK/weights/jointtransformer/final_p2_462fc683e.ckpt" ] || \
  hf download zcafan/jointtransformer-arctic final_p2_462fc683e.ckpt --local-dir "$WORK/weights/jointtransformer"

# SimpleHand (~600MB)
mkdir -p "$WORK/weights/simplehand"
[ -f "$WORK/weights/simplehand/simpleHand_epoch_200.pt" ] || \
  hf download patienceFromZhou/simpleHand simpleHand_epoch_200.pt --local-dir "$WORK/weights/simplehand"

# MeshGraphormer (~855MB + 513MB HRNet init)
mkdir -p "$WORK/checkpoints/meshgraphormer"
[ -f "$WORK/checkpoints/meshgraphormer/graphormer_hand_state_dict.bin" ] || \
  hf download microsoft/MeshGraphormer graphormer_hand_state_dict.bin --local-dir "$WORK/checkpoints/meshgraphormer"
[ -f "$WORK/checkpoints/meshgraphormer/hrnetv2_w64_imagenet_pretrained.pth" ] || \
  hf download microsoft/MeshGraphormer hrnetv2_w64_imagenet_pretrained.pth --local-dir "$WORK/checkpoints/meshgraphormer"
[ -f "$WORK/checkpoints/meshgraphormer/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml" ] || \
  cp "$WORK/repos/MeshGraphormer/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml" \
     "$WORK/checkpoints/meshgraphormer/" 2>/dev/null || \
  echo "  HRNet config will be loaded from MeshGraphormer/models/hrnet/ at runtime"

# MANO models (HaMeR ships a copy; WildHands/MGFM use it too)
mkdir -p "$WORK/weights/mano"
[ -f "$WORK/weights/mano/MANO_RIGHT.pkl" ] && echo "  MANO already present" || \
  echo "  PLACE MANO_RIGHT.pkl + MANO_LEFT.pkl into $WORK/weights/mano (download from https://mano.is.tue.mpg.de/ - requires registration)"

echo ""
echo "=========================================================="
echo "Setup complete."
echo "Dataset:  $WORK/v2_clean_v2/{train,validation}/*.parquet"
echo "Repos:    $WORK/repos/{hamer,WiLoR,WildHands,JointTransformer,simpleHand,MeshGraphormer}"
echo "Weights:  $WORK/weights/  +  $WORK/checkpoints/meshgraphormer/"
echo ""
echo "Next: bash reproduce.sh"
echo "=========================================================="
