#!/bin/bash
# reproduce.sh - runs ZS inference for all 6 backbones, trains JRN, computes PA-P2D, generates figures.
# Assumes setup.sh has been run.
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-/workspace}"
OUT="${OUT:-$REPO_ROOT/outputs}"
mkdir -p "$OUT"

# Make eval/ importable as a package
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=========================================================="
echo "[1/4] Zero-shot inference: 5 backbones × 2 partitions"
echo "=========================================================="
for bb in hamer wilor wildhands jointtransformer simplehand; do
  for split in aria_v2_train_mps aria_v2_val; do
    echo "  >>> $bb on $split"
    python "$REPO_ROOT/eval/run_eval.py" \
      --dataset "$split" --models "$bb" --batch 32 \
      || { echo "FAILED: $bb $split"; exit 1; }
  done
done

echo "=========================================================="
echo "[2/4] Zero-shot inference: MeshGraphormer (standalone runner)"
echo "=========================================================="
python "$REPO_ROOT/eval/run_meshgraphormer.py" --dataset aria_v2_train_mps --batch 32
python "$REPO_ROOT/eval/run_meshgraphormer.py" --dataset aria_v2_val --batch 32

echo "=========================================================="
echo "[3/4] JRN training + PA-P2D for all 6 backbones"
echo "=========================================================="
for bb in hamer wilor wildhands jointtransformer simplehand meshgraphormer; do
  echo "  >>> JRN $bb"
  python "$REPO_ROOT/jrn/train_jrn.py" --backbone "$bb" --device cuda --out_dir "$OUT/jrn"
  echo "  >>> PA-P2D $bb"
  python "$REPO_ROOT/jrn/pa_p2d.py" --backbone "$bb" --device cuda --out_dir "$OUT/pa_p2d"
done

echo "=========================================================="
echo "[4/4] Figures + headline summary"
echo "=========================================================="
python "$REPO_ROOT/figures/make_figures.py" \
  --summaries-dir "$OUT/jrn" \
  --pa-p2d-dir "$OUT/pa_p2d" \
  --out-dir "$OUT/figures"

# Build headline JSON
python - <<PYEOF
import json, glob, os
out_dir = os.environ.get("OUT", "$OUT")
hdl = {}
for f in sorted(glob.glob(out_dir + "/jrn/*_v2train_jrn_summary_fixed.json")):
    bb = os.path.basename(f).replace("_v2train_jrn_summary_fixed.json", "")
    with open(f) as fh: d = json.load(fh)
    hdl[bb] = {
        "n_train": d["n_train"], "n_val": d["n_val"], "n_params": d.get("n_params"),
        "zs_pa_mpjpe":   d["zs"]["pa_mpjpe"]["mean"],
        "jrn_pa_mpjpe":  d["jrn"]["pa_mpjpe"]["mean"],
        "zs_ci":         [d["zs"]["pa_mpjpe"]["ci_low"],  d["zs"]["pa_mpjpe"]["ci_high"]],
        "jrn_ci":        [d["jrn"]["pa_mpjpe"]["ci_low"], d["jrn"]["pa_mpjpe"]["ci_high"]],
        "zs_mpjpe_abs":  d["zs"]["mpjpe_abs"]["mean"],
        "jrn_mpjpe_abs": d["jrn"]["mpjpe_abs"]["mean"],
    }
with open(out_dir + "/headline_pa_mpjpe.json", "w") as f: json.dump(hdl, f, indent=2)
print("Wrote", out_dir + "/headline_pa_mpjpe.json")
print()
print("HEADLINE - PA-MPJPE-MPS (mm) and MPJPE-abs (mm):")
print(f"{'Backbone':<20s} {'PA-ZS':>8s} {'PA-JRN':>8s}    {'abs-ZS':>8s} {'abs-JRN':>8s}")
for bb, v in sorted(hdl.items(), key=lambda x: x[1]["jrn_pa_mpjpe"]):
    print(f"{bb:<20s} {v['zs_pa_mpjpe']:>8.2f} {v['jrn_pa_mpjpe']:>8.2f}    {v['zs_mpjpe_abs']:>8.1f} {v['jrn_mpjpe_abs']:>8.1f}")
PYEOF

echo ""
echo "=========================================================="
echo "Reproduction complete. Headline at: $OUT/headline_pa_mpjpe.json"
echo "Compare to: $REPO_ROOT/examples/headline.json (frozen reference)"
echo "=========================================================="
