"""Standalone MeshGraphormer inference on aria_v2_train_mps.

Output: /workspace/eval_v1_out/aria_v2_train_mps/meshgraphormer_aria_v2_train_mps_per_sample.pkl
+ metrics.json

Adapted from /workspace/code/MeshGraphormer/src/tools/run_gphmer_handmesh_inference.py.
Uses the existing phase0/models/meshgraphormer_model.py wrapper.
"""
from __future__ import annotations
import os, sys, time, json, gc
from pathlib import Path

REPRO = "/workspace/repro"
PHASE0_REPO = f"{REPRO}/phase0-eval"
MGFM_DIR = "/workspace/code/MeshGraphormer"
MGFM_CKPT_DIR = "/workspace/checkpoints/meshgraphormer"

# HF_TOKEN
if not os.environ.get("HF_TOKEN"):
    for cand in ["/workspace/root_backup/env.sh", "/root/env.sh"]:
        if os.path.exists(cand):
            for line in open(cand):
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.strip().split("=", 1)
                    if "HF_TOKEN" in k:
                        os.environ.setdefault("HF_TOKEN", v.strip().strip('"').strip("'").replace("export ", ""))
            break
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing"

sys.path.insert(0, PHASE0_REPO)
sys.path.insert(0, MGFM_DIR)
# MANO uses relative paths; chdir into MGFM root so 'src/modeling/data/MANO_RIGHT.pkl' resolves
os.chdir(MGFM_DIR)

from phase0.pytorch3d_stub import install as _install_p3d_stub
_install_p3d_stub()

import torch
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="aria_v2_train_mps", choices=["aria_v2_train_mps", "aria_v2_val", "aria_v1_locked"])
ap.add_argument("--max", type=int, default=None)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

OUT_DIR = f"/workspace/eval_v1_out/{args.dataset}"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[setup] device={device}  dataset={args.dataset}")

# === Build MGFM model ===
print("[load] MGFM modules...")
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config

# MANO + Mesh
mano_model = MANO().to(device)
mano_model.layer = mano_model.layer.to(device)
mesh_sampler = Mesh()

# 3 transformer-encoder blocks
input_feat_dim = [2051, 512, 128]
hidden_feat_dim = [1024, 256, 64]
output_feat_dim = input_feat_dim[1:] + [3]
which_blk_graph = [0, 0, 1]

trans_encoder = []
for i in range(len(output_feat_dim)):
    config = BertConfig.from_pretrained(f"{MGFM_DIR}/src/modeling/bert/bert-base-uncased/")
    config.output_attentions = False
    config.img_feature_dim = input_feat_dim[i]
    config.output_feature_dim = output_feat_dim[i]
    config.hidden_size = hidden_feat_dim[i]
    config.intermediate_size = int(hidden_feat_dim[i] * 2)
    config.num_hidden_layers = 4
    config.num_attention_heads = 4
    if which_blk_graph[i] == 1:
        config.graph_conv = True
    else:
        config.graph_conv = False
    config.mesh_type = "hand"
    assert config.hidden_size % config.num_attention_heads == 0
    trans_encoder.append(Graphormer(config=config))

# HRNet-W64 backbone
hrnet_yaml = f"{MGFM_CKPT_DIR}/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
hrnet_checkpoint = f"{MGFM_CKPT_DIR}/hrnetv2_w64_imagenet_pretrained.pth"
hrnet_update_config(hrnet_config, hrnet_yaml)
backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)

trans_encoder = torch.nn.Sequential(*trans_encoder)
class _Args: pass
_args_dummy = _Args()
_model = Graphormer_Network(_args_dummy, config, backbone, trans_encoder)
state_dict = torch.load(f"{MGFM_CKPT_DIR}/graphormer_hand_state_dict.bin", map_location="cpu")
_model.load_state_dict(state_dict, strict=False)
del state_dict; gc.collect()

# enable hidden states / attentions on last block (matches inference script)
setattr(_model.trans_encoder[-1].config, 'output_attentions', True)
setattr(_model.trans_encoder[-1].config, 'output_hidden_states', True)
_model.trans_encoder[-1].bert.encoder.output_attentions = True
_model.trans_encoder[-1].bert.encoder.output_hidden_states = True
for iter_layer in range(4):
    _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True

_model = _model.to(device).eval()
print(f"[load] MGFM ready: {sum(p.numel() for p in _model.parameters())/1e6:.1f}M params")

# === Wrap with phase0 ===
from phase0.models.meshgraphormer_model import MeshGraphormerModel
wrapper = MeshGraphormerModel(_model, mano_model, mesh_sampler, device)

# === Run via phase0 eval_loop ===
from phase0 import DEFAULT_DATASETS
from phase0.eval_loop import EvalConfig, eval_stream
SPEC = DEFAULT_DATASETS[args.dataset]

cfg = EvalConfig(
    batch_size=args.batch,
    max_total=args.max,
    log_every=5,
    show_progress=True,
    hf_token=os.environ["HF_TOKEN"],
    out_dir=OUT_DIR,
    save_per_sample=True,
    viz_count=0,
    save_pred_vertices=False,
)
print(f"\n[run] MGFM on {args.dataset} (batch={args.batch})")
t0 = time.time()
r = eval_stream(wrapper, SPEC, args.dataset, device, cfg, mano_fitter=None)
dt = time.time() - t0
print(f"[done] n={r.n_samples}  elapsed={dt:.0f}s")
for k, v in r.metrics.items():
    if isinstance(v, dict) and v.get("mean") is not None:
        print(f"  {k:24s}: {v['mean']:7.3f}  n={v['n']}")
