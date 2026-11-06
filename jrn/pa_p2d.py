"""PA-P2D: Procrustes-align pred 3D to GT 3D, project through Aria pinhole approx, measure pixel L2 vs GT 2D.

For each backbone:
1. Load v2 train pkl (predictions on training partition).
2. Train JRN MLP on residuals.
3. Apply to v2 val (filtered v1 val_locked).
4. For both ZS and JRN-corrected:
   - Procrustes-align (Umeyama) pred 3D -> gt_v2_3d
   - Project aligned-pred and gt through Aria pinhole (f=606.5, c=(704,704))
   - Pixel L2 on OP_VALID joints
5. Bootstrap 95% CI over 13 PRs.

Output: /workspace/v2_jrn_out_fixed/{bb}_pa_p2d_summary.json
"""
from __future__ import annotations
import os, pickle, json, argparse, glob, time
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True,
                choices=["hamer","wilor","wildhands","jointtransformer","simplehand","meshgraphormer","handoccnet"])
ap.add_argument("--device", default="cuda", choices=["cpu","cuda"])
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--out_dir", default="/workspace/v2_jrn_out_fixed")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device)
print(f"[setup] backbone={args.backbone}  device={device}")

# Aria pinhole approx (matches MPS undistorted central focal)
FX = FY = 606.5
CX = CY = 704.0

OP_VALID = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int64)

def procrustes_align(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)
    mu_p = pred.mean(0); mu_g = gt.mean(0)
    p = pred - mu_p; g = gt - mu_g
    M = p.T @ g
    U, S, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy(); Vt[-1] *= -1
        S = S.copy(); S[-1] *= -1
        R = Vt.T @ U.T
    var_p = (p ** 2).sum()
    scale = S.sum() / var_p if var_p > 1e-12 else 1.0
    return (scale * (p @ R.T) + mu_g).astype(np.float32)

def project_pinhole(joints_3d):
    """3D points in camera frame (meters) -> 2D pixel coords. Z must be > 0."""
    z = np.maximum(joints_3d[:, 2], 1e-6)
    u = FX * joints_3d[:, 0] / z + CX
    v = FY * joints_3d[:, 1] / z + CY
    return np.stack([u, v], axis=-1)

def pa_p2d_px(pred_3d, gt_3d):
    """PA-P2D: align 3D, project both through Aria pinhole, mean pixel L2."""
    aligned = procrustes_align(pred_3d, gt_3d)
    proj_aligned = project_pinhole(aligned)
    proj_gt = project_pinhole(gt_3d)
    return float(np.linalg.norm(proj_aligned - proj_gt, axis=-1).mean())

def per_pr_bootstrap(values, prs, n_boot=1000, seed=42):
    pr_to_vals = defaultdict(list)
    for v, p in zip(values, prs):
        pr_to_vals[p].append(v)
    pr_means = {p: float(np.mean(vs)) for p, vs in pr_to_vals.items()}
    pr_ns    = {p: len(vs) for p, vs in pr_to_vals.items()}
    pr_list = list(pr_means.keys())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(pr_list, size=len(pr_list), replace=True)
        total_n = sum(pr_ns[p] for p in sampled)
        if total_n == 0: continue
        m = sum(pr_means[p] * pr_ns[p] for p in sampled) / total_n
        means.append(m)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

class ResMLP(nn.Module):
    def __init__(self, in_dim=63, hidden=128, depth=2):
        super().__init__()
        layers = []; d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 63)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x): return self.head(self.trunk(x))

def gt_field(r):
    for k in ('gt_v2_joints_3d', 'gt_joints_3d'):
        v = r.get(k)
        if v is None: continue
        try: g = np.asarray(v)
        except Exception: continue
        if g.shape == (21,3): return g
    return None

def filter_train(records):
    out = []
    for r in records:
        if 'pred_joints_3d' not in r: continue
        p = np.asarray(r['pred_joints_3d'])
        if p.shape != (21,3) or not np.isfinite(p).all(): continue
        g = gt_field(r)
        if g is None or not np.isfinite(g).all(): continue
        r['_gt'] = g
        out.append(r)
    return out

def filter_val(records, v2_keys):
    out = []
    seen = set()
    for r in records:
        fk = "{}/{:05d}_{}".format(r["sequence_name"], r["frame_id"], r["hand_side"])
        if fk not in v2_keys or fk in seen: continue
        seen.add(fk)
        if 'pred_joints_3d' not in r: continue
        p = np.asarray(r['pred_joints_3d'])
        if p.shape != (21,3) or not np.isfinite(p).all(): continue
        g_v2 = r.get('gt_v2_joints_3d')
        if g_v2 is None:
            g_v2 = r.get('gt_joints_3d')
        if g_v2 is None: continue
        g_v2 = np.asarray(g_v2)
        if g_v2.shape != (21,3) or not np.isfinite(g_v2).all(): continue
        r['_gt_v2'] = g_v2
        out.append(r)
    return out

def encode(rec):
    j = np.asarray(rec['pred_joints_3d'])
    return (j - j[0:1]).reshape(-1).astype(np.float32)

def residual(rec):
    p = np.asarray(rec['pred_joints_3d']); g = rec['_gt']
    return (g - g[0:1] - (p - p[0:1])).flatten().astype(np.float32)

# === LOAD V2 TRAIN ===
v2_train_path = f"/workspace/eval_v1_out/aria_v2_train_mps/{args.backbone}_aria_v2_train_mps_per_sample.pkl"
print(f"[load] {v2_train_path}")
with open(v2_train_path, "rb") as f:
    train_recs = pickle.load(f)
train_recs = filter_train(train_recs)
print(f"  valid train: {len(train_recs)}")

# === LOAD V2 VAL ===
v2_keys = set()
for f in sorted(glob.glob("/workspace/v2_clean_v2/validation/*.parquet")):
    df = pq.ParquetFile(f).read(columns=["frame_key"]).to_pandas()
    v2_keys.update(df["frame_key"].tolist())

# Prefer aria_v2_val (dedicated v2 inference) over filtered v1_locked.
v2_val_path = f"/workspace/eval_v1_out/aria_v2_val/{args.backbone}_aria_v2_val_per_sample.pkl"
v1_val_path = f"/workspace/eval_v1_out/aria_v1_locked/{args.backbone}_aria_v1_locked_per_sample.pkl"
val_pkl = v2_val_path if os.path.exists(v2_val_path) else v1_val_path
print(f"  using val pkl: {val_pkl}")
with open(val_pkl, "rb") as f:
    v1_val_recs = pickle.load(f)
val_recs = filter_val(v1_val_recs, v2_keys)
print(f"  valid v2 val: {len(val_recs)}")

# === TRAIN JRN ===
print(f"\n[train] JRN on {len(train_recs)} samples for {args.epochs} epochs")
torch.manual_seed(42); np.random.seed(42)
Xtr = np.stack([encode(r) for r in train_recs]).astype(np.float32)
Ytr = np.stack([residual(r) for r in train_recs]).astype(np.float32)
model = ResMLP(in_dim=63, hidden=args.hidden, depth=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
X = torch.from_numpy(Xtr).to(device); Y = torch.from_numpy(Ytr).to(device)
n = X.shape[0]; bs = 512
for ep in range(args.epochs):
    idx = torch.randperm(n, device=device)
    for i in range(0, n, bs):
        b = idx[i:i+bs]
        loss = (model(X[b]) - Y[b]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
print(f"  JRN trained in {time.time():.0f}s")

# === COMPUTE PA-P2D for ZS and JRN ===
model.eval()
Xv = np.stack([encode(r) for r in val_recs]).astype(np.float32)
with torch.no_grad():
    deltas = model(torch.from_numpy(Xv).to(device)).cpu().numpy().reshape(-1, 21, 3)

zs_pa_p2d = []
jrn_pa_p2d = []
zs_p2d_raw = []
jrn_p2d_raw = []  # without Procrustes - projects pred_3d directly through pinhole
for r, d in zip(val_recs, deltas):
    p = np.asarray(r['pred_joints_3d']); g = r['_gt_v2']
    corrected = p + d
    p_op, g_op, c_op = p[OP_VALID], g[OP_VALID], corrected[OP_VALID]
    # PA-P2D
    zs_pa_p2d.append(pa_p2d_px(p_op, g_op))
    jrn_pa_p2d.append(pa_p2d_px(c_op, g_op))
    # Raw P2D (no Procrustes, projecting through real Aria K with whatever root the model gave)
    proj_p = project_pinhole(p_op)
    proj_c = project_pinhole(c_op)
    proj_g = project_pinhole(g_op)
    zs_p2d_raw.append(float(np.linalg.norm(proj_p - proj_g, axis=-1).mean()))
    jrn_p2d_raw.append(float(np.linalg.norm(proj_c - proj_g, axis=-1).mean()))

prs = [r["sequence_name"] for r in val_recs]

def summarize(values, name):
    mean = float(np.mean(values))
    median = float(np.median(values))
    lo, hi = per_pr_bootstrap(values, prs)
    print(f"  {name:30s}: mean={mean:8.2f}  median={median:8.2f}  CI=[{lo:7.2f}, {hi:7.2f}]")
    return {"mean": mean, "median": median, "ci_low": lo, "ci_high": hi, "n": len(values)}

print(f"\n[results] PA-P2D and raw-P2D on v2 val (n={len(val_recs)}, OP_VALID 20-joint, Aria pinhole f=606.5)")
out = {
    "backbone": args.backbone,
    "n_train": len(train_recs),
    "n_val": len(val_recs),
    "projection": "Aria pinhole approx (fx=fy=606.5, cx=cy=704)",
    "zs":  {"pa_p2d_px": summarize(zs_pa_p2d,  "ZS PA-P2D px"),
            "p2d_raw_px": summarize(zs_p2d_raw, "ZS raw-P2D px (no PA)"),},
    "jrn_v2train": {"pa_p2d_px": summarize(jrn_pa_p2d,  "+JRN PA-P2D px"),
                    "p2d_raw_px": summarize(jrn_p2d_raw, "+JRN raw-P2D px (no PA)"),},
}
out_path = f"{args.out_dir}/{args.backbone}_pa_p2d_summary.json"
with open(out_path, "w") as f: json.dump(out, f, indent=2)
print(f"\n[saved] {out_path}")
