"""Train and evaluate JRN (Joint-Residual Network) on a frozen-backbone prediction stream.

JRN is a 58K-parameter two-headed MLP corrector stacked on a frozen monocular hand-pose
backbone. Both heads are zero-initialized; at init, JRN is a no-op. Training:

    - Pose head: 2-layer MLP (63 -> 128 -> 128 -> 63), 32,831 params.
        Input:  root-relative predicted joints, flattened (21*3 = 63 dim).
        Output: root-relative joint residual to add to the prediction (preserves PA-MPJPE invariance).
    - Translation head: 2-layer MLP (66 -> 128 -> 128 -> 3), 25,475 params.
        Input:  root-relative pose (63) + raw predicted wrist xyz (3) = 66 dim.
        Output: 3-dim wrist correction. Broadcast to all 21 joints (skeleton-rigid translation).

Total: 58,306 params. Pose head dominates PA-MPJPE/MPJPE-RR; translation head
dominates root err / MPJPE-abs. The heads are orthogonal: applying the translation
correction is bit-identical to JRN-pose under any translation-invariant metric.

For each backbone:
1. Load v2 train pkl (frozen-backbone predictions on the training partition).
2. Train JRN for 60 epochs with seed=42 on (pose_residual, translation_residual) targets.
3. Apply to v2 val (PR-disjoint by construction; runtime-asserted).
4. Report 5 metrics x 2 conditions (ZS, +JRN): PA-MPJPE, MPJPE-RR, root err, MPJPE-abs.
5. Bootstrap 95% CI over 13 PRs.
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
torch.manual_seed(42); np.random.seed(42)
print(f"[setup] backbone={args.backbone}  device={device}")

# OP_VALID: 20 OP joint indices that have an MPS analog (drops OP[1] thumb_CMC).
OP_VALID = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int64)

# === Eval framework's Procrustes (rotation + uniform scale + translation, Umeyama) ===
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

def pa_mpjpe_mm(pred, gt):
    return float(np.linalg.norm(procrustes_align(pred, gt) - gt, axis=-1).mean() * 1000)

def mpjpe_root_relative_mm(pred, gt):
    return float(np.linalg.norm((pred - pred[0:1]) - (gt - gt[0:1]), axis=-1).mean() * 1000)

def root_err_mm(pred, gt):
    return float(np.linalg.norm(pred[0] - gt[0]) * 1000)

def mpjpe_abs_mm(pred, gt):
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000)

def per_pr_bootstrap(values, prs, n_boot=1000, seed=42):
    pr_to_vals = defaultdict(list)
    for v, p in zip(values, prs): pr_to_vals[p].append(v)
    pr_means = {p: float(np.mean(vs)) for p, vs in pr_to_vals.items()}
    pr_ns    = {p: len(vs) for p, vs in pr_to_vals.items()}
    pr_list  = list(pr_means.keys())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        s = rng.choice(pr_list, size=len(pr_list), replace=True)
        tn = sum(pr_ns[p] for p in s)
        if tn == 0: continue
        means.append(sum(pr_means[p] * pr_ns[p] for p in s) / tn)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# === Architecture: JRN = 2-headed MLP (pose + translation), 58,306 params total ===
class ResMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=2):
        super().__init__()
        layers = []; d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x): return self.head(self.trunk(x))

class JRN(nn.Module):
    """Two-headed residual corrector. Both heads zero-initialized -> no-op at init."""
    def __init__(self, hidden=128):
        super().__init__()
        self.pose_head  = ResMLP(in_dim=63, hidden=hidden, out_dim=63)  # 32,831 params
        self.trans_head = ResMLP(in_dim=66, hidden=hidden, out_dim=3)   # 25,475 params
    def forward(self, x_rel, x_root):
        delta_pose  = self.pose_head(x_rel)                              # (B, 63)
        delta_trans = self.trans_head(torch.cat([x_rel, x_root], -1))    # (B, 3)
        return delta_pose, delta_trans

def gt_field(r):
    """Prefer Aria MPS GT (gt_v2_joints_3d); fall back to gt_joints_3d when v2 missing."""
    for k in ('gt_v2_joints_3d', 'gt_joints_3d'):
        v = r.get(k)
        if v is None: continue
        try: g = np.asarray(v)
        except Exception: continue
        if g.shape == (21,3): return g
    return None

def filter_valid(records):
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

def encode_rel(rec):
    j = np.asarray(rec['pred_joints_3d'])
    return (j - j[0:1]).reshape(-1).astype(np.float32)

def encode_root(rec):
    return np.asarray(rec['pred_joints_3d'])[0].astype(np.float32)

def residual_pose(rec):
    p = np.asarray(rec['pred_joints_3d']); g = rec['_gt']
    return ((g - g[0:1]) - (p - p[0:1])).flatten().astype(np.float32)

def residual_trans(rec):
    p = np.asarray(rec['pred_joints_3d']); g = rec['_gt']
    return (g[0] - p[0]).astype(np.float32)

# === Load v2 train pred ===
v2_train_path = f"/workspace/eval_v1_out/aria_v2_train_mps/{args.backbone}_aria_v2_train_mps_per_sample.pkl"
print(f"[load] {v2_train_path}")
with open(v2_train_path, "rb") as f:
    train_recs = pickle.load(f)
print(f"  raw: {len(train_recs)}")
train_recs = filter_valid(train_recs)
print(f"  valid: {len(train_recs)}")

# === Load v2 val: prefer dedicated v2_val pkl; fall back to v1_locked filtered to v2 frame_keys ===
v2_keys = set()
for f in sorted(glob.glob("/workspace/v2_clean_v2/validation/*.parquet")):
    df = pq.ParquetFile(f).read(columns=["frame_key"]).to_pandas()
    v2_keys.update(df["frame_key"].tolist())
print(f"[load] v2 val frame_keys: {len(v2_keys)}")

v2_val_path = f"/workspace/eval_v1_out/aria_v2_val/{args.backbone}_aria_v2_val_per_sample.pkl"
v1_val_path = f"/workspace/eval_v1_out/aria_v1_locked/{args.backbone}_aria_v1_locked_per_sample.pkl"
val_pkl = v2_val_path if os.path.exists(v2_val_path) else v1_val_path
print(f"  using val pkl: {val_pkl}")
with open(val_pkl, "rb") as f:
    raw_val = pickle.load(f)
print(f"  loaded: {len(raw_val)} val records")

seen = set(); val_recs = []
for r in raw_val:
    fk = "{}/{:05d}_{}".format(r["sequence_name"], r["frame_id"], r["hand_side"])
    if fk in v2_keys and fk not in seen:
        seen.add(fk); val_recs.append(r)
val_recs = filter_valid(val_recs)
print(f"  v2 val (filtered, deduped, valid): {len(val_recs)}")

# === Sanity: PR-disjoint ===
tprs = {r["sequence_name"] for r in train_recs}
vprs = {r["sequence_name"] for r in val_recs}
overlap = tprs & vprs
assert len(overlap) == 0, f"PR overlap detected: {overlap}"
print(f"[sanity] train PRs={len(tprs)}  val PRs={len(vprs)}  overlap=0")

# === Train JRN on v2 train ===
print(f"\n[train] JRN on v2 train ({len(train_recs)} samples) for {args.epochs} epochs")
Xrel_tr  = np.stack([encode_rel(r)      for r in train_recs]).astype(np.float32)
Xroot_tr = np.stack([encode_root(r)     for r in train_recs]).astype(np.float32)
Ypose_tr = np.stack([residual_pose(r)   for r in train_recs]).astype(np.float32)
Ytrans_tr= np.stack([residual_trans(r)  for r in train_recs]).astype(np.float32)
print(f"  Xrel={Xrel_tr.shape}  Xroot={Xroot_tr.shape}  Ypose={Ypose_tr.shape}  Ytrans={Ytrans_tr.shape}")

model = JRN(hidden=args.hidden).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[arch] JRN  {n_params} params")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
Xrel  = torch.from_numpy(Xrel_tr).to(device)
Xroot = torch.from_numpy(Xroot_tr).to(device)
Ypose = torch.from_numpy(Ypose_tr).to(device)
Ytrans= torch.from_numpy(Ytrans_tr).to(device)
n = Xrel.shape[0]; bs = 512; t0 = time.time()
for ep in range(args.epochs):
    idx = torch.randperm(n, device=device)
    ep_lp = ep_lt = 0.0; nb = 0
    for i in range(0, n, bs):
        b = idx[i:i+bs]
        dp, dt = model(Xrel[b], Xroot[b])
        loss_pose  = (dp - Ypose[b]).pow(2).mean()
        loss_trans = (dt - Ytrans[b]).pow(2).mean()
        loss = loss_pose + loss_trans
        opt.zero_grad(); loss.backward(); opt.step()
        ep_lp += loss_pose.item(); ep_lt += loss_trans.item(); nb += 1
    if (ep+1) % 20 == 0:
        print(f"  epoch {ep+1}/{args.epochs}  loss_pose={ep_lp/nb:.6f}  loss_trans={ep_lt/nb:.6f}  elapsed={time.time()-t0:.0f}s")

# === Eval: ZS + JRN on v2 val ===
print(f"\n[eval] full metrics on v2 val ({len(val_recs)} samples)")
model.eval()
Xrel_v  = np.stack([encode_rel(r)  for r in val_recs]).astype(np.float32)
Xroot_v = np.stack([encode_root(r) for r in val_recs]).astype(np.float32)
with torch.no_grad():
    dp, dt = model(torch.from_numpy(Xrel_v).to(device),
                   torch.from_numpy(Xroot_v).to(device))
    deltas_pose  = dp.cpu().numpy().reshape(-1, 21, 3)
    deltas_trans = dt.cpu().numpy()  # (N, 3)

zs_pa, jrn_pa     = [], []
zs_mp, jrn_mp     = [], []
zs_root, jrn_root = [], []
zs_abs, jrn_abs   = [], []

for r, dpi, dti in zip(val_recs, deltas_pose, deltas_trans):
    p = np.asarray(r['pred_joints_3d']); g = r['_gt']
    # JRN application: pose residual + translation broadcast to all 21 joints (skeleton-rigid).
    # The translation broadcast preserves PA-MPJPE invariance (Procrustes absorbs translation).
    corrected = p + dpi + dti[None, :]
    p_op, g_op, c_op = p[OP_VALID], g[OP_VALID], corrected[OP_VALID]
    p_op_rel = p_op - p_op[0]; g_op_rel = g_op - g_op[0]; c_op_rel = c_op - c_op[0]
    zs_pa.append(pa_mpjpe_mm(p_op_rel, g_op_rel))
    jrn_pa.append(pa_mpjpe_mm(c_op_rel, g_op_rel))
    zs_mp.append(mpjpe_root_relative_mm(p_op, g_op))
    jrn_mp.append(mpjpe_root_relative_mm(c_op, g_op))
    zs_root.append(root_err_mm(p_op, g_op))
    jrn_root.append(root_err_mm(c_op, g_op))
    zs_abs.append(mpjpe_abs_mm(p_op, g_op))
    jrn_abs.append(mpjpe_abs_mm(c_op, g_op))

prs = [r["sequence_name"] for r in val_recs]

def summarize(values, name):
    mean = float(np.mean(values)); median = float(np.median(values))
    lo, hi = per_pr_bootstrap(values, prs)
    print(f"  {name:30s}: mean={mean:7.3f}  median={median:7.3f}  CI=[{lo:6.2f}, {hi:6.2f}]")
    return {"mean": mean, "median": median, "ci_low": lo, "ci_high": hi, "n": len(values)}

print(f"\n[ZS]")
zs_summary = {
    "pa_mpjpe":  summarize(zs_pa,   "PA-MPJPE-MPS"),
    "mpjpe_rr":  summarize(zs_mp,   "MPJPE root-relative"),
    "root_err":  summarize(zs_root, "root err"),
    "mpjpe_abs": summarize(zs_abs,  "MPJPE abs"),
}
print(f"\n[+JRN]")
jrn_summary = {
    "pa_mpjpe":  summarize(jrn_pa,   "PA-MPJPE-MPS"),
    "mpjpe_rr":  summarize(jrn_mp,   "MPJPE root-relative"),
    "root_err":  summarize(jrn_root, "root err"),
    "mpjpe_abs": summarize(jrn_abs,  "MPJPE abs"),
}

out = {
    "backbone": args.backbone,
    "n_train": len(train_recs),
    "n_val": len(val_recs),
    "n_params": n_params,
    "epochs": args.epochs,
    "hidden": args.hidden,
    "architecture": "JRN: 2-headed MLP (pose 32831 + translation 25475 = 58306 params), zero-init heads",
    "procrustes": "rotation+uniform_scale+translation (Umeyama)",
    "zs": zs_summary,
    "jrn": jrn_summary,
}
out_path = f"{args.out_dir}/{args.backbone}_v2train_jrn_summary_fixed.json"
with open(out_path, "w") as f: json.dump(out, f, indent=2)
print(f"\n[saved] {out_path}")
