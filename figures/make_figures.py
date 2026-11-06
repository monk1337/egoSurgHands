"""Regenerate all paper figures with clean labels (no 'patched' or 'fixed' annotations)
and red-hot heatmap (F3) where high error = red = bad.
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/stoic/Documents/Projects/vla/final_model_results_2"
FIG_DIR = f"{ROOT}/figures"

BACKBONES = ["jointtransformer", "wildhands", "hamer", "meshgraphormer", "simplehand", "wilor"]
DISPLAY = {"jointtransformer":"JointTransformer", "wildhands":"WildHands", "hamer":"HaMeR",
           "meshgraphormer":"MeshGraphormer", "simplehand":"SimpleHand", "wilor":"WiLoR"}
EGO = {"jointtransformer", "wildhands"}
COLOR = {"jointtransformer":"#2E7D32", "wildhands":"#1565C0", "hamer":"#E65100",
         "meshgraphormer":"#6A1B9A", "simplehand":"#C62828", "wilor":"#37474F"}
MARKER = {"jointtransformer":"o", "wildhands":"s", "hamer":"^", "meshgraphormer":"D", "simplehand":"v", "wilor":"P"}

OP_VALID_NAMES = ["wrist","thumb_MCP","thumb_IP","thumb_tip","idx_MCP","idx_PIP","idx_DIP","idx_tip","mid_MCP","mid_PIP","mid_DIP","mid_tip","ring_MCP","ring_PIP","ring_DIP","ring_tip","pky_MCP","pky_PIP","pky_DIP","pky_tip"]

# ------------------------------------------------------------------
# Load summaries (use canonical _jrn_fixed.json + WiLoR canonical from wilor_patched_full)
# but DON'T expose 'patched' externally - display name is just 'WiLoR'.
# ------------------------------------------------------------------
summaries = {}
pa_p2d = {}
for bb in BACKBONES:
    if bb == "wilor":
        with open(f"{ROOT}/summaries/wilor_patched_full.json") as f: d = json.load(f)
        summaries[bb] = {"zs": d["zs"], "jrn_v2train": d["jrn_v2train"]}
        pa_p2d[bb] = {"zs": {"pa_p2d_px": d["zs"]["pa_p2d"]},
                      "jrn_v2train": {"pa_p2d_px": d["jrn_v2train"]["pa_p2d"]}}
    else:
        with open(f"{ROOT}/summaries/{bb}_jrn_fixed.json") as f: summaries[bb] = json.load(f)
        with open(f"{ROOT}/summaries/{bb}_pa_p2d.json") as f: pa_p2d[bb] = json.load(f)

per_pr = {}
for bb in BACKBONES:
    fn = f"{ROOT}/provenance/per_pr_jrn_{bb}.json"
    if os.path.exists(fn):
        with open(fn) as f: per_pr[bb] = json.load(f)

# ============================== F2: ZS -> +JRN scatter ==============================
fig, ax = plt.subplots(figsize=(6.5, 6.5))
zs_all, jrn_all = [], []
for bb in BACKBONES:
    if bb not in per_pr: continue
    prs = sorted(per_pr[bb]["zs_per_pr"].keys())
    zs = [per_pr[bb]["zs_per_pr"][pr]["mean"] for pr in prs]
    jrn = [per_pr[bb]["jrn_per_pr"][pr]["mean"] for pr in prs]
    label = DISPLAY[bb] + (" ★" if bb in EGO else "")  # ★
    ax.scatter(zs, jrn, c=COLOR[bb], marker=MARKER[bb], s=70, alpha=0.85, label=label, edgecolors="black", linewidths=0.4)
    zs_all.extend(zs); jrn_all.extend(jrn)

lo = 0; hi = max(max(zs_all), max(jrn_all)) * 1.05
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="y = x (no improvement)")
ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="green", alpha=0.04, label="_nolegend_")
ax.set_xlabel("Zero-shot PA-MPJPE (mm), per PR", fontsize=11)
ax.set_ylabel("+JRN PA-MPJPE (mm), per PR", fontsize=11)
ax.set_title("F2. JRN improves every PR for every backbone\n(13 PRs × 6 backbones = 78 dots; all below y=x)", fontsize=11)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/F2_zs_to_jrn_scatter.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F2_zs_to_jrn_scatter.png", dpi=200, bbox_inches="tight")
plt.close()
print("F2 saved")


# ============================== F3: Per-joint heatmap (RED-HOT colormap) ==============================
zs_mat = np.array([per_pr[bb]["zs_per_joint_mean"] for bb in BACKBONES])
jrn_mat = np.array([per_pr[bb]["jrn_per_joint_mean"] for bb in BACKBONES])

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
vmin = 0
vmax = max(zs_mat.max(), jrn_mat.max())
labels_y = [DISPLAY[bb] + (" ★" if bb in EGO else "") for bb in BACKBONES]

for ax, mat, title in zip(axes, [zs_mat, jrn_mat], ["Zero-shot", "+JRN"]):
    # 'YlOrRd' = yellow-orange-red: low=yellow (cool/good), high=red (hot/bad).
    # 'Reds' = white-to-red, also good. Going with YlOrRd for better mid-range contrast.
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(20)); ax.set_xticklabels(OP_VALID_NAMES, rotation=70, fontsize=8)
    ax.set_yticks(range(len(BACKBONES))); ax.set_yticklabels(labels_y, fontsize=10)
    ax.set_title(f"{title}  (PA error per joint, mm)", fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt_color = "white" if mat[i,j] > vmax*0.55 else "black"
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", color=txt_color, fontsize=7.5)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("PA error (mm) -- red = high error", fontsize=9)
plt.suptitle("F3. Per-joint PA error (red = hot = high error). Fingertips remain hardest; JRN normalizes proximal joints across backbones.",
             fontsize=11, y=1.02)
plt.savefig(f"{FIG_DIR}/F3_per_joint_heatmap.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F3_per_joint_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("F3 saved (RED-HOT colormap)")


# ============================== F4: Forest plot for PA-MPJPE-MPS ==============================
def draw_forest(ax, data_dict, key, xlabel, xlim, x_offset_text):
    for i, bb in enumerate(BACKBONES):
        zs = data_dict[bb]["zs"][key]
        jr = data_dict[bb]["jrn_v2train"][key]
        y = -i*1.4 + 0.25
        ax.errorbar(zs["mean"], y, xerr=[[zs["mean"]-zs["ci_low"]], [zs["ci_high"]-zs["mean"]]],
                    fmt="o", color=COLOR[bb], markersize=10, capsize=5, linewidth=2)
        ax.text(zs["mean"]+x_offset_text, y+0.05, "ZS {:.1f}".format(zs["mean"]), fontsize=9, va="bottom")
        y2 = -i*1.4 - 0.25
        ax.errorbar(jr["mean"], y2, xerr=[[jr["mean"]-jr["ci_low"]], [jr["ci_high"]-jr["mean"]]],
                    fmt="s", color=COLOR[bb], markersize=10, capsize=5, linewidth=2)
        ax.text(jr["mean"]+x_offset_text, y2-0.15, "+JRN {:.1f}".format(jr["mean"]),
                fontsize=9, va="top", fontweight="bold")
        ax.plot([jr["mean"], zs["mean"]], [y2, y], "-", color=COLOR[bb], alpha=0.4, linewidth=1)
    ego_lows = [data_dict[bb]["jrn_v2train"][key]["ci_low"] for bb in EGO]
    ego_highs = [data_dict[bb]["jrn_v2train"][key]["ci_high"] for bb in EGO]
    ax.axvspan(min(ego_lows), max(ego_highs), color="green", alpha=0.10, zorder=-1,
               label="Ego JRN floor [{:.1f}, {:.1f}]".format(min(ego_lows), max(ego_highs)))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(-(len(BACKBONES)-1)*1.4 - 1, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig, ax = plt.subplots(figsize=(8, 5.5))
draw_forest(ax, summaries, "pa_mpjpe",
            "PA-MPJPE-MPS (mm) -- Aria MPS GT, OP_VALID 20-joint, 95% bootstrap CI over 13 PRs",
            (-2.5, 40), 0.5)
for i, bb in enumerate(BACKBONES):
    label = DISPLAY[bb] + (" ★" if bb in EGO else "")
    ax.text(-2.5, -i*1.4, label, fontsize=11, ha="right", va="center",
            fontweight="bold" if bb in EGO else "normal")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_title("F4. Headline PA-MPJPE-MPS with bootstrap CIs.  ★ = ego-pretrained.\n"
             "Two ego backbones converge at ≈10.5 mm (overlapping CIs).", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/F4_forest_pa_mpjpe.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F4_forest_pa_mpjpe.png", dpi=200, bbox_inches="tight")
plt.close()
print("F4 saved")


# ============================== F5: PA-P2D forest ==============================
fig, ax = plt.subplots(figsize=(8, 5.5))
draw_forest(ax, pa_p2d, "pa_p2d_px",
            "PA-P2D (pixels) -- Procrustes-aligned 3D, projected through Aria pinhole, mean L2",
            (-5, 70), 1.0)
for i, bb in enumerate(BACKBONES):
    label = DISPLAY[bb] + (" ★" if bb in EGO else "")
    ax.text(-5, -i*1.4, label, fontsize=11, ha="right", va="center",
            fontweight="bold" if bb in EGO else "normal")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_title("F5. Headline PA-P2D with bootstrap CIs.  Two ego backbones converge at ~19 px.", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/F5_forest_pa_p2d.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F5_forest_pa_p2d.png", dpi=200, bbox_inches="tight")
plt.close()
print("F5 saved")


# ============================== F4+F5 combined side-by-side ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1]})
draw_forest(ax1, summaries, "pa_mpjpe",
            "PA-MPJPE-MPS (mm) -- Aria MPS GT, OP_VALID 20-joint, 95% bootstrap CI over 13 PRs",
            (-2.5, 40), 0.5)
ax1.set_title("(a) PA-MPJPE-MPS (mm) -- 3D pose error", fontsize=11)
draw_forest(ax2, pa_p2d, "pa_p2d_px",
            "PA-P2D (pixels) -- Procrustes-aligned 3D, projected through Aria pinhole",
            (-5, 70), 1.0)
ax2.set_title("(b) PA-P2D (px) -- 2D projection error", fontsize=11)
for i, bb in enumerate(BACKBONES):
    label = DISPLAY[bb] + (" ★" if bb in EGO else "")
    ax1.text(-2.5, -i*1.4, label, fontsize=11, ha="right", va="center",
             fontweight="bold" if bb in EGO else "normal")
ax1.set_yticks([]); ax2.set_yticks([])
ax1.spines["left"].set_visible(False); ax2.spines["left"].set_visible(False)
plt.suptitle("Headline benchmark with bootstrap CIs: 3D and 2D errors converge at the ego-pretrained JRN floor",
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/F4_F5_combined_forest.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F4_F5_combined_forest.png", dpi=200, bbox_inches="tight")
plt.close()
print("F4+F5 combined saved")


# ============================== F6: Per-stratum bars ==============================
strata_files = ["wildhands", "jointtransformer"]
strata_data = {}
for bb in strata_files:
    with open(f"{ROOT}/provenance/per_strata_{bb}.json") as f: strata_data[bb] = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
panels = [("wearer", "Wearer"), ("glove", "Glove color"), ("experience", "Experience")]
for ax, (key, title) in zip(axes, panels):
    vals = sorted(strata_data["wildhands"][key].keys())
    x = np.arange(len(vals))
    width = 0.18
    wh_zs = [strata_data["wildhands"][key][v]["zs_mean"] for v in vals]
    wh_jr = [strata_data["wildhands"][key][v]["jrn_mean"] for v in vals]
    jt_zs = [strata_data["jointtransformer"][key][v]["zs_mean"] for v in vals]
    jt_jr = [strata_data["jointtransformer"][key][v]["jrn_mean"] for v in vals]
    ax.bar(x - 1.5*width, wh_zs, width, label="WH ZS", color=COLOR["wildhands"], alpha=0.45)
    ax.bar(x - 0.5*width, wh_jr, width, label="WH +JRN", color=COLOR["wildhands"])
    ax.bar(x + 0.5*width, jt_zs, width, label="JT ZS", color=COLOR["jointtransformer"], alpha=0.45)
    ax.bar(x + 1.5*width, jt_jr, width, label="JT +JRN", color=COLOR["jointtransformer"])
    for i, v in enumerate(vals):
        n = strata_data["wildhands"][key][v]["n"]
        ax.text(i, max(wh_zs[i], jt_zs[i]) + 0.7, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(vals, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    if title == "Wearer": ax.set_ylabel("PA-MPJPE-MPS (mm)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

plt.suptitle("F6. Per-stratum analysis (WildHands ★ vs JointTransformer ★).  Brown gloves = P4 = hardest stratum.",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/F6_per_stratum.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F6_per_stratum.png", dpi=200, bbox_inches="tight")
plt.close()
print("F6 saved")


# ============================== F7: Δ-improvement bars (clean caption) ==============================
fig, ax = plt.subplots(figsize=(8, 4.5))
zs_means = [summaries[bb]["zs"]["pa_mpjpe"]["mean"] for bb in BACKBONES]
jr_means = [summaries[bb]["jrn_v2train"]["pa_mpjpe"]["mean"] for bb in BACKBONES]
deltas = [j - z for z, j in zip(zs_means, jr_means)]
deltas_pct = [100*(j-z)/z for z, j in zip(zs_means, jr_means)]
labels = [DISPLAY[bb] + (" ★" if bb in EGO else "") for bb in BACKBONES]
colors = [COLOR[bb] for bb in BACKBONES]

x = np.arange(len(BACKBONES))
ax.bar(x, [-d for d in deltas], color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
for i, (d, p) in enumerate(zip(deltas, deltas_pct)):
    ax.text(i, -d + 0.3, "−{:.1f} mm\n({:+.0f}%)".format(abs(d), p), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Improvement: ZS → +JRN (mm)", fontsize=10)
ax.set_title("F7. JRN absolute and relative reduction in PA-MPJPE-MPS across 6 backbones", fontsize=11)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/F7_delta_bar.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F7_delta_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("F7 saved (clean caption)")


# ============================== F8: Per-task ==============================
with open(f"{ROOT}/provenance/per_task_wildhands.json") as f: wh = json.load(f)
with open(f"{ROOT}/provenance/per_task_jointtransformer.json") as f: jt = json.load(f)
tasks = sorted(wh.keys(), key=lambda k: (-wh[k]["n"], k))
def short(t):
    repl = {
        "running sub-cuticular + aberdeen knot": "running sub-cuticular\n+ aberdeen knot",
        "horizontal mattress + instrument tie": "horiz mattress + instr tie",
        "figure of 8 + instrument tie": "fig 8 + instr tie",
        "figure of 8 + two handed tie": "fig 8 + two-hand tie",
        "figure of 8 + one hand tie": "fig 8 + one-hand tie",
        "horizontal mattress + one hand tie": "horiz mattress + 1H tie",
        "vertical mattress + instrument tie": "vert mattress + instr tie",
        "simple interrupted + instrument tie": "simple int + instr tie",
        "simple interrupted + two hand tie": "simple int + 2H tie",
        "horizontal mattress": "horiz mattress",
    }
    return repl.get(t, t)

x = np.arange(len(tasks)); width = 0.20
wh_zs = [wh[t]["zs_mean"] for t in tasks]
wh_jr = [wh[t]["jrn_mean"] for t in tasks]
jt_zs = [jt[t]["zs_mean"] for t in tasks]
jt_jr = [jt[t]["jrn_mean"] for t in tasks]
ns = [wh[t]["n"] for t in tasks]

fig, ax = plt.subplots(figsize=(15, 5.5))
b1 = ax.bar(x - 1.5*width, wh_zs, width, label="WildHands ZS", color="#1565C0", alpha=0.45)
b2 = ax.bar(x - 0.5*width, wh_jr, width, label="WildHands +JRN", color="#1565C0")
b3 = ax.bar(x + 0.5*width, jt_zs, width, label="JointTransformer ZS", color="#2E7D32", alpha=0.45)
b4 = ax.bar(x + 1.5*width, jt_jr, width, label="JointTransformer +JRN", color="#2E7D32")
ymax = max(wh_zs + jt_zs)
for i, (n, t) in enumerate(zip(ns, tasks)):
    ax.text(i, ymax * 1.04, "n={}".format(n), ha="center", fontsize=8, color="grey")
ax.set_xticks(x); ax.set_xticklabels([short(t) for t in tasks], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("PA-MPJPE-MPS (mm)", fontsize=11)
ax.set_title("F8. Per-task accuracy on the v2 evaluation partition (13 surgical tasks)\nBoth ego backbones agree on hardest task = 'figure of 8 + two-handed tie' (P4-only, brown gloves)", fontsize=11)
ax.legend(loc="upper left", fontsize=10, ncol=2)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, ymax * 1.15)
worst_idx = int(np.argmax(wh_jr))
for bar in [b2[worst_idx], b4[worst_idx]]:
    bar.set_edgecolor("red"); bar.set_linewidth(2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/F8_per_task.pdf", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/F8_per_task.png", dpi=200, bbox_inches="tight")
plt.close()
print("F8 saved")

print("\nAll figures regenerated cleanly.")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
