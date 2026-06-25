"""Regenerate the figures embedded in docs/findings.tex.

Run from the repository root (reads the committed result CSVs):
    uv run python docs/make_figs.py

Outputs: docs/fig_rare_recall.png, docs/fig_d1_auc.png
(docs/montage_v2.png is produced separately by the v2 generator inspection.)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Fig 1: rare-class recall across the 4 main conditions (per-class, bootstrap CIs) ---
pc = pd.read_csv("verify_perclass.csv")
rare = pc[pc["rare"]].copy()
conds = ["clean", "oversample", "randaugment", "physics"]
classes = ["Diode-Multi", "Hot-Spot", "Hot-Spot-Multi", "Soiling"]
colors = {"clean": "#1f77b4", "oversample": "#ff7f0e", "randaugment": "#2ca02c", "physics": "#d62728"}
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(classes)); w = 0.2
for i, c in enumerate(conds):
    sub = rare[rare.condition == c].set_index("class").reindex(classes)
    vals = sub["recall"].values
    lo = vals - sub["ci_lo"].values; hi = sub["ci_hi"].values - vals
    ax.bar(x + (i - 1.5) * w, vals, w, yerr=[lo, hi], capsize=2, label=c, color=colors[c])
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
ax.set_ylabel("recall"); ax.set_ylim(0, 1.05)
ax.set_title("Rare-class recall by condition (95% bootstrap CI)")
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout(); plt.savefig("docs/fig_rare_recall.png", dpi=130, bbox_inches="tight")
print("fig_rare_recall saved")

# --- Fig 2: D1 discriminator AUC (lower = more realistic); values from d1_*.log runs ---
labels = ["v1\n(blur0)", "v1\nblur2", "v1\nblur4", "v2\n+grain", "v2\nno-grain"]
auc = [0.966, 0.934, 0.867, 1.000, 0.934]
cols = ["#1f77b4"] * 3 + ["#d62728", "#9467bd"]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
b = ax.bar(labels, auc, color=cols)
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(4.3, 0.515, "chance (0.5)", fontsize=8, ha="right", color="gray")
ax.axhline(0.75, ls=":", c="green", lw=1); ax.text(0, 0.76, "refiner target ≤0.75", fontsize=8, color="green")
ax.set_ylim(0, 1.05); ax.set_ylabel("discriminator AUC")
ax.set_title("Synthetic-vs-real separability (lower = more realistic)")
for r, v in zip(b, auc):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("docs/fig_d1_auc.png", dpi=130, bbox_inches="tight")
print("fig_d1_auc saved")

# --- Fig 3: realism gate across generator approaches (D1 AUC, from d1_*.log) ---
labels = ["v1\nsteady-state", "v2\nno-grain", "SimGAN\nrefiner", "perturb\nreal faults"]
auc = [0.966, 0.934, 0.980, 0.711]
cols = ["#d62728", "#d62728", "#d62728", "#2ca02c"]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
b = ax.bar(labels, auc, color=cols)
ax.axhline(0.75, ls=":", c="green", lw=1.2); ax.text(3.45, 0.76, "realism gate ≤0.75", fontsize=8, color="green", ha="right")
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(0, 0.515, "chance", fontsize=8, color="gray")
ax.set_ylim(0, 1.06); ax.set_ylabel("synthetic-vs-real discriminator AUC")
ax.set_title("Closing the domain gap: only perturbing real faults passes")
for r, v in zip(b, auc):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("docs/fig_realism_gate.png", dpi=130, bbox_inches="tight")
print("fig_realism_gate saved")

# --- Fig 4: logit-adjustment τ tradeoff (verify_summary.csv + analyze aggregate CIs) ---
tau = [1.0, 1.5, 2.0]
rare = [0.634, 0.696, 0.728]; rlo = [0.57, 0.64, 0.67]; rhi = [0.69, 0.76, 0.79]
acc = [0.808, 0.763, 0.683]
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.errorbar(tau, rare, yerr=[[r - l for r, l in zip(rare, rlo)], [h - r for r, h in zip(rare, rhi)]],
            marker="o", capsize=3, color="#d62728", label="rare-class recall (95% CI)")
ax.plot(tau, acc, marker="s", color="#1f77b4", label="accuracy")
ax.axhline(0.612, ls="--", color="#d62728", alpha=0.5); ax.text(2.02, 0.612, "clean rare-recall 0.612", fontsize=8, color="#d62728", va="center")
ax.axhline(0.782, ls="--", color="#1f77b4", alpha=0.5); ax.text(2.02, 0.782, "clean acc 0.782", fontsize=8, color="#1f77b4", va="center")
ax.set_xlabel("logit-adjustment τ"); ax.set_ylabel("score"); ax.set_xticks(tau)
ax.set_ylim(0.55, 0.84); ax.set_xlim(0.9, 2.45)
ax.set_title("Logit adjustment: τ trades accuracy for rare-fault recall")
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("docs/fig_logit_adj.png", dpi=130, bbox_inches="tight")
print("fig_logit_adj saved")
