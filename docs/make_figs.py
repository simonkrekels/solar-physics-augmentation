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
