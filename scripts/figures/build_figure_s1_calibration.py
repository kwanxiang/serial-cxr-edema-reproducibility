"""
Supplementary Figure S1 — reliability (calibration) curves for the three
representations under each reference, from locked matched-128 scores.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data"
CASES = DATA / "final_adjudicated_cases.csv"
SCORES = DATA / "unified128_test_scores.csv"
FIGDIR = ROOT / "generated_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

scores = pd.read_csv(SCORES)
cases = pd.read_csv(CASES)
sub = cases.merge(scores[["pair_key", "current_score", "difference_score", "temporal_score"]],
                  on="pair_key", how="left", validate="one_to_one")

KINDS = [("current_score", "Current-only", "#4477AA", ":"),
         ("difference_score", "Difference-only", "#EE7733", "--"),
         ("temporal_score", "Temporal", "#009988", "-")]
refs = [("a", "Report-derived held-out test set (n = 5,341)",
         scores["target"].to_numpy(int), {k: scores[k].to_numpy(float) for k, *_ in KINDS}),
        ("b", "Adjudicated expert reference (n = 500)",
         sub["final_target"].to_numpy(int), {k: sub[k].to_numpy(float) for k, *_ in KINDS})]

BINS = 10
edges = np.linspace(0, 1, BINS + 1)
mid = (edges[:-1] + edges[1:]) / 2

fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.6), dpi=300)
for ax, (letter, title, y, S) in zip(axes, refs):
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=1)
    for col, label, color, ls in KINDS:
        p = np.clip(S[col], 1e-6, 1 - 1e-6)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, BINS - 1)
        xs, ys = [], []
        for b in range(BINS):
            m = idx == b
            if m.sum() >= 5:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=color, ls=ls, label=label, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    if letter == "a":
        ax.set_ylabel("Observed worsening frequency")
    ax.set_title(title, fontsize=8.5)
    ax.text(-0.06, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(True, color="0.9", lw=0.5)

fig.tight_layout()
tif = FIGDIR / "Figure S1.tif"
png = FIGDIR / "preview_figure_s1.png"
fig.savefig(tif, dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(png, dpi=150)
# Matplotlib may emit an RGBA TIFF even though the artwork is opaque. Convert
# explicitly so the submission file has the journal-compatible RGB colour mode.
with Image.open(tif) as src:
    src.convert("RGB").save(tif, dpi=(300, 300), compression="tiff_lzw")
print("wrote", tif)
print("wrote", png)
im = Image.open(tif)
print("TIF:", im.size, im.mode, im.info.get("dpi"))
