#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figures 2 and 3 -- Discrimination curves (Original Article, Insights into Imaging).

Figure 2  ROC curves, two panels:
    a  report-derived held-out test set (n = 5,341) scored against `target`
    b  adjudicated expert reference (n = 500) scored against `final_target`

Figure 3  Precision-recall curves, same two references, each with a dashed
          horizontal no-skill line at that reference's worsening prevalence.

Every curve is recomputed from the unified PCA=128 score table at run time.
No value is transcribed from an earlier version of these figures.

Inputs
------
  <root>/data/unified128_test_scores.csv
      pair_key, patient_cluster_id, target, current_score, difference_score, temporal_score
  <root>/data/final_adjudicated_cases.csv
      joined on pair_key; `final_target` is the adjudicated expert label

Outputs
-------
  <root>/generated_figures/Figure 2.tif             300 dpi, LZW, RGB
  <root>/generated_figures/Figure 3.tif             300 dpi, LZW, RGB
  <root>/generated_figures/preview_figure2.png       300 dpi (review only)
  <root>/generated_figures/preview_figure3.png       300 dpi (review only)

Correctness gate
----------------
Each of the 12 computed statistics (3 models x 2 references x {AUC, AP}) is
checked against the value the manuscript text reports. Any mismatch beyond
ATOL aborts the build with a non-zero exit status and a printed diff, so the
figures cannot silently disagree with the prose.

Design
------
Panel letters are LOWERCASE per journal style. Series are separated by BOTH
hue and dash pattern, using an Okabe-Ito colour-blind-safe palette whose three
hues also differ in luminance, so the panels survive greyscale reproduction.

Run from the repository root:
    python scripts/figures/build_roc_pr_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data"
FIGDIR = ROOT / "generated_figures"

SCORES_CSV = DATA / "unified128_test_scores.csv"
EXPERT_CSV = DATA / "final_adjudicated_cases.csv"

FIG2_TIF = FIGDIR / "Figure 2.tif"
FIG3_TIF = FIGDIR / "Figure 3.tif"
FIG2_PNG = FIGDIR / "preview_figure2.png"
FIG3_PNG = FIGDIR / "preview_figure3.png"

DPI = 300

# --------------------------------------------------------------------------
# Expected values -- the numbers the manuscript reports. The build fails if the
# data no longer reproduce them.
# --------------------------------------------------------------------------

EXPECTED = {
    ("test", "Current-only"):    {"auc": 0.5839, "ap": 0.2843},
    ("test", "Difference-only"): {"auc": 0.6542, "ap": 0.3482},
    ("test", "Temporal"):        {"auc": 0.6679, "ap": 0.3577},
    ("expert", "Current-only"):    {"auc": 0.6434, "ap": 0.4335},
    ("expert", "Difference-only"): {"auc": 0.7678, "ap": 0.6154},
    ("expert", "Temporal"):        {"auc": 0.7845, "ap": 0.5965},
}
ATOL = 1e-4          # expectations are quoted to 4 dp; half-ulp is 5e-5

N_TEST_EXPECTED = 5_341
N_EXPERT_EXPECTED = 500

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

TEXT = "#1a1a1a"
SPINE = "#4d4d4d"
GRID = "#e6e6e6"          # light grey (L=0.79), lighter than every curve
# Reference lines (chance / no-skill) must read as subordinate to every data
# series in greyscale as well as colour, so this grey is deliberately LIGHTER
# than all three series (L=0.58 vs max series L=0.42) rather than mid-grey:
# a mid-grey (#8c8c8c, L=0.26) collides with the Temporal green (L=0.26) and
# the two become the same shade once the figure is printed in greyscale.
CHANCE = "#c8c8c8"
CHANCE_LW = 0.9           # slightly heavier to stay legible now that it is light

# Okabe-Ito, colour-blind safe. The three hues are also chosen to separate by
# sRGB relative luminance -- blue 0.15 < green 0.26 < orange 0.42 -- so the
# series stay distinguishable in greyscale; dash pattern and weight carry the
# distinction independently of hue.
MODELS = [
    # key,               label,             colour,    dash,             lw
    ("current_score",    "Current-only",    "#0072B2", (0, (1, 1.2)),    1.4),
    ("difference_score", "Difference-only", "#E69F00", (0, (4.5, 1.8)),  1.5),
    ("temporal_score",   "Temporal",        "#009E73", (0, ()),          1.7),
]

FIG_W_MM = 180.0
MM_PER_IN = 25.4

# Margins chosen so each axes box comes out square without aspect-ratio
# shrinking (which would strand whitespace between the panels).
LEFT, RIGHT, BOTTOM, TOP, WSPACE = 0.070, 0.990, 0.115, 0.930, 0.280

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.0,
        "text.color": TEXT,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.7,
        "axes.labelcolor": TEXT,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "legend.fontsize": 6.6,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_references() -> dict[str, dict]:
    """Load the score table and build the two evaluation references."""
    if not SCORES_CSV.exists():
        sys.exit(f"FATAL: score table not found: {SCORES_CSV}")
    if not EXPERT_CSV.exists():
        sys.exit(f"FATAL: expert reference not found: {EXPERT_CSV}")

    scores = pd.read_csv(SCORES_CSV)
    expert = pd.read_csv(EXPERT_CSV, usecols=["pair_key", "final_target"])

    needed = {"pair_key", "target", "current_score", "difference_score", "temporal_score"}
    missing = needed - set(scores.columns)
    if missing:
        sys.exit(f"FATAL: score table missing columns: {sorted(missing)}")

    if scores["pair_key"].duplicated().any():
        sys.exit("FATAL: duplicate pair_key in score table; join would fan out")
    if expert["pair_key"].duplicated().any():
        sys.exit("FATAL: duplicate pair_key in expert reference")

    score_cols = [k for k, *_ in MODELS]
    if scores[score_cols + ["target"]].isna().any().any():
        sys.exit("FATAL: null scores/labels in the held-out score table")

    # Expert labels carry the scores of an earlier fit in their own columns; those
    # are deliberately ignored. Only pair_key and final_target are read, and the
    # PCA=128 scores are taken from the unified table.
    merged = expert.merge(scores, on="pair_key", how="inner", validate="one_to_one")
    if len(merged) != len(expert):
        sys.exit(
            f"FATAL: only {len(merged)}/{len(expert)} expert pair_key values matched "
            "the score table"
        )
    if merged["final_target"].isna().any():
        sys.exit("FATAL: null final_target after join")

    if len(scores) != N_TEST_EXPECTED:
        sys.exit(f"FATAL: held-out n = {len(scores)}, expected {N_TEST_EXPECTED}")
    if len(merged) != N_EXPERT_EXPECTED:
        sys.exit(f"FATAL: expert n = {len(merged)}, expected {N_EXPERT_EXPECTED}")

    return {
        "test": {
            "df": scores,
            "y": scores["target"].to_numpy(),
            "letter": "a",
            "title": "Report-derived held-out test set (n = 5,341)",
        },
        "expert": {
            "df": merged,
            "y": merged["final_target"].to_numpy(),
            "letter": "b",
            "title": "Adjudicated expert reference (n = 500)",
        },
    }


def compute_metrics(refs: dict) -> dict:
    """AUC and AP for every model under every reference, straight from the data."""
    out = {}
    for ref_key, ref in refs.items():
        y = ref["y"]
        for score_col, label, *_ in MODELS:
            s = ref["df"][score_col].to_numpy()
            out[(ref_key, label)] = {
                "auc": roc_auc_score(y, s),
                "ap": average_precision_score(y, s),
            }
    return out


def verify(metrics: dict) -> None:
    """Fail loudly if the recomputed statistics drift from the reported ones."""
    rows, bad = [], []
    for ref_key, ref_name in (("test", "Held-out test"), ("expert", "Expert reference")):
        for _, label, *_ in MODELS:
            got, exp = metrics[(ref_key, label)], EXPECTED[(ref_key, label)]
            for stat in ("auc", "ap"):
                d = abs(got[stat] - exp[stat])
                ok = d <= ATOL
                if not ok:
                    bad.append(
                        f"  {ref_name:16s} {label:16s} {stat.upper():3s}  "
                        f"computed {got[stat]:.4f}  expected {exp[stat]:.4f}  "
                        f"diff {d:.5f}"
                    )
                rows.append(
                    (ref_name, label, stat.upper(), got[stat], exp[stat], d, ok)
                )

    print()
    print("Computed vs expected (recomputed from unified128 scores at PCA = 128)")
    print("-" * 78)
    print(f"{'Reference':17s}{'Model':17s}{'Stat':5s}{'Computed':>10s}"
          f"{'Expected':>10s}{'Diff':>10s}  OK")
    print("-" * 78)
    for ref_name, label, stat, g, e, d, ok in rows:
        print(f"{ref_name:17s}{label:17s}{stat:5s}{g:10.4f}{e:10.4f}"
              f"{d:10.5f}  {'yes' if ok else 'NO'}")
    print("-" * 78)

    if bad:
        print()
        print("!" * 78, file=sys.stderr)
        print("METRIC MISMATCH -- figures NOT written. The data no longer reproduce",
              file=sys.stderr)
        print("the values reported in the manuscript:", file=sys.stderr)
        print("\n".join(bad), file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        sys.exit(1)

    print(f"All {len(rows)} statistics match the expected values "
          f"(|diff| <= {ATOL:g}).")


def _luminance(hex_colour: str) -> float:
    """sRGB relative luminance (WCAG), i.e. what greyscale conversion sees."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def qa_palette() -> None:
    """
    Guard the greyscale story. Hue alone is not enough: the figure must survive
    printing in greyscale, so the three series must separate in luminance, the
    reference line must sit ABOVE every series (subordinate), and the gridlines
    must be lighter than everything.
    """
    MIN_SERIES_GAP = 0.08     # series also differ in dash pattern and weight
    MIN_REF_GAP = 0.12        # reference line must be clearly off every series

    series = {label: _luminance(c) for _, label, c, *_ in MODELS}
    ref, grid = _luminance(CHANCE), _luminance(GRID)

    print()
    print("Palette QA -- sRGB relative luminance (what greyscale printing sees)")
    for label, L in series.items():
        print(f"  {label:16s} L = {L:.3f}")
    print(f"  {'reference line':16s} L = {ref:.3f}")
    print(f"  {'gridlines':16s} L = {grid:.3f}")

    problems = []
    items = list(series.items())
    for i, (la, La) in enumerate(items):
        for lb, Lb in items[i + 1:]:
            if abs(La - Lb) < MIN_SERIES_GAP:
                problems.append(
                    f"  series {la!r} and {lb!r} are {abs(La - Lb):.3f} apart in "
                    f"luminance (< {MIN_SERIES_GAP}); they will merge in greyscale"
                )
    for label, L in series.items():
        if abs(ref - L) < MIN_REF_GAP:
            problems.append(
                f"  reference line and series {label!r} are {abs(ref - L):.3f} apart "
                f"(< {MIN_REF_GAP}); they will be the same shade in greyscale"
            )
    if ref <= max(series.values()):
        problems.append(
            "  reference line is not lighter than every series; it will compete "
            "with the data instead of receding"
        )
    if grid <= max(list(series.values()) + [ref]):
        problems.append("  gridlines are not lighter than every drawn line")

    if problems:
        print()
        print("Palette QA failed:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)

    gaps = [abs(a - b) for i, (_, a) in enumerate(items) for _, b in items[i + 1:]]
    print(f"Palette QA: min series separation {min(gaps):.3f}, "
          f"reference line clear of every series, grid lightest.")


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def new_canvas() -> tuple[plt.Figure, np.ndarray]:
    """180 mm-wide canvas whose two axes boxes are exactly square."""
    fig_w_in = FIG_W_MM / MM_PER_IN
    axes_w_frac = (RIGHT - LEFT) / (2.0 + WSPACE)
    axes_w_in = axes_w_frac * fig_w_in
    fig_h_in = axes_w_in / (TOP - BOTTOM)          # square box

    fig, axes = plt.subplots(1, 2, figsize=(fig_w_in, fig_h_in))
    fig.subplots_adjust(
        left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=WSPACE
    )
    return fig, axes


def style_panel(ax, letter: str, title: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.012, 1.012)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.5, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", pad=1.8)
    ax.set_title(title, pad=4.5, fontsize=7.2)
    # Lowercase panel letter, bold, top-left, outside the plotting box.
    ax.text(
        -0.155, 1.045, letter,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold", color=TEXT,
        ha="left", va="bottom",
    )


def build_roc(refs: dict, metrics: dict) -> plt.Figure:
    fig, axes = new_canvas()
    for ax, ref_key in zip(axes, ("test", "expert")):
        ref = refs[ref_key]
        y = ref["y"]

        ax.plot([0, 1], [0, 1], linestyle=(0, (3, 3)), color=CHANCE,
                linewidth=CHANCE_LW, zorder=1, label="Chance")

        for score_col, label, colour, dash, lw in MODELS:
            fpr, tpr, _ = roc_curve(y, ref["df"][score_col].to_numpy())
            auc = metrics[(ref_key, label)]["auc"]
            ax.plot(fpr, tpr, color=colour, linestyle=dash, linewidth=lw,
                    solid_capstyle="round", dash_capstyle="round", zorder=3,
                    label=f"{label} (AUC = {auc:.3f})")

        style_panel(ax, ref["letter"], ref["title"])
        ax.set_xlabel("1 - specificity (false-positive rate)")
        ax.set_ylabel("Sensitivity (true-positive rate)")

        leg = ax.legend(loc="lower right", frameon=False,
                        handlelength=2.6, handletextpad=0.6,
                        labelspacing=0.42, borderpad=0.2)
        for t in leg.get_texts():
            t.set_color(TEXT)
    return fig


def build_pr(refs: dict, metrics: dict) -> plt.Figure:
    fig, axes = new_canvas()
    for ax, ref_key in zip(axes, ("test", "expert")):
        ref = refs[ref_key]
        y = ref["y"]
        prev = float(np.mean(y))

        ax.axhline(prev, linestyle=(0, (3, 3)), color=CHANCE, linewidth=CHANCE_LW,
                   zorder=1, label=f"No skill ({prev:.3f})")

        for score_col, label, colour, dash, lw in MODELS:
            precision, recall, _ = precision_recall_curve(
                y, ref["df"][score_col].to_numpy()
            )
            ap = metrics[(ref_key, label)]["ap"]
            # steps-post matches the step-wise definition used by average_precision
            ax.plot(recall, precision, color=colour, linestyle=dash, linewidth=lw,
                    drawstyle="steps-post", solid_capstyle="round",
                    dash_capstyle="round", zorder=3,
                    label=f"{label} (AP = {ap:.3f})")

        style_panel(ax, ref["letter"], ref["title"])
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylabel("Precision (positive predictive value)")

        leg = ax.legend(loc="upper right", frameon=False,
                        handlelength=2.6, handletextpad=0.6,
                        labelspacing=0.42, borderpad=0.2)
        for t in leg.get_texts():
            t.set_color(TEXT)
    return fig


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export(fig: plt.Figure, png_path: Path, tif_path: Path, name: str) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tif_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=DPI, facecolor="white")

    # Flatten to RGB (no alpha channel) and write LZW-compressed TIFF at 300 dpi.
    with Image.open(png_path) as im:
        im.convert("RGB").save(
            tif_path, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI)
        )
    plt.close(fig)

    with Image.open(tif_path) as im:
        w, h = im.size
        print(f"{name}")
        print(f"  TIF  : {tif_path}")
        print(f"  size : {w} x {h} px  "
              f"({w / DPI * MM_PER_IN:.1f} x {h / DPI * MM_PER_IN:.1f} mm)")
        print(f"  mode : {im.mode}")
        print(f"  dpi  : {im.info.get('dpi')}")
        print(f"  compr: {im.info.get('compression')}")
        print(f"  bytes: {tif_path.stat().st_size:,}")
    print(f"  PNG  : {png_path}  ({png_path.stat().st_size:,} bytes)")


def main() -> None:
    refs = load_references()
    for key, ref in refs.items():
        print(f"{key:7s} n = {len(ref['df']):5d}   "
              f"prevalence = {float(np.mean(ref['y'])):.4f}")

    metrics = compute_metrics(refs)
    verify(metrics)                       # exits non-zero on any mismatch
    qa_palette()                          # exits non-zero if greyscale would fail

    print()
    export(build_roc(refs, metrics), FIG2_PNG, FIG2_TIF, "Figure 2 (ROC)")
    print()
    export(build_pr(refs, metrics), FIG3_PNG, FIG3_TIF, "Figure 3 (precision-recall)")


if __name__ == "__main__":
    main()
