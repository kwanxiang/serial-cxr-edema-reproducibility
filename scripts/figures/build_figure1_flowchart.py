#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1 -- Study flow diagram (Original Article, Insights into Imaging).

Renders a single-panel, CONSORT/STARD-style flow diagram running top-to-bottom
from the initial CheXTemporal retrieval to the final analytic cohort, the
patient-level grouped three-way split, and the adjudicated expert reference
standard. Exclusions branch to the right and are always reported as positive
counts.

Outputs
-------
  <root>/generated_figures/Figure 1.tif             300 dpi, LZW, RGB
  <root>/generated_figures/preview_figure1.png       300 dpi (review only)

Layout is specified directly in millimetres: the axes span 0..180 mm in x and
0..H mm in y, and the figure is created at exactly that physical size, so one
data unit == one millimetre in both directions. Font sizes stay in points.
The y axis is inverted so the diagram can be stacked top-down.

A QA pass measures every rendered string against the width of the box that
contains it and fails the build on overflow, so the figure cannot silently
regress into clipped text.

Run from the repository root:
    python scripts/figures/build_figure1_flowchart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

# --------------------------------------------------------------------------
# Output locations
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIGDIR = ROOT / "generated_figures"
TIF_PATH = FIGDIR / "Figure 1.tif"
PNG_PATH = FIGDIR / "preview_figure1.png"

DPI = 300

# --------------------------------------------------------------------------
# Cohort numbers -- single source of truth for the figure.
# Counts marked "derived" are exact complements of the retained counts and are
# shown so that the diagram balances at every step.
# --------------------------------------------------------------------------

N_SOURCE = 416_038          # CheXTemporal training split, sentence-level rows
N_EDEMA = 68_184            # rows mentioning edema
N_FRONTAL = 55_292          # rows with a frontal image in current AND prior study
N_PAIRS = 28_203            # unique image pairs after aggregation
N_CONFLICT = 1_952          # pairs with both worsening and non-worsening votes
N_FINAL = 26_251            # final analytic cohort (pairs)
N_FINAL_PT = 10_704         # final analytic cohort (patients)
N_FINAL_POS = 6_065         # worsening pairs in the final cohort

N_EXCL_EDEMA = N_SOURCE - N_EDEMA        # derived: 347,854
N_EXCL_FRONTAL = N_EDEMA - N_FRONTAL     # derived: 12,892

SPLIT = {                   # patient-level grouped split: (pairs, patients)
    "train": (18_279, 7_492),
    "tune": (2_631, 1_071),
    "test": (5_341, 2_141),
}
N_TEST_POS = 1_199          # worsening pairs in the held-out test set

N_EXPERT = 500              # patient-unique sample sent to radiologist review
N_TEST_POS_SUB = 112        # worsening pairs in the sample under report-derived labels (22.4%, fixed by stratification)
N_EXPERT_POS = 150          # worsening pairs under the adjudicated reference


def check_integrity() -> None:
    """Fail loudly rather than ship a diagram whose arithmetic does not close."""
    assert N_PAIRS - N_CONFLICT == N_FINAL, "step 5 does not balance"
    assert sum(v[0] for v in SPLIT.values()) == N_FINAL, "split pairs != cohort"
    assert sum(v[1] for v in SPLIT.values()) == N_FINAL_PT, "split patients != cohort"
    assert N_EXCL_EDEMA == 347_854 and N_EXCL_FRONTAL == 12_892
    assert N_EXPERT <= SPLIT["test"][0], "expert sample exceeds test set"
    assert round(100 * N_FINAL_POS / N_FINAL, 1) == 23.1
    assert round(100 * N_TEST_POS / SPLIT["test"][0], 1) == 22.4
    assert round(100 * N_EXPERT_POS / N_EXPERT, 1) == 30.0


# --------------------------------------------------------------------------
# Geometry (millimetres)
# --------------------------------------------------------------------------

FIG_W = 180.0
MARGIN = 3.0

TRUNK_X = 70.0              # vertical spine of the main flow
MAIN_W = 74.0               # main-flow box width          -> spans 33..107
EXCL_X0 = 111.0             # left edge of exclusion column
EXCL_W = 66.0               # exclusion box width          -> spans 111..177

SPLIT_W = 40.0              # three parallel boxes, centred on the trunk
SPLIT_GAP = 7.0
SPLIT_CX = (
    TRUNK_X - (SPLIT_W + SPLIT_GAP),
    TRUNK_X,
    TRUNK_X + (SPLIT_W + SPLIT_GAP),
)                           # -> 23, 70, 117

EXPERT_CX = SPLIT_CX[2]     # expert branch descends from the held-out test box
EXPERT_W = 72.0             # -> spans 81..153

LINE_H = 3.2                # mm per text line
PAD_V = 2.2                 # mm padding above/below the text block
PAD_H = 2.2                 # mm minimum clearance each side (QA threshold)
ROUND = 1.1                 # corner rounding radius (mm)

GAP_PLAIN = 6.0             # vertical gap spanned by a bare arrow
GAP_EXCL = 16.0             # vertical gap that must accommodate an exclusion box
FAN_H = 14.0                # vertical room for the three-way split connector
FAN_BAR = 0.60              # bar position within FAN_H (fraction from the top)
FAN_NOTE = 0.28             # split-note position within FAN_H

# --------------------------------------------------------------------------
# Typography and colour
# --------------------------------------------------------------------------

FS_TITLE = 7.4
FS_BODY = 7.0
FS_EXCL_TITLE = 6.8
FS_EXCL_BODY = 6.5
FS_SPLIT_TITLE = 7.0
FS_SPLIT_BODY = 6.7
FS_NOTE = 6.5

TEXT = "#000000"
BORDER = "#9a9a9a"          # thin grey box borders
BORDER_STRONG = "#4d4d4d"   # final analytic cohort / expert reference standard
ARROW = "#404040"
FACE = "#ffffff"
FACE_EXCL = "#f5f5f5"
FACE_KEY = "#eeeeee"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": TEXT,
        "axes.linewidth": 0.0,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Registry for the QA pass: (text artist, permitted width in mm, description)
_TEXTS: list[tuple[object, float, str]] = []


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------

BOX_SOURCE = (
    "CheXTemporal training split",
    [f"{N_SOURCE:,} sentence-level temporal annotations"],
)

EXCL_EDEMA = (
    f"Excluded: {N_EXCL_EDEMA:,} annotations",
    ["No “edema” mention in the pathology field",
     "or the source sentence (case-insensitive)"],
)

BOX_EDEMA = (
    "Edema-related annotations",
    [f"n = {N_EDEMA:,}"],
)

EXCL_FRONTAL = (
    f"Excluded: {N_EXCL_FRONTAL:,} annotations",
    ["No frontal image identifiable in the",
     "current study, the prior study, or both"],
)

BOX_FRONTAL = (
    "Annotations with a frontal image",
    ["in both the current and the prior study",
     f"n = {N_FRONTAL:,}"],
)

BOX_PAIRS = (
    "Aggregation to the image-pair level",
    [f"{N_PAIRS:,} unique current–prior image pairs"],
)

EXCL_CONFLICT = (
    f"Excluded: {N_CONFLICT:,} internally conflicting pairs",
    ["Image pairs carrying both worsening",
     "and non-worsening sentence-level votes"],
)

BOX_FINAL = (
    "Final analytic cohort",
    [f"{N_FINAL:,} image pairs from {N_FINAL_PT:,} patients",
     f"Worsening prevalence {100 * N_FINAL_POS / N_FINAL:.1f}% "
     f"({N_FINAL_POS:,}/{N_FINAL:,})"],
)

SPLIT_BOXES = [
    ("Training set",
     [f"{SPLIT['train'][0]:,} pairs", f"{SPLIT['train'][1]:,} patients"]),
    ("Tuning set",
     [f"{SPLIT['tune'][0]:,} pairs", f"{SPLIT['tune'][1]:,} patients"]),
    ("Held-out test set",
     [f"{SPLIT['test'][0]:,} pairs", f"{SPLIT['test'][1]:,} patients",
      f"Worsening {N_TEST_POS:,} ({100 * N_TEST_POS / SPLIT['test'][0]:.1f}%)"]),
]

BOX_SAMPLE = (
    "Patient-unique sample, label-stratified",
    [f"{N_EXPERT} pairs; worsening {N_TEST_POS_SUB} "
     f"({100 * N_TEST_POS_SUB / N_EXPERT:.1f}%), fixed by design"],
)

BOX_REVIEW = (
    "Independent two-radiologist review",
    ["Disagreements resolved by",
     "third-radiologist adjudication"],
)

BOX_EXPERT = (
    "Adjudicated expert reference standard",
    [f"Same {N_EXPERT} pairs, expert relabelled",
     f"Worsening {N_EXPERT_POS}/{N_EXPERT} "
     f"({100 * N_EXPERT_POS / N_EXPERT:.1f}%)"],
)

SPLIT_NOTE = "Patient-level grouped split (no patient appears in more than one partition)"


# --------------------------------------------------------------------------
# Primitives (all speak the inverted, top-down y axis)
# --------------------------------------------------------------------------


def box_height(n_lines: int) -> float:
    return n_lines * LINE_H + 2 * PAD_V


def draw_box(
    ax,
    y_top: float,
    cx: float,
    w: float,
    title: str | None,
    body: list[str],
    *,
    fs_title: float = FS_TITLE,
    fs_body: float = FS_BODY,
    face: str = FACE,
    edge: str = BORDER,
    lw: float = 0.6,
    min_lines: int = 0,
) -> float:
    """Draw a box with its text block top-anchored at ``y_top``.

    ``min_lines`` pads the box height without adding text, which is how the
    three split boxes are kept at identical height so they read as parallel.

    Returns the bottom edge y.
    """
    n_lines = max((1 if title else 0) + len(body), min_lines)
    h = box_height(n_lines)
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2.0, y_top),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={ROUND}",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            mutation_aspect=1.0,
            zorder=2,
        )
    )
    # Vertically centre the text block inside the (possibly padded) box.
    used = ((1 if title else 0) + len(body)) * LINE_H
    ty = y_top + (h - used) / 2.0 + LINE_H / 2.0
    if title:
        t = ax.text(cx, ty, title, ha="center", va="center",
                    fontsize=fs_title, fontweight="bold", color=TEXT, zorder=3)
        _TEXTS.append((t, w, title))
        ty += LINE_H
    for ln in body:
        t = ax.text(cx, ty, ln, ha="center", va="center",
                    fontsize=fs_body, color=TEXT, zorder=3)
        _TEXTS.append((t, w, ln))
        ty += LINE_H
    return y_top + h


def v_arrow(ax, x: float, y0: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x, y0), (x, y1),
            arrowstyle="-|>", mutation_scale=7.0,
            linewidth=0.8, color=ARROW,
            shrinkA=0, shrinkB=0, zorder=1,
        )
    )


def h_arrow(ax, x0: float, x1: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y),
            arrowstyle="-|>", mutation_scale=7.0,
            linewidth=0.8, color=ARROW,
            shrinkA=0, shrinkB=0, zorder=1,
        )
    )


def line(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.plot([x0, x1], [y0, y1], color=ARROW, linewidth=0.8,
            solid_capstyle="butt", zorder=1)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def total_height() -> float:
    """Sum the stack so the canvas is exactly as tall as the diagram."""
    h = 0.0
    h += box_height(2)          # source
    h += GAP_EXCL
    h += box_height(2)          # edema
    h += GAP_EXCL
    h += box_height(3)          # frontal
    h += GAP_PLAIN
    h += box_height(2)          # pairs
    h += GAP_EXCL
    h += box_height(3)          # final analytic cohort
    h += FAN_H
    h += box_height(4)          # split boxes (equalised to the tallest)
    h += GAP_PLAIN
    h += box_height(2)          # patient-unique sample
    h += GAP_PLAIN
    h += box_height(3)          # two-radiologist review
    h += GAP_PLAIN
    h += box_height(3)          # adjudicated expert reference standard
    return h + 2 * MARGIN


def build() -> plt.Figure:
    check_integrity()
    _TEXTS.clear()

    fig_h = total_height()
    fig = plt.figure(figsize=(FIG_W / 25.4, fig_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(fig_h, 0)       # inverted: larger y == lower on the page
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    def excl_step(y_from: float, gap: float, excl) -> float:
        """Vertical arrow across ``gap`` with an exclusion box branching right."""
        y_to = y_from + gap
        v_arrow(ax, TRUNK_X, y_from, y_to)
        title, body = excl
        h = box_height(1 + len(body))
        y_mid = y_from + gap / 2.0
        draw_box(ax, y_mid - h / 2.0, EXCL_X0 + EXCL_W / 2.0, EXCL_W, title, body,
                 face=FACE_EXCL, fs_title=FS_EXCL_TITLE, fs_body=FS_EXCL_BODY)
        h_arrow(ax, TRUNK_X, EXCL_X0, y_mid)
        return y_to

    def plain_step(y_from: float) -> float:
        y_to = y_from + GAP_PLAIN
        v_arrow(ax, TRUNK_X, y_from, y_to)
        return y_to

    # --- main flow ---------------------------------------------------------
    y = MARGIN
    y = draw_box(ax, y, TRUNK_X, MAIN_W, *BOX_SOURCE)
    y = excl_step(y, GAP_EXCL, EXCL_EDEMA)
    y = draw_box(ax, y, TRUNK_X, MAIN_W, *BOX_EDEMA)
    y = excl_step(y, GAP_EXCL, EXCL_FRONTAL)
    y = draw_box(ax, y, TRUNK_X, MAIN_W, *BOX_FRONTAL)
    y = plain_step(y)
    y = draw_box(ax, y, TRUNK_X, MAIN_W, *BOX_PAIRS)
    y = excl_step(y, GAP_EXCL, EXCL_CONFLICT)
    y = draw_box(ax, y, TRUNK_X, MAIN_W, *BOX_FINAL,
                 edge=BORDER_STRONG, lw=1.0, face=FACE_KEY)

    # --- three-way patient-level grouped split -----------------------------
    y_bar = y + FAN_H * FAN_BAR
    y_split_top = y + FAN_H
    line(ax, TRUNK_X, y, TRUNK_X, y_bar)
    line(ax, SPLIT_CX[0], y_bar, SPLIT_CX[-1], y_bar)
    for cx in SPLIT_CX:
        v_arrow(ax, cx, y_bar, y_split_top)
    note = ax.text(TRUNK_X, y + FAN_H * FAN_NOTE, SPLIT_NOTE,
                   ha="center", va="center", fontsize=FS_NOTE, color=TEXT,
                   zorder=3,
                   bbox=dict(boxstyle="square,pad=0.20", facecolor="white",
                             edgecolor="none"))
    _TEXTS.append((note, FIG_W - 2 * MARGIN, SPLIT_NOTE))

    bottoms = [
        draw_box(ax, y_split_top, cx, SPLIT_W, title, body,
                 fs_title=FS_SPLIT_TITLE, fs_body=FS_SPLIT_BODY, min_lines=4)
        for cx, (title, body) in zip(SPLIT_CX, SPLIT_BOXES)
    ]
    y = bottoms[2]              # expert branch descends from the held-out test box

    # --- expert-reference branch -------------------------------------------
    for content, kw in (
        (BOX_SAMPLE, {}),
        (BOX_REVIEW, {}),
        (BOX_EXPERT, dict(edge=BORDER_STRONG, lw=1.0, face=FACE_KEY)),
    ):
        y_to = y + GAP_PLAIN
        v_arrow(ax, EXPERT_CX, y, y_to)
        y = draw_box(ax, y_to, EXPERT_CX, EXPERT_W, *content, **kw)

    return fig


# --------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------


def qa_text_fit(fig) -> list[str]:
    """Measure every string against its box; return human-readable violations."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems: list[str] = []
    for artist, box_w, label in _TEXTS:
        bb = artist.get_window_extent(renderer=renderer)
        width_mm = bb.width / fig.dpi * 25.4
        allowed = box_w - 2 * PAD_H
        if width_mm > allowed:
            problems.append(
                f"  OVERFLOW {width_mm:6.1f} mm > {allowed:5.1f} mm : {label!r}"
            )
    return problems


def main() -> None:
    fig = build()

    problems = qa_text_fit(fig)
    if problems:
        print("Text-fit QA failed:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        plt.close(fig)
        sys.exit(1)
    print(f"Text-fit QA: {len(_TEXTS)} strings, all within box width "
          f"(>= {PAD_H} mm clearance each side)")

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIF_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(PNG_PATH, dpi=DPI, facecolor="white")

    # Flatten to RGB (no alpha channel) and write LZW-compressed TIFF at 300 dpi.
    with Image.open(PNG_PATH) as im:
        im.convert("RGB").save(
            TIF_PATH, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI)
        )
    plt.close(fig)

    with Image.open(TIF_PATH) as im:
        print(f"TIF   : {TIF_PATH}")
        print(f"  size : {im.size[0]} x {im.size[1]} px")
        print(f"  mode : {im.mode}")
        print(f"  dpi  : {im.info.get('dpi')}")
        print(f"  compr: {im.info.get('compression')}")
    print(f"  bytes: {TIF_PATH.stat().st_size:,}")
    print(f"PNG   : {PNG_PATH}  ({PNG_PATH.stat().st_size:,} bytes)")
    print(f"canvas: {FIG_W:.1f} x {total_height():.1f} mm")


if __name__ == "__main__":
    main()
