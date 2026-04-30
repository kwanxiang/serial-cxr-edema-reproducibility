#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "derived"
FIG_DIR = ROOT / "figures"

SCORES = {
    "current_only_pixels": ("current_only_pixels_score", "Current-only"),
    "difference_only_pixels": ("difference_only_pixels_score", "Difference-only"),
    "prior_current_temporal_pixels": ("prior_current_temporal_pixels_score", "Temporal"),
}


def plot_roc(held: pd.DataFrame, expert: pd.DataFrame) -> None:
    panels = [
        ("A", "Report-label held-out cohort", held, "report_target"),
        ("B", "Expert-reference subset", expert, "expert_target"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 5.1), dpi=300)
    for ax, (panel, title, df, target_col) in zip(axes, panels):
        y = df[target_col].astype(int)
        for _, (score_col, label) in SCORES.items():
            scores = df[score_col].astype(float)
            fpr, tpr, _ = roc_curve(y, scores)
            model_auc = roc_auc_score(y, scores)
            ax.plot(fpr, tpr, linewidth=2.0, label=f"{label} (AUC {model_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.55")
        ax.set_title(f"{panel}. {title}", fontweight="bold", loc="left")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_pr(held: pd.DataFrame, expert: pd.DataFrame) -> None:
    panels = [
        ("A", "Report-label held-out cohort", held, "report_target"),
        ("B", "Expert-reference subset", expert, "expert_target"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 5.1), dpi=300)
    for ax, (panel, title, df, target_col) in zip(axes, panels):
        y = df[target_col].astype(int)
        prevalence = y.mean()
        for _, (score_col, label) in SCORES.items():
            scores = df[score_col].astype(float)
            precision, recall, _ = precision_recall_curve(y, scores)
            ap = average_precision_score(y, scores)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, linewidth=2.0, label=f"{label} (AP {ap:.3f})")
        ax.axhline(prevalence, linestyle="--", linewidth=1.0, color="0.55", label=f"Prevalence {prevalence:.3f}")
        ax.set_title(f"{panel}. {title}", fontweight="bold", loc="left")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curves.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    held = pd.read_csv(DATA_DIR / "held_out_predictions_sanitized.csv")
    expert = pd.read_csv(DATA_DIR / "expert_reference_cases_sanitized.csv")
    plot_roc(held, expert)
    plot_pr(held, expert)
    print(f"Wrote {FIG_DIR / 'roc_curves.png'}")
    print(f"Wrote {FIG_DIR / 'pr_curves.png'}")


if __name__ == "__main__":
    main()
