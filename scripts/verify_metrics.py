#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "derived"
REPORT_PATH = DATA_DIR / "held_out_predictions_sanitized.csv"
EXPERT_PATH = DATA_DIR / "expert_reference_cases_sanitized.csv"
REPORT_METRICS_PATH = DATA_DIR / "report_label_model_metrics.csv"
EXPERT_METRICS_PATH = DATA_DIR / "expert_reference_model_metrics.csv"
OUT_PATH = ROOT / "docs" / "verification_report.json"

SCORE_COLUMNS = {
    "current_only_pixels": "current_only_pixels_score",
    "difference_only_pixels": "difference_only_pixels_score",
    "prior_current_temporal_pixels": "prior_current_temporal_pixels_score",
}


def compute_metrics(df: pd.DataFrame, target_col: str, metrics_df: pd.DataFrame) -> list[dict[str, object]]:
    y = df[target_col].astype(int).to_numpy()
    out: list[dict[str, object]] = []
    metrics_by_model = metrics_df.set_index("model")
    for model, score_col in SCORE_COLUMNS.items():
        s = df[score_col].astype(float).to_numpy()
        threshold = float(metrics_by_model.loc[model, "threshold"])
        y_hat = (s >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()
        observed = {
            "auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
            "brier": float(brier_score_loss(y, s)),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        expected = {
            "auc": float(metrics_by_model.loc[model, "auc"]),
            "ap": float(metrics_by_model.loc[model, "ap"]),
            "brier": float(metrics_by_model.loc[model, "brier"]),
            "tp": int(metrics_by_model.loc[model, "tp"]),
            "fp": int(metrics_by_model.loc[model, "fp"]),
            "tn": int(metrics_by_model.loc[model, "tn"]),
            "fn": int(metrics_by_model.loc[model, "fn"]),
        }
        diffs = {
            key: float(abs(observed[key] - expected[key]))
            for key in ["auc", "ap", "brier"]
        }
        counts_match = all(observed[key] == expected[key] for key in ["tp", "fp", "tn", "fn"])
        continuous_match = all(value <= 5e-6 for value in diffs.values())
        out.append(
            {
                "model": model,
                "observed": observed,
                "expected": expected,
                "absolute_differences": diffs,
                "counts_match": counts_match,
                "continuous_metrics_match": continuous_match,
            }
        )
    return out


def main() -> None:
    held = pd.read_csv(REPORT_PATH)
    expert = pd.read_csv(EXPERT_PATH)
    report_metrics = pd.read_csv(REPORT_METRICS_PATH)
    expert_metrics = pd.read_csv(EXPERT_METRICS_PATH)

    report_checks = compute_metrics(held, "report_target", report_metrics)
    expert_checks = compute_metrics(expert, "expert_target", expert_metrics)

    crosstab = pd.crosstab(expert["report_target"].astype(int), expert["expert_target"].astype(int))
    crosstab = crosstab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    result = {
        "report_label_cohort": {
            "n": int(len(held)),
            "positive_n": int(held["report_target"].sum()),
            "checks": report_checks,
        },
        "expert_reference_subset": {
            "n": int(len(expert)),
            "positive_n": int(expert["expert_target"].sum()),
            "report_positive_n": int(expert["report_target"].sum()),
            "report_vs_expert_crosstab": {
                "report_0_expert_0": int(crosstab.loc[0, 0]),
                "report_0_expert_1": int(crosstab.loc[0, 1]),
                "report_1_expert_0": int(crosstab.loc[1, 0]),
                "report_1_expert_1": int(crosstab.loc[1, 1]),
            },
            "checks": expert_checks,
        },
    }
    all_ok = all(
        item["counts_match"] and item["continuous_metrics_match"]
        for section in [report_checks, expert_checks]
        for item in section
    )
    result["all_checks_passed"] = bool(all_ok)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all_ok:
        raise SystemExit("Verification failed.")


if __name__ == "__main__":
    main()
