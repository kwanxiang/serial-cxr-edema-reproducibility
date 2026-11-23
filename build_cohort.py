#!/usr/bin/env python3
"""Build the pair-level pulmonary edema worsening cohort and the patient-level split.

    python build_cohort.py --chextemporal chextemporal_chexpert_train.csv --out pairs.csv \
        [--chexpert-csv CheXpert-v1.0-small/train.csv]

Rules:
  * keep CheXTemporal rows whose Pathology field or Sentence contains "edema";
  * take the first frontal image listed for the current and for the prior study;
  * aggregate rows to image pairs; New/Worse are worsening votes, Stable/Better/Resolve are
    non-worsening votes; pairs with both kinds of vote are removed as conflicting;
  * split by patient: 20% test, then 12.5% of the remainder for tuning (seed 42).
"""
from __future__ import annotations

import argparse
import ast

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PREFIX = "CheXpert-v1.0-small/"
WORSE, NOT_WORSE = {"New", "Worse"}, {"Stable", "Better", "Resolve"}
SEED = 42


def first_frontal(cell) -> str | None:
    try:
        paths = ast.literal_eval(cell) if isinstance(cell, str) else []
    except (ValueError, SyntaxError):
        return None
    frontal = [p for p in paths if isinstance(p, str) and "frontal" in p.lower()]
    return PREFIX + frontal[0] if frontal else None


def build_pairs(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    flow = {"raw rows": len(rows)}
    edema = (rows["Pathology"].fillna("").str.contains("edema", case=False)
             | rows["Sentence"].fillna("").str.contains("edema", case=False))
    rows = rows[edema].copy()
    flow["edema-related rows"] = len(rows)
    rows["current_path"] = rows["image_paths"].map(first_frontal)
    rows["previous_path"] = rows["previous_images"].map(first_frontal)
    rows = rows.dropna(subset=["current_path", "previous_path"])
    flow["rows with serial frontal pairs"] = len(rows)
    rows["pair_id"] = rows["previous_path"] + ">>" + rows["current_path"]
    pairs = rows.groupby("pair_id").agg(
        patient_id=("patient_id", "first"),
        current_path=("current_path", "first"),
        previous_path=("previous_path", "first"),
        label_set=("Label", lambda s: "|".join(sorted(set(s)))),
        worse=("Label", lambda s: s.isin(WORSE).sum()),
        not_worse=("Label", lambda s: s.isin(NOT_WORSE).sum()),
    ).reset_index()
    flow["unique pairs"] = len(pairs)
    conflict = (pairs.worse > 0) & (pairs.not_worse > 0)
    flow["conflicting pairs removed"] = int(conflict.sum())
    pairs = pairs[~conflict].copy()
    pairs["target"] = (pairs.worse > 0).astype(int)
    flow["final pairs"] = len(pairs)
    flow["patients"] = pairs.patient_id.nunique()
    return pairs.drop(columns=["worse", "not_worse"]), flow


def split_by_patient(pairs: pd.DataFrame) -> pd.DataFrame:
    rest, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(pairs, groups=pairs.patient_id))
    rest_pairs = pairs.iloc[rest]
    train, tune = next(GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
                       .split(rest_pairs, groups=rest_pairs.patient_id))
    pairs = pairs.copy()
    pairs["split"] = "test"
    pairs.loc[rest_pairs.index[train], "split"] = "train"
    pairs.loc[rest_pairs.index[tune], "split"] = "tune"
    return pairs


def describe(pairs: pd.DataFrame, chexpert_csv: str | None) -> pd.DataFrame:
    groups = {"all": pairs, **{s: pairs[pairs.split == s] for s in ("train", "tune", "test")}}
    table = pd.DataFrame({name: {"pairs": len(g), "worsening": int(g.target.sum()),
                                 "worsening %": round(100 * g.target.mean(), 1),
                                 "patients": g.patient_id.nunique()} for name, g in groups.items()})
    if chexpert_csv:
        meta = pd.read_csv(chexpert_csv).set_index("Path")
        for name, g in groups.items():
            m = meta.loc[g.current_path]
            table.loc["age mean", name] = round(m["Age"].mean(), 1)
            table.loc["age SD", name] = round(m["Age"].std(ddof=1), 1)
            table.loc["female %", name] = round(100 * (m["Sex"] == "Female").mean(), 1)
            table.loc["AP %", name] = round(100 * (m["AP/PA"] == "AP").mean(), 1)
    table = table.astype(object)
    for row in ("pairs", "worsening", "patients"):
        table.loc[row] = table.loc[row].map(lambda v: f"{int(v):,}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chextemporal", required=True, help="chextemporal_chexpert_train.csv")
    ap.add_argument("--out", required=True, help="output pair table (CSV)")
    ap.add_argument("--chexpert-csv", help="CheXpert-v1.0-small/train.csv, for age, sex and projection")
    args = ap.parse_args()

    pairs, flow = build_pairs(pd.read_csv(args.chextemporal))
    pairs = split_by_patient(pairs)
    pairs.to_csv(args.out, index=False)
    for step, n in flow.items():
        print(f"{step:32s}{n:>9,}")
    print()
    print(describe(pairs, args.chexpert_csv).to_string())


if __name__ == "__main__":
    main()
