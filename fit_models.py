#!/usr/bin/env python3
"""Fit the three input representations and write tuning and test scores.

    python fit_models.py --pairs pairs.csv --image-root /path/to/parent/of/CheXpert-v1.0-small \
        --out scores.csv [--size 32] [--components 128]

Each image: grayscale, bilinear resize to size x size, scale to [0, 1], clip to its 1st-99th
percentile range, then subtract its mean and divide by (standard deviation + 1e-6).

Representations (flattened):
  current     current image
  difference  current - prior, |current - prior|
  temporal    current, prior, current - prior, |current - prior|

Model: PCA (randomized SVD) -> standardization -> L2 logistic regression with balanced class
weights, all fitted on the training split only (seed 42).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
KINDS = ("current", "difference", "temporal")


def preprocess(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), resample=Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip(a, lo, hi)
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32).ravel()


def features(cur: np.ndarray, prev: np.ndarray, kind: str) -> np.ndarray:
    if kind == "current":
        return cur
    diff = cur - prev
    parts = [diff, np.abs(diff)] if kind == "difference" else [cur, prev, diff, np.abs(diff)]
    return np.hstack(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="output of build_cohort.py")
    ap.add_argument("--image-root", required=True, type=Path, help="folder that contains CheXpert-v1.0-small/")
    ap.add_argument("--out", required=True, help="output score table (CSV)")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--components", type=int, default=128)
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    paths = sorted(set(pairs.current_path) | set(pairs.previous_path))
    index = {p: i for i, p in enumerate(paths)}
    images = np.stack([preprocess(args.image_root / p, args.size) for p in paths])
    print(f"preprocessed {len(paths):,} images at {args.size}x{args.size}", flush=True)

    split = {s: pairs[pairs.split == s] for s in ("train", "tune", "test")}
    cur = {s: images[d.current_path.map(index)] for s, d in split.items()}
    prev = {s: images[d.previous_path.map(index)] for s, d in split.items()}

    out = pd.concat([split["tune"], split["test"]])[["pair_id", "patient_id", "split", "target"]].copy()
    for kind in KINDS:
        model = make_pipeline(
            PCA(n_components=args.components, svd_solver="randomized", random_state=SEED),
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED),
        )
        model.fit(features(cur["train"], prev["train"], kind), split["train"].target)
        out[f"{kind}_score"] = np.concatenate(
            [model.predict_proba(features(cur[s], prev[s], kind))[:, 1] for s in ("tune", "test")])
        print(f"fitted {kind}", flush=True)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
