"""
Reproduce the statistical results of
"Serial Images Improve AI Detection of Pulmonary Edema Worsening:
 An Expert-Adjudicated Benchmark" (INSI-D-26-00883)
from the locked, matched-dimensionality (PCA = 128) per-pair model scores.

This entry-point script recomputes the primary AUC/AP point estimates and paired
AUC contrasts from the locked per-pair scores in ./data. Additional scripts in
./scripts recompute calibration, operating-point, reader-agreement, clustered-
bootstrap, and sensitivity-reference results. Source-image model fitting is not
included because the underlying images are governed by separate data-use terms.

Usage:  python reproduce_statistics.py
Requires: numpy, pandas, scipy  (see requirements.txt)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

HERE = Path(__file__).resolve().parent
SCORES = HERE / "data" / "unified128_test_scores.csv"
CASES = HERE / "data" / "final_adjudicated_cases.csv"

SEED = 42
B = 5000
N_PERM = 20000
KINDS = ["current", "difference", "temporal"]
NICE = {"current": "Current-only", "difference": "Difference-only", "temporal": "Temporal"}


def fast_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    p, n = int(y.sum()), int(len(y) - y.sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - p * (p + 1) / 2.0) / (p * n)


def fast_ap(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    p = int(y.sum())
    if p == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    sy = y[order]; tp = np.cumsum(sy); ranks = np.arange(1, len(sy) + 1)
    return float(((tp / ranks) * sy).sum() / p)


def boot_ci(fn, y, s, seed=SEED):
    rng = np.random.default_rng(seed); vals = []; n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(fn(yy, s[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def paired_boot(fn, y, sa, sb, seed=SEED):
    obs = fn(y, sa) - fn(y, sb)
    rng = np.random.default_rng(seed); vals = []; n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(fn(yy, sa[idx]) - fn(yy, sb[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def delong(y, sa, sb):
    y = np.asarray(y, int); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    m, n = len(pos), len(neg); idx = np.concatenate([pos, neg])
    preds = np.vstack([np.asarray(sa, float)[idx], np.asarray(sb, float)[idx]])
    tx = np.empty([2, m]); ty = np.empty([2, n]); tz = np.empty([2, m + n])
    for r in range(2):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r, :])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float("nan")
    return float(2 * stats.norm.sf(abs((aucs[0] - aucs[1]) / np.sqrt(var))))


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items); adj = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p); adj[k] = min(running, 1.0)
    return adj


def main() -> None:
    scores = pd.read_csv(SCORES)
    cases = pd.read_csv(CASES)
    sub = cases[["pair_key", "final_target", "target"]].merge(
        scores[["pair_key"] + [f"{k}_score" for k in KINDS]], on="pair_key", validate="one_to_one")

    refs = {
        "R1 report-derived held-out test (n=5,341)":
            (scores["target"].to_numpy(int), {k: scores[f"{k}_score"].to_numpy(float) for k in KINDS}),
        "R2 same 500 under report labels":
            (sub["target"].to_numpy(int), {k: sub[f"{k}_score"].to_numpy(float) for k in KINDS}),
        "R3 adjudicated expert reference (n=500)":
            (sub["final_target"].to_numpy(int), {k: sub[f"{k}_score"].to_numpy(float) for k in KINDS}),
    }

    print("=" * 72)
    print("POINT ESTIMATES (matched PCA = 128)")
    print("=" * 72)
    for name, (y, S) in refs.items():
        print(f"\n{name}   prevalence={y.mean():.3f}")
        for k in KINDS:
            auc = fast_auc(y, S[k]); lo, hi = boot_ci(fast_auc, y, S[k])
            ap = fast_ap(y, S[k])
            print(f"  {NICE[k]:16s} AUC={auc:.3f} ({lo:.3f}-{hi:.3f})  AP={ap:.3f}")

    print("\n" + "=" * 72)
    print("BETWEEN-REPRESENTATION AUC CONTRASTS + Holm across 9 tests")
    print("=" * 72)
    praw = {}
    contrasts = [("temporal", "difference"), ("temporal", "current"), ("difference", "current")]
    store = []
    for name, (y, S) in refs.items():
        for a, b in contrasts:
            d, lo, hi = paired_boot(fast_auc, y, S[a], S[b])
            p = delong(y, S[a], S[b])
            key = f"{name} :: {a}-{b}"
            praw[key] = p
            store.append((key, d, lo, hi, p))
    adj = holm(praw)
    for key, d, lo, hi, p in store:
        print(f"  {key:56s} dAUC={d:+.3f} ({lo:+.3f},{hi:+.3f}) DeLong p={p:.3g} Holm={adj[key]:.3g}")

    print("\n" + "=" * 72)
    print("KEY RESULT CHECK")
    print("=" * 72)
    y1, S1 = refs["R1 report-derived held-out test (n=5,341)"]
    y3, S3 = refs["R3 adjudicated expert reference (n=500)"]
    tc = fast_auc(y1, S1["temporal"]) - fast_auc(y1, S1["current"])
    td3 = fast_auc(y3, S3["temporal"]) - fast_auc(y3, S3["difference"])
    print(f"  temporal - current (report, n=5,341) = {tc:+.3f}   [manuscript +0.084, established]")
    print(f"  temporal - difference (expert, n=500) = {td3:+.3f}   [manuscript +0.017, NOT established]")
    print("\nReproduction of statistical results complete.")


if __name__ == "__main__":
    main()
