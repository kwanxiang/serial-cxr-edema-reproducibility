#!/usr/bin/env python3
"""Held-out evaluation of the three representations.

    python evaluate.py --scores scores.csv [--out results.csv]

Thresholds maximize Youden's J on the tuning split and are applied unchanged to the test split.
Confidence intervals: percentile bootstrap, 5000 resamples, seed 42 (fresh generator per interval);
resamples containing one outcome class are skipped. Patient-clustered intervals resample patients.
Partial AUC: unstandardized area under the ROC curve over the inclusive 0.80-1.00 sensitivity band,
integrated over the false-positive rate. Average precision is the step-wise (tie-aware) definition.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve

SEED, B = 42, 5000
KINDS = ("current", "difference", "temporal")
CONTRASTS = (("temporal", "current"), ("temporal", "difference"), ("difference", "current"))


def fmt(x: float, nd: int = 3) -> str:
    """Round once, half up."""
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{abs(v) if v == 0 else v:.{nd}f}"


def auc(y, s) -> float:
    y = np.asarray(y, bool)
    m, n = y.sum(), (~y).sum()
    return float((rankdata(s)[y].sum() - m * (m + 1) / 2) / (m * n))


def ap(y, s) -> float:
    return float(average_precision_score(y, s))


def partial_auc(y, s, lo=0.80, hi=1.00) -> float:
    order = np.argsort(-np.asarray(s), kind="mergesort")
    ys, ss = np.asarray(y)[order], np.asarray(s)[order]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]          # merge tied scores
    tpr = np.r_[0, np.cumsum(ys)[last] / ys.sum()]
    fpr = np.r_[0, np.cumsum(1 - ys)[last] / (len(ys) - ys.sum())]
    area = 0.0
    for t0, t1, f0, f1 in zip(tpr[:-1], tpr[1:], fpr[:-1], fpr[1:]):
        if f1 == f0:
            continue
        if t1 == t0:                                                     # horizontal segment
            area += (f1 - f0) * t0 if lo - 1e-12 <= t0 <= hi + 1e-12 else 0.0
            continue
        a, b = max(t0, lo), min(t1, hi)                                  # sloped segment (ties)
        if b > a:
            area += (f1 - f0) * (b - a) / (t1 - t0) * (a + b) / 2
    return area


def bootstrap(fn, y, *scores, clusters=None):
    rng = np.random.default_rng(SEED)
    if clusters is not None:
        groups, inverse = np.unique(clusters, return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(len(groups))]
    values = []
    for _ in range(B):
        if clusters is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([members[g] for g in rng.integers(0, len(groups), len(groups))])
        if y[idx].min() == y[idx].max():
            continue
        values.append(fn(y[idx], *[s[idx] for s in scores]))
    return np.percentile(values, [2.5, 97.5])


def delong(y, sa, sb):
    """Two correlated AUCs (DeLong et al., 1988); returns (difference, two-sided p)."""
    y = np.asarray(y, bool)
    m, n = y.sum(), (~y).sum()
    v10, v01, aucs = [], [], []
    for s in (sa, sb):
        r, rx, ry = rankdata(s), rankdata(s[y]), rankdata(s[~y])
        aucs.append((r[y].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((r[y] - rx) / n)
        v01.append(1 - (r[~y] - ry) / m)
    cov = np.cov(v10) / m + np.cov(v01) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return aucs[0] - aucs[1], 2 * stats.norm.sf(abs(z))


def youden_threshold(y, s) -> float:
    fpr, tpr, cut = roc_curve(y, s)
    return float(cut[np.argmax(tpr - fpr)])


def calibration(y, p, bins=10) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    X = np.c_[np.ones_like(x), x]
    beta = np.zeros(2)
    for _ in range(100):                                                 # Newton-Raphson
        mu = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve((X * (mu * (1 - mu))[:, None]).T @ X, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-12:
            break
    citl = optimize.minimize_scalar(lambda a: np.sum(np.logaddexp(0, a + x) - y * (a + x)),
                                    bounds=(-10, 10), method="bounded", options={"xatol": 1e-12}).x
    b = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    g = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("y", "size"), f=("p", "mean"), o=("y", "mean"))
    return {"mean predicted": p.mean(), "calibration slope": beta[1], "calibration-in-the-large": citl,
            "ECE": (g.n * (g.o - g.f).abs()).sum() / len(y),
            "Brier reliability (binned)": (g.n * (g.f - g.o) ** 2).sum() / len(y),
            "Brier resolution (binned)": (g.n * (g.o - y.mean()) ** 2).sum() / len(y)}


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--scores", required=True, help="output of fit_models.py")
    ap_.add_argument("--out", help="optional CSV with unrounded results")
    args = ap_.parse_args()

    scores = pd.read_csv(args.scores)
    tune, test = scores[scores.split == "tune"], scores[scores.split == "test"]
    y = test.target.to_numpy()
    S = {k: test[f"{k}_score"].to_numpy() for k in KINDS}
    rows = []

    def add(section, item, value, model=""):
        rows.append({"section": section, "item": item, "model": model, "value": float(value)})

    for k in KINDS:
        s = S[k]
        add("discrimination", "AUC", auc(y, s), k)
        lo, hi = bootstrap(auc, y, s); add("discrimination", "AUC 95% CI lower", lo, k); add("discrimination", "AUC 95% CI upper", hi, k)
        add("discrimination", "AP", ap(y, s), k)
        lo, hi = bootstrap(ap, y, s); add("discrimination", "AP 95% CI lower", lo, k); add("discrimination", "AP 95% CI upper", hi, k)
        add("discrimination", "partial AUC (0.80-1.00 sensitivity)", partial_auc(y, s), k)
        thr = youden_threshold(tune.target.to_numpy(), tune[f"{k}_score"].to_numpy())
        pred = s >= thr
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        tn, fn = int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        for item, v in (("threshold (tuning split)", thr), ("sensitivity", sens), ("specificity", spec),
                        ("PPV", tp / (tp + fp)), ("NPV", tn / (tn + fn)), ("F1", 2 * tp / (2 * tp + fp + fn)),
                        ("balanced accuracy", (sens + spec) / 2), ("Brier score", np.mean((s - y) ** 2)),
                        ("TP", tp), ("FP", fp), ("TN", tn), ("FN", fn)):
            add("operating point", item, v, k)
        for item, v in calibration(y.astype(float), s).items():
            add("calibration", item, v, k)

    for a, b in CONTRASTS:
        name = f"{a} - {b}"
        d, p = delong(y, S[a], S[b])
        add("contrast", "delta AUC", d, name)
        lo, hi = bootstrap(lambda yy, sa, sb: auc(yy, sa) - auc(yy, sb), y, S[a], S[b])
        add("contrast", "delta AUC 95% CI lower", lo, name); add("contrast", "delta AUC 95% CI upper", hi, name)
        lo, hi = bootstrap(lambda yy, sa, sb: auc(yy, sa) - auc(yy, sb), y, S[a], S[b], clusters=test.patient_id.to_numpy())
        add("contrast", "delta AUC patient-clustered 95% CI lower", lo, name)
        add("contrast", "delta AUC patient-clustered 95% CI upper", hi, name)
        add("contrast", "DeLong p", p, name)
        add("contrast", "delta AP", ap(y, S[a]) - ap(y, S[b]), name)
        lo, hi = bootstrap(lambda yy, sa, sb: ap(yy, sa) - ap(yy, sb), y, S[a], S[b])
        add("contrast", "delta AP 95% CI lower", lo, name); add("contrast", "delta AP 95% CI upper", hi, name)

    res = pd.DataFrame(rows)
    if args.out:
        res.to_csv(args.out, index=False)
    print(f"Test split: {len(y):,} pairs, {int(y.sum()):,} worsening ({100 * y.mean():.1f}%)\n")
    for section, block in res.groupby("section", sort=False):
        table = (block.pivot(index="item", columns="model", values="value")
                 .reindex(index=block.item.unique(), columns=block.model.unique()))
        shown = table.copy().astype(object)
        for item, row in table.iterrows():
            for model, v in row.items():
                shown.loc[item, model] = (f"{v:.2g}" if item == "DeLong p"
                                          else f"{int(v)}" if v == int(v) and abs(v) > 1 else fmt(v))
        print(f"[{section}]\n{shown.to_string()}\n")


if __name__ == "__main__":
    main()
