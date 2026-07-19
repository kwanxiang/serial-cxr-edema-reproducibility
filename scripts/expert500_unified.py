"""
Expert-500 analysis at MATCHED PCA=128 for INSI-D-26-00883.

Reuses the exact DeLong / permutation / Holm / calibration implementations from
run_revision_battle_analyses.py so results stay directly comparable to the
published (mismatched-capacity) numbers.

Reference settings (Holm family = 3 contrasts x 3 references = 9 AUC tests):
  R1  report-label held-out test  (n=5,341)
  R2  report-label same-500       (n=500, report labels)
  R3  expert adjudicated 500      (n=500, final_target)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CASES = DATA / "final_adjudicated_cases.csv"
SCORES = DATA / "unified128_test_scores.csv"

SEED = 42
B = 5000
N_PERM = 20000

KINDS = ["current", "difference", "temporal"]
NICE = {"current": "Current-only", "difference": "Difference-only", "temporal": "Temporal"}


# ---------- metrics (verbatim from run_revision_battle_analyses.py) ----------
def fast_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    np_, nn = int(y.sum()), int(len(y) - y.sum())
    if np_ == 0 or nn == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def fast_ap(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    np_ = int(y.sum())
    if np_ == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    sy = y[order]; tp = np.cumsum(sy); ranks = np.arange(1, len(sy) + 1)
    return float(((tp / ranks) * sy).sum() / np_)


def boot_ci(fn, y, s, B, seed=SEED):
    rng = np.random.default_rng(seed); vals = []; n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(fn(yy, s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_boot_diff(fn, y, sa, sb, B, seed=SEED, clusters=None):
    obs = fn(y, sa) - fn(y, sb)
    rng = np.random.default_rng(seed); vals = []; n = len(y)
    if clusters is None:
        for _ in range(B):
            idx = rng.integers(0, n, n); yy = y[idx]
            if np.unique(yy).size < 2:
                continue
            vals.append(fn(yy, sa[idx]) - fn(yy, sb[idx]))
    else:
        uniq = np.unique(clusters)
        grp = {c: np.where(clusters == c)[0] for c in uniq}
        for _ in range(B):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([grp[uniq[k]] for k in pick])
            yy = y[idx]
            if np.unique(yy).size < 2:
                continue
            vals.append(fn(yy, sa[idx]) - fn(yy, sb[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(y, sa, sb):
    y = np.asarray(y, int)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    m = len(pos); n = len(neg)
    idx = np.concatenate([pos, neg])
    preds = np.vstack([np.asarray(sa, float)[idx], np.asarray(sb, float)[idx]])
    k = 2
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def perm_auc_diff(y, sa, sb, B=N_PERM, seed=SEED):
    y = np.asarray(y, int); sa = np.asarray(sa, float); sb = np.asarray(sb, float)
    obs = abs(fast_auc(y, sa) - fast_auc(y, sb))
    rng = np.random.default_rng(seed); n = len(y); cnt = 0
    A = np.vstack([sa, sb])
    for _ in range(B):
        swap = rng.integers(0, 2, n).astype(bool)
        pa = np.where(swap, A[1], A[0]); pb = np.where(swap, A[0], A[1])
        if abs(fast_auc(y, pa) - fast_auc(y, pb)) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (B + 1)


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items); adj = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        a = (m - i) * p
        running = max(running, a)
        adj[k] = min(running, 1.0)
    return adj


CONTRASTS = [
    ("temporal_minus_difference", "temporal", "difference"),
    ("temporal_minus_current", "temporal", "current"),
    ("difference_minus_current", "difference", "current"),
]


def main() -> None:
    scores = pd.read_csv(SCORES)
    cases = pd.read_csv(CASES)

    # Join unified-128 scores onto the expert-500 by the random repository key.
    sub = cases[["pair_key", "final_target", "target"]].merge(
        scores[["pair_key", "patient_cluster_id"] + [f"{k}_score" for k in KINDS]],
        on="pair_key", how="left", validate="one_to_one")
    missing = sub[[f"{k}_score" for k in KINDS]].isna().any(axis=1).sum()
    print(f"[join] expert cases={len(cases)}  matched={len(sub) - missing}  missing={missing}")
    if missing:
        raise SystemExit("expert-500 pairs missing unified-128 scores; aborting.")

    refs = {
        "R1_report_full_test": (scores["target"].to_numpy(int),
                                {k: scores[f"{k}_score"].to_numpy(float) for k in KINDS},
                                scores["patient_cluster_id"].to_numpy()),
        "R2_report_same500": (sub["target"].to_numpy(int),
                              {k: sub[f"{k}_score"].to_numpy(float) for k in KINDS},
                              None),
        "R3_expert_500": (sub["final_target"].to_numpy(int),
                          {k: sub[f"{k}_score"].to_numpy(float) for k in KINDS},
                          None),
    }

    rows = []
    print("\n=== POINT ESTIMATES (unified PCA=128) ===")
    for rname, (y, S, _) in refs.items():
        print(f"\n{rname}  n={len(y)}  prevalence={y.mean():.3f}  positives={int(y.sum())}")
        for k in KINDS:
            auc = fast_auc(y, S[k]); ap = fast_ap(y, S[k])
            alo, ahi = boot_ci(fast_auc, y, S[k], B)
            plo, phi = boot_ci(fast_ap, y, S[k], B)
            rows.append({"reference": rname, "model": NICE[k], "n": len(y),
                         "prevalence": round(float(y.mean()), 4),
                         "auc": round(auc, 4), "auc_lo": round(alo, 4), "auc_hi": round(ahi, 4),
                         "ap": round(ap, 4), "ap_lo": round(plo, 4), "ap_hi": round(phi, 4)})
            print(f"  {NICE[k]:16s} AUC={auc:.4f} ({alo:.4f}-{ahi:.4f})  "
                  f"AP={ap:.4f} ({plo:.4f}-{phi:.4f})")
    pd.DataFrame(rows).to_csv(OUT / "unified128_point_estimates.csv", index=False)

    # ---- contrasts ----
    crows = []
    praw = {}
    print("\n=== CONTRASTS (unified PCA=128) ===")
    for rname, (y, S, clus) in refs.items():
        print(f"\n{rname}")
        for cname, a, b in CONTRASTS:
            d_auc, lo, hi, = paired_boot_diff(fast_auc, y, S[a], S[b], B)
            d_ap, aplo, aphi = paired_boot_diff(fast_ap, y, S[a], S[b], B)
            _, _, z, p = delong(y, S[a], S[b])
            pp = perm_auc_diff(y, S[a], S[b])
            key = f"{rname}::{cname}"
            praw[key] = p
            crows.append({"reference": rname, "contrast": cname,
                          "d_auc": round(d_auc, 4), "d_auc_lo": round(lo, 4), "d_auc_hi": round(hi, 4),
                          "d_ap": round(d_ap, 4), "d_ap_lo": round(aplo, 4), "d_ap_hi": round(aphi, 4),
                          "delong_z": round(z, 4), "delong_p": p, "perm_p": pp})
            print(f"  {cname:28s} dAUC={d_auc:+.4f} ({lo:+.4f},{hi:+.4f})  "
                  f"dAP={d_ap:+.4f} ({aplo:+.4f},{aphi:+.4f})  "
                  f"DeLong p={p:.4g}  perm p={pp:.4g}")

    adj = holm(praw)
    for r in crows:
        r["holm_p"] = adj[f"{r['reference']}::{r['contrast']}"]
    pd.DataFrame(crows).to_csv(OUT / "unified128_contrasts.csv", index=False)

    print("\n=== HOLM-ADJUSTED (family of 9 AUC contrasts) ===")
    for r in crows:
        print(f"  {r['reference']:22s} {r['contrast']:28s} "
              f"raw={r['delong_p']:.4g}  holm={r['holm_p']:.4g}")

    # ---- patient-clustered bootstrap on full test ----
    print("\n=== PATIENT-CLUSTERED BOOTSTRAP (R1) ===")
    y1, S1, clus1 = refs["R1_report_full_test"]
    crows2 = []
    for cname, a, b in CONTRASTS:
        d, lo, hi = paired_boot_diff(fast_auc, y1, S1[a], S1[b], B, clusters=clus1)
        crows2.append({"contrast": cname, "d_auc": round(d, 4),
                       "lo_patient_clustered": round(lo, 4), "hi_patient_clustered": round(hi, 4)})
        print(f"  {cname:28s} dAUC={d:+.4f} ({lo:+.4f},{hi:+.4f})")
    pd.DataFrame(crows2).to_csv(OUT / "unified128_clustered_bootstrap.csv", index=False)

    # ---- tipping point on expert reference ----
    print("\n=== TIPPING POINT (expert reference, temporal vs difference) ===")
    y3 = sub["final_target"].to_numpy(int).copy()
    yrep = sub["target"].to_numpy(int)
    disc = np.where(y3 != yrep)[0]
    infl = np.argsort(-np.abs(sub["temporal_score"].to_numpy(float)
                              - sub["difference_score"].to_numpy(float)))
    order = [i for i in infl if i in set(disc.tolist())]
    print(f"  discordant pairs: {len(disc)}")
    ytmp = y3.copy()
    temporal = sub["temporal_score"].to_numpy(float)
    difference = sub["difference_score"].to_numpy(float)
    d, lo, hi = paired_boot_diff(fast_auc, ytmp, temporal, difference, 2000)
    base_crosses_zero = bool(lo <= 0 <= hi)
    tip = 0 if base_crosses_zero else None
    tiprows = [{"n_reverted": 0, "d_auc": round(d, 4),
                "lo": round(lo, 4), "hi": round(hi, 4),
                "ci_crosses_zero": base_crosses_zero}]
    print(f"  reverted={0:2d}  dAUC={d:+.4f} ({lo:+.4f},{hi:+.4f})"
          f"{'  <-- base CI crosses zero' if base_crosses_zero else ''}")

    # A tipping-point calculation is informative only when the starting
    # contrast is statistically established. Here the matched-128 base CI
    # already includes zero, so no label reversions are needed to cross zero.
    if not base_crosses_zero:
        for n_rev, i in enumerate(order, start=1):
            ytmp[i] = yrep[i]
            d, lo, hi = paired_boot_diff(fast_auc, ytmp, temporal, difference, 2000)
            crosses_zero = bool(lo <= 0 <= hi)
            tiprows.append({"n_reverted": n_rev, "d_auc": round(d, 4),
                            "lo": round(lo, 4), "hi": round(hi, 4),
                            "ci_crosses_zero": crosses_zero})
            print(f"  reverted={n_rev:2d}  dAUC={d:+.4f} ({lo:+.4f},{hi:+.4f})"
                  f"{'  <-- CI crosses zero' if crosses_zero else ''}")
            if crosses_zero:
                tip = n_rev
                break
    pd.DataFrame(tiprows).to_csv(OUT / "unified128_tipping_point.csv", index=False)

    (OUT / "unified128_headline.json").write_text(json.dumps({
        "spec": "matched PCA=128 for all three representations",
        "tipping_point_n_reverted": tip,
        "tipping_point_informative": not base_crosses_zero,
        "base_ci_crosses_zero": base_crosses_zero,
        "n_discordant": int(len(disc)),
    }, indent=2) + "\n", encoding="utf8")

    if base_crosses_zero:
        print("\n[tipping point] Base CI already includes zero; label-reversion tipping point is not informative")
    elif tip is None:
        print("\n[tipping point] CI did not cross zero within the tested discordant labels")
    else:
        print(f"\n[tipping point] CI first crosses zero after reverting {tip} label(s)")
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
