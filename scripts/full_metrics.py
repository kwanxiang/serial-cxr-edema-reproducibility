"""
Complete metric set at the chosen primary spec (matched PCA=128) for INSI-D-26-00883.

Produces the complete metric set for Tables 3/4/5 and Supplementary S1/S3,
plus the corrected tipping-point check:
  - point estimates + bootstrap CIs (AUC, AP)
  - operating-point metrics at the tune-locked Youden threshold
  - partial AUC (0.80-1.00 sensitivity band)
  - calibration (slope, intercept, CITL, ECE, Brier decomposition) + reliability bins
  - expert-reader agreement / adjudication summary
  - report-vs-expert label crosstab
  - corrected tipping-point (evaluates n=0 first)
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
SUMMARY = OUT / "unified128_summary.json"

SEED = 42
B = 5000
KINDS = ["current", "difference", "temporal"]
NICE = {"current": "Current-only", "difference": "Difference-only", "temporal": "Temporal"}


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


def partial_auc(y, s, lo=0.80, hi=1.00):
    """Non-standardized pAUC over a TPR band, integrated wrt FPR (as in the published code)."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    P = ys.sum(); N = len(ys) - P
    if P == 0 or N == 0:
        return float("nan")
    tpr = np.concatenate([[0], np.cumsum(ys) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - ys) / N])
    area = 0.0
    for i in range(1, len(tpr)):
        t0, t1 = tpr[i - 1], tpr[i]
        f0, f1 = fpr[i - 1], fpr[i]
        if t1 <= lo or t0 >= hi:
            continue
        df = f1 - f0
        if df == 0:
            continue
        tt = max(t0, t1)
        if lo <= tt <= hi or (t0 < lo < t1) or (t0 < hi < t1):
            area += df * tt
    return float(area)


def boot_ci(fn, y, s, B, seed=SEED):
    rng = np.random.default_rng(seed); vals = []; n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(fn(yy, s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_boot_diff(fn, y, sa, sb, B, seed=SEED):
    obs = fn(y, sa) - fn(y, sb)
    rng = np.random.default_rng(seed); vals = []; n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(fn(yy, sa[idx]) - fn(yy, sb[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def op_metrics(y, s, thr):
    y = np.asarray(y, int); s = np.asarray(s, float)
    yp = (s >= thr).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    tn = int(((yp == 0) & (y == 0)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return dict(threshold=round(thr, 4), tp=tp, fp=fp, tn=tn, fn=fn,
                sensitivity=round(sens, 4), specificity=round(spec, 4),
                ppv=round(ppv, 4), npv=round(npv, 4), f1=round(f1, 4),
                balanced_accuracy=round(0.5 * (sens + spec), 4))


def calibration(y, p, bins=10):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0]); inter = float(lr.intercept_[0])

    def nll(a):
        z = a + logit; q = 1 / (1 + np.exp(-z))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    citl = float(optimize.minimize_scalar(nll, bounds=(-10, 10), method="bounded").x)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    obar = y.mean(); N = len(y); ece = 0.0; rel = 0.0; res = 0.0
    rows = []
    for b in range(bins):
        m = idx == b; nb = int(m.sum())
        if nb == 0:
            continue
        fb = p[m].mean(); ob = y[m].mean()
        ece += nb / N * abs(ob - fb)
        rel += nb * (fb - ob) ** 2
        res += nb * (ob - obar) ** 2
        rows.append({"bin": b, "n": nb, "mean_pred": round(float(fb), 4),
                     "obs_freq": round(float(ob), 4)})
    rel /= N; res /= N
    return {
        "mean_pred": round(float(p.mean()), 4), "prevalence": round(float(obar), 4),
        "cal_slope": round(slope, 4), "cal_intercept": round(inter, 4),
        "calibration_in_the_large": round(citl, 4),
        "ECE": round(float(ece), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
        "brier_reliability": round(float(rel), 4), "brier_resolution": round(float(res), 4),
        "brier_uncertainty": round(float(obar * (1 - obar)), 4),
    }, rows


def cohen_kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    po = float((a == b).mean())
    cats = np.unique(np.concatenate([a, b]))
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def quad_weighted_kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    cats = np.unique(np.concatenate([a, b])); k = len(cats)
    ci = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y_ in zip(a, b):
        O[ci[x], ci[y_]] += 1
    O /= O.sum()
    ha = np.array([float((a == c).mean()) for c in cats])
    hb = np.array([float((b == c).mean()) for c in cats])
    E = np.outer(ha, hb)
    W = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    return 1 - (W * O).sum() / (W * E).sum()


def boot_stat_ci(fn, *arrays, B=B, seed=SEED):
    rng = np.random.default_rng(seed); n = len(arrays[0]); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    scores = pd.read_csv(SCORES)
    cases = pd.read_csv(CASES)
    thr = json.loads(SUMMARY.read_text())["thresholds"]

    sub = cases.merge(scores[["pair_key"] + [f"{k}_score" for k in KINDS]],
                      on="pair_key", how="left", validate="one_to_one")

    refs = {
        "R1_report_full_test": (scores["target"].to_numpy(int),
                                {k: scores[f"{k}_score"].to_numpy(float) for k in KINDS}),
        "R2_report_same500": (sub["target"].to_numpy(int),
                              {k: sub[f"{k}_score"].to_numpy(float) for k in KINDS}),
        "R3_expert_500": (sub["final_target"].to_numpy(int),
                          {k: sub[f"{k}_score"].to_numpy(float) for k in KINDS}),
    }

    # ---------- Tables 3 / 5 : full metric rows ----------
    rows = []
    for rname, (y, S) in refs.items():
        for k in KINDS:
            auc = fast_auc(y, S[k]); ap = fast_ap(y, S[k])
            alo, ahi = boot_ci(fast_auc, y, S[k], B)
            plo, phi = boot_ci(fast_ap, y, S[k], B)
            op = op_metrics(y, S[k], thr[k])
            pa = partial_auc(y, S[k])
            rows.append({"reference": rname, "model": NICE[k], "pca": 128,
                         "n": len(y), "positives": int(y.sum()),
                         "prevalence": round(float(y.mean()), 4),
                         "auc": round(auc, 4), "auc_lo": round(alo, 4), "auc_hi": round(ahi, 4),
                         "ap": round(ap, 4), "ap_lo": round(plo, 4), "ap_hi": round(phi, 4),
                         "partial_auc_80_100": round(pa, 4),
                         "brier": round(float(np.mean((S[k] - y) ** 2)), 4), **op})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "FINAL_table3_table5_metrics.csv", index=False)
    print("=== TABLE 3 / 5 METRICS (matched PCA=128) ===")
    print(tbl[["reference", "model", "auc", "auc_lo", "auc_hi", "ap", "sensitivity",
               "specificity", "ppv", "balanced_accuracy", "partial_auc_80_100"]].to_string(index=False))

    # ---------- calibration ----------
    crows, brows = [], []
    for rname, (y, S) in refs.items():
        for k in KINDS:
            c, bins = calibration(y, S[k])
            crows.append({"reference": rname, "model": NICE[k], **c})
            for b in bins:
                brows.append({"reference": rname, "model": NICE[k], **b})
    pd.DataFrame(crows).to_csv(OUT / "FINAL_calibration.csv", index=False)
    pd.DataFrame(brows).to_csv(OUT / "FINAL_calibration_bins.csv", index=False)
    print("\n=== CALIBRATION (S3) ===")
    print(pd.DataFrame(crows)[["reference", "model", "mean_pred", "prevalence", "cal_slope",
                               "calibration_in_the_large", "ECE", "brier"]].to_string(index=False))

    # ---------- expert reader agreement (Table 4) ----------
    r1t = cases["reader1_target"].to_numpy(int)
    r2t = cases["reader2_target"].to_numpy(int)
    agree = float((r1t == r2t).mean())
    alo, ahi = boot_stat_ci(lambda a, b: float((a == b).mean()), r1t, r2t)
    kap = cohen_kappa(r1t, r2t)
    klo, khi = boot_stat_ci(cohen_kappa, r1t, r2t)
    qp = quad_weighted_kappa(cases["reader1_prior_severity"].to_numpy(int),
                             cases["reader2_prior_severity"].to_numpy(int))
    qplo, qphi = boot_stat_ci(quad_weighted_kappa,
                              cases["reader1_prior_severity"].to_numpy(int),
                              cases["reader2_prior_severity"].to_numpy(int))
    qc = quad_weighted_kappa(cases["reader1_current_severity"].to_numpy(int),
                             cases["reader2_current_severity"].to_numpy(int))
    qclo, qchi = boot_stat_ci(quad_weighted_kappa,
                              cases["reader1_current_severity"].to_numpy(int),
                              cases["reader2_current_severity"].to_numpy(int))
    full_conc = int(cases["full_agreement"].sum())
    adjud = int(cases["adjudicator_used"].sum())

    rep = cases["target"].to_numpy(int); exp = cases["final_target"].to_numpy(int)
    lab_agree = float((rep == exp).mean())
    llo, lhi = boot_stat_ci(lambda a, b: float((a == b).mean()), rep, exp)
    rn_ep = int(((rep == 0) & (exp == 1)).sum())
    rp_en = int(((rep == 1) & (exp == 0)).sum())

    agreement = {
        "pairwise_worsening_agreement": round(agree, 4),
        "pairwise_worsening_agreement_ci": [round(alo, 4), round(ahi, 4)],
        "cohen_kappa_worsening": round(kap, 4),
        "cohen_kappa_ci": [round(klo, 4), round(khi, 4)],
        "qwk_prior_severity": round(qp, 4), "qwk_prior_ci": [round(qplo, 4), round(qphi, 4)],
        "qwk_current_severity": round(qc, 4), "qwk_current_ci": [round(qclo, 4), round(qchi, 4)],
        "two_reader_full_concordance_n": full_conc,
        "adjudicated_n": adjud,
        "expert_worsening_n": int(exp.sum()),
        "expert_prevalence": round(float(exp.mean()), 4),
        "report_vs_expert_agreement": round(lab_agree, 4),
        "report_vs_expert_agreement_ci": [round(llo, 4), round(lhi, 4)],
        "report_neg_expert_pos": rn_ep,
        "report_pos_expert_neg": rp_en,
        "n_discordant": rn_ep + rp_en,
        "unevaluable_final": int(cases["final_unevaluable"].sum()),
    }
    (OUT / "FINAL_agreement.json").write_text(json.dumps(agreement, indent=2) + "\n",
                                              encoding="utf8")
    print("\n=== TABLE 4 AGREEMENT ===")
    for k, v in agreement.items():
        print(f"  {k}: {v}")

    # ---------- corrected tipping point ----------
    print("\n=== TIPPING POINT (expert ref, temporal vs difference, matched 128) ===")
    y3 = sub["final_target"].to_numpy(int).copy()
    yrep = sub["target"].to_numpy(int)
    st = sub["temporal_score"].to_numpy(float); sd = sub["difference_score"].to_numpy(float)
    d0, lo0, hi0 = paired_boot_diff(fast_auc, y3, st, sd, B)
    print(f"  n_reverted= 0  dAUC={d0:+.4f} ({lo0:+.4f},{hi0:+.4f})"
          f"  CI_crosses_zero={lo0 <= 0}")
    tip_rows = [{"n_reverted": 0, "d_auc": round(d0, 4), "lo": round(lo0, 4),
                 "hi": round(hi0, 4), "ci_crosses_zero": bool(lo0 <= 0)}]
    note = ("Base contrast at matched PCA=128 is already non-significant "
            "(CI includes zero); a tipping-point analysis is not informative here.")
    if lo0 > 0:
        disc = set(np.where(y3 != yrep)[0].tolist())
        order = [i for i in np.argsort(-np.abs(st - sd)) if i in disc]
        ytmp = y3.copy()
        for n_rev, i in enumerate(order, start=1):
            ytmp[i] = yrep[i]
            d, lo, hi = paired_boot_diff(fast_auc, ytmp, st, sd, 2000)
            tip_rows.append({"n_reverted": n_rev, "d_auc": round(d, 4), "lo": round(lo, 4),
                             "hi": round(hi, 4), "ci_crosses_zero": bool(lo <= 0)})
            print(f"  n_reverted={n_rev:2d}  dAUC={d:+.4f} ({lo:+.4f},{hi:+.4f})")
            if lo <= 0:
                note = f"CI first includes zero after reverting {n_rev} discordant label(s)."
                break
    print(f"  NOTE: {note}")
    pd.DataFrame(tip_rows).to_csv(OUT / "FINAL_tipping_point.csv", index=False)
    (OUT / "FINAL_tipping_point_note.json").write_text(
        json.dumps({"note": note}, indent=2) + "\n", encoding="utf8")

    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
