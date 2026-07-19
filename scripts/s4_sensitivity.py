"""
Supplementary Table S4 sensitivity references at matched PCA=128:
  (i) three-reader majority-vote reference
  (ii) worsening restricted to clinically actionable transitions into grade 2-3
Both evaluate the temporal-vs-difference and temporal-vs-current AUC contrasts
on the locked unified-128 scores.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CASES = DATA / "final_adjudicated_cases.csv"
SCORES = DATA / "unified128_test_scores.csv"
SEED = 42; B = 5000


def fast_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    p, n = int(y.sum()), int(len(y) - y.sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    return (ranks[y == 1].sum() - p*(p+1)/2.0) / (p*n)


def paired_boot(y, sa, sb, B=B, seed=SEED):
    obs = fast_auc(y, sa) - fast_auc(y, sb)
    rng = np.random.default_rng(seed); vals = []; n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(fast_auc(yy, sa[idx]) - fast_auc(yy, sb[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def delong_p(y, sa, sb):
    y = np.asarray(y, int); pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    m, n = len(pos), len(neg); idx = np.concatenate([pos, neg])
    preds = np.vstack([np.asarray(sa,float)[idx], np.asarray(sb,float)[idx]])
    tx = np.empty([2,m]); ty = np.empty([2,n]); tz = np.empty([2,m+n])
    for r in range(2):
        tx[r]=_midrank(preds[r,:m]); ty[r]=_midrank(preds[r,m:]); tz[r]=_midrank(preds[r,:])
    aucs = tz[:,:m].sum(1)/m/n - (m+1.0)/2.0/n
    cov = np.cov((tz[:,:m]-tx)/n)/m + np.cov(1.0-(tz[:,m:]-ty)/m)/n
    var = cov[0,0]+cov[1,1]-2*cov[0,1]
    if var <= 0: return float("nan")
    return float(2*stats.norm.sf(abs((aucs[0]-aucs[1])/np.sqrt(var))))


cases = pd.read_csv(CASES)
scores = pd.read_csv(SCORES)
sub = cases.merge(scores[["pair_key","current_score","difference_score","temporal_score"]],
                  on="pair_key", how="left", validate="one_to_one")
st = sub["temporal_score"].to_numpy(float)
sd = sub["difference_score"].to_numpy(float)
sc = sub["current_score"].to_numpy(float)

# (i) three-reader majority vote
r1 = sub["reader1_target"].to_numpy(int)
r2 = sub["reader2_target"].to_numpy(int)
adj_used = sub["adjudicator_used"].fillna(False).to_numpy(bool)
adj_t = (sub["adjudicator_current_severity"].fillna(-1).to_numpy(float)
         > sub["adjudicator_prior_severity"].fillna(-1).to_numpy(float)).astype(int)
third = np.where(adj_used, adj_t, r1)  # concordant pairs: r1==r2, third vote irrelevant
maj = ((r1 + r2 + third) >= 2).astype(int)

# (ii) transitions into grade 2-3
fp = sub["final_prior_severity"].to_numpy(int)
fc = sub["final_current_severity"].to_numpy(int)
grade23 = ((fc >= 2) & (fc > fp)).astype(int)

refs = {
    "three_reader_majority_vote": maj,
    "worsening_into_grade_2_3": grade23,
    "final_adjudicated (reference)": sub["final_target"].to_numpy(int),
}

rows = []
print("=== S4 sensitivity references (matched PCA=128) ===")
for name, y in refs.items():
    dtd, lotd, hitd = paired_boot(y, st, sd)
    ptd = delong_p(y, st, sd)
    dtc, lotc, hitc = paired_boot(y, st, sc)
    ptc = delong_p(y, st, sc)
    rows.append({"reference": name, "n_worsening": int(y.sum()), "prevalence": round(float(y.mean()),4),
                 "temporal_auc": round(fast_auc(y, st),4),
                 "difference_auc": round(fast_auc(y, sd),4),
                 "current_auc": round(fast_auc(y, sc),4),
                 "T_minus_D_auc": round(dtd,4), "TmD_lo": round(lotd,4), "TmD_hi": round(hitd,4), "TmD_delong_p": ptd,
                 "T_minus_C_auc": round(dtc,4), "TmC_lo": round(lotc,4), "TmC_hi": round(hitc,4), "TmC_delong_p": ptc})
    print(f"\n{name}: n_worse={int(y.sum())} prev={y.mean():.3f}")
    print(f"  T={fast_auc(y,st):.4f} D={fast_auc(y,sd):.4f} C={fast_auc(y,sc):.4f}")
    print(f"  T-D dAUC={dtd:+.4f} ({lotd:+.4f},{hitd:+.4f}) p={ptd:.4g}  {'SIG' if lotd>0 else 'ns'}")
    print(f"  T-C dAUC={dtc:+.4f} ({lotc:+.4f},{hitc:+.4f}) p={ptc:.4g}  {'SIG' if lotc>0 else 'ns'}")

pd.DataFrame(rows).to_csv(OUT / "FINAL_s4_sensitivity.csv", index=False)
print(f"\n[out] {OUT / 'FINAL_s4_sensitivity.csv'}")
