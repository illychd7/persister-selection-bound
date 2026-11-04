"""Deterministic common map vs stochastic diversification.
A common map f stretches within- and between-clone variance alike (to first order); stochastic,
non-heritable diversification inflates within-clone variance only. Between-clone variance is
noise-corrected with split halves (covariance of clone centroids across independent halves)."""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import json, numpy as np, pandas as pd
z = np.load(PC9 + "/pc9_pp.npz", allow_pickle=True)
Y = z["pc"][:, :20].astype(float); samp = z["sample"]; clone = z["clone"]
TRE = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
rng = np.random.default_rng(11)
def comps(Ya, ca, K, w, nsplit=100):
    within = np.array([np.var(Ya[ca == c], 0, ddof=1).sum() for c in K]) @ w
    btw = []
    for _ in range(nsplit):
        ma, mb = [], []
        for c in K:
            i = rng.permutation(np.where(ca == c)[0]); ma.append(Ya[i[: len(i) // 2]].mean(0)); mb.append(Ya[i[len(i) // 2:]].mean(0))
        ma, mb = np.array(ma), np.array(mb); ma -= w @ ma; mb -= w @ mb
        btw.append(w @ (ma * mb).sum(1))
    return float(within), float(np.median(btw))
out = {}
a0 = samp == "0"
for t in TRE:
    at = samp == t; c0, ct = clone[a0], clone[at]
    s0 = pd.Series(c0[c0 != ""]).value_counts(); st = pd.Series(ct[ct != ""]).value_counts()
    K = [c for c in st.index if c in s0.index and s0[c] >= 4 and st[c] >= 4]
    w = np.array([st[c] for c in K], float); w /= w.sum()          # same clone weights at both times
    w0, b0 = comps(Y[a0], c0, K, w); wt, bt = comps(Y[at], ct, K, w)
    out[t] = dict(n_clones=len(K), within_ratio=wt / w0, between_ratio=bt / b0, between_share_day0=b0 / (b0 + w0), between_share_t=bt / (bt + wt))
    print(t, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[t].items()}, flush=True)
json.dump(out, open(RESULTS + "/within_vs_between.json", "w"), indent=1)
