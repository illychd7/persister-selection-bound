#!/usr/bin/env python3
"""Appendix A: clone-cross-fitted response models and the power ceiling of that design.

Reconstruction (2026-11-03) of the analysis described in Appendix A. The original script was not
available; the design follows the text, and the ceiling formula and the per-bound level were
verified against the original crossfit_power_ceiling.csv (exact match with alpha = 0.05/32).
Clone eligibility and fold assignment of the original run could not be recovered, so cell numbers
and bounds differ from the original Table 3.

Design
  targets      3_rep1, 3_rep2, 7_rep1, 7_rep2
  clones       barcoded clones with >= 4 cells at day 0 and at time t (all cells)
  folds        2-fold KFold over clones (shuffle, random_state = 0); TEST = one fold, TRAIN = the other
  models       fitted on TRAIN-clone cells only, evaluated on TEST-clone cells:
                 identity      y
                 translation   y + (mean_t - mean_0)
                 ridge affine  m_t + A (y - m_0), A from ridge regression of TRAIN clone centroids
                               at t on clone centroids at day 0 (RidgeCV, centred)
                 OT+KNN        exact optimal transport between TRAIN day-0 and day-t cells
                               (<= 2000 each, uniform weights); barycentric displacement of each
                               TRAIN day-0 cell; a TEST cell moves by the mean displacement of its
                               10 nearest TRAIN day-0 cells
  bound        seltest.lower_bound(K(Y0_test), Yt_test), alpha = 0.05 / 32
               (4 targets x 2 folds x 4 models; 224 tail-set events)
  ceiling      largest attainable bound in this design: no reference estimation cell and every
               target cell in the tail, LB(n_t, n_t) / UB(0, n_est), at the same level
  prediction   weighted squared error of predicted TEST clone centroids at t (clone weights at t)

Output: crossfit_appendixA.csv
"""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import sys, numpy as np, pandas as pd, ot
from scipy.stats import beta
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
pass
from seltest import lower_bound, GRID

Q, NMIN, ALPHA = 20, 4, 0.05 / 32
TARGETS = ["3_rep1", "3_rep2", "7_rep1", "7_rep2"]
z = np.load(PC9 + "/pc9_pp.npz", allow_pickle=True)
Y = z["pc"][:, :Q].astype(float); samp, clone = z["sample"], z["clone"]


def n_estimation(n0, mS=400, fracT=0.4):
    """Size of the estimation set U inside seltest.lower_bound."""
    mS = min(mS, n0 // 3); rest = n0 - mS
    return rest - int(round(fracT * rest))


def ceiling(n_est, nt, alpha=ALPHA):
    a = alpha / (2 * len(GRID))
    return float(beta.ppf(a, nt, 1) / beta.ppf(1 - a, 1, n_est))


def fit_models(Y0, c0, Yt, ct, K, rng):
    m0, mt = Y0.mean(0), Yt.mean(0)
    C0 = np.array([Y0[c0 == c].mean(0) for c in K]); Ct = np.array([Yt[ct == c].mean(0) for c in K])
    r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(C0 - C0.mean(0), Ct - Ct.mean(0))
    A = r.coef_
    i0 = rng.choice(len(Y0), min(2000, len(Y0)), replace=False); it = rng.choice(len(Yt), min(2000, len(Yt)), replace=False)
    S0, St = Y0[i0], Yt[it]
    P = ot.emd(np.full(len(S0), 1 / len(S0)), np.full(len(St), 1 / len(St)), ot.dist(S0, St))
    disp = (P @ St) / P.sum(1, keepdims=True) - S0
    nn = NearestNeighbors(n_neighbors=10).fit(S0)
    return {"identity": lambda X: X,
            "translation": lambda X: X + (mt - m0),
            "affine": lambda X: mt + (X - m0) @ A.T,
            "ot_knn": lambda X: X + disp[nn.kneighbors(X, return_distance=False)].mean(1)}


rows = []
for t in TARGETS:
    a0, at = (samp == "0") & (clone != ""), (samp == t) & (clone != "")
    s0, st = pd.Series(clone[a0]).value_counts(), pd.Series(clone[at]).value_counts()
    K = np.array(sorted(set(s0.index[s0 >= NMIN]) & set(st.index[st >= NMIN])))
    for fold, (tr, te) in enumerate(KFold(2, shuffle=True, random_state=0).split(K)):
        Ktr, Kte = K[tr], K[te]
        m0tr, mttr = a0 & np.isin(clone, Ktr), at & np.isin(clone, Ktr)
        m0te, mtte = a0 & np.isin(clone, Kte), at & np.isin(clone, Kte)
        models = fit_models(Y[m0tr], clone[m0tr], Y[mttr], clone[mttr], Ktr, np.random.default_rng(fold))
        Y0te, Ytte, c0te, ctte = Y[m0te], Y[mtte], clone[m0te], clone[mtte]
        n_est = n_estimation(len(Y0te))
        row = dict(target=t, fold=fold, n_train_clones=len(Ktr), n_test_clones=len(Kte), n0_test=len(Y0te),
                   n_ref_est=n_est, n_target=len(Ytte), ceiling=ceiling(n_est, len(Ytte)))
        wt = np.array([(ctte == c).sum() for c in Kte], float); wt /= wt.sum()
        obs = np.array([Ytte[ctte == c].mean(0) for c in Kte])
        for name, f in models.items():
            row[name] = lower_bound(f(Y0te), Ytte, np.random.default_rng(0), alpha=ALPHA)
            pred = np.array([f(Y0te[c0te == c]).mean(0) for c in Kte])
            row[f"centroid_err_{name}"] = float(wt @ ((pred - obs) ** 2).sum(1))
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
pd.DataFrame(rows).to_csv(RESULTS + "/crossfit_appendixA.csv", index=False)
