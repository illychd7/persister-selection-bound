#!/usr/bin/env python3
"""Finite-sample lower confidence bound for the selective enrichment M* = exp D_inf(nu || mu).

Sample splitting: baseline cells -> reference S, threshold set T, estimation set U (disjoint).
For thresholds s_k = q_k-quantiles of d(T, S) (k = 1..K), with B_k = {y : d(y,S) > s_k}:
    p0_k = mu(B_k) <= UB_k  (one-sided Clopper-Pearson from U, level alpha/(2K))
    pt_k = nu(B_k) >= LB_k  (one-sided Clopper-Pearson from the time-t sample, level alpha/(2K))
Theorem 5.7(iii) gives M* >= pt_k / p0_k for every k, hence
    M_L = max_k LB_k / UB_k   satisfies   P(M* >= M_L) >= 1 - alpha.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta

GRID = (0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 1.0)

def cp_lower(k, n, a):
    return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))

def cp_upper(k, n, a):
    return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))

def nn_dist(Q, S, chunk=4000):
    return np.concatenate([cdist(Q[i:i + chunk], S).min(1) for i in range(0, len(Q), chunk)])

def lower_bound(Y0, Yt, rng, alpha=0.05, grid=GRID, mS=400, fracT=0.4, return_detail=False):
    """Y0: baseline cells (n0 x q), Yt: time-t cells. Returns the (1-alpha) lower confidence bound for M*."""
    p = rng.permutation(len(Y0))
    mS = min(mS, len(Y0) // 3)
    S = Y0[p[:mS]]; rest = p[mS:]
    nT = int(round(fracT * len(rest)))
    T, U = Y0[rest[:nT]], Y0[rest[nT:]]
    dT, dU, dt = nn_dist(T, S), nn_dist(U, S), nn_dist(Yt, S)
    K = len(grid); a = alpha / (2 * K)
    best, det = 0.0, []
    for q in grid:
        s = np.quantile(dT, q, method="inverted_cdf") if q < 1 else dT.max()
        k0, kt = int((dU > s).sum()), int((dt > s).sum())
        LB, UB = cp_lower(kt, len(dt), a), cp_upper(k0, len(dU), a)
        r = LB / UB
        det.append(dict(q=q, s=float(s), k0=k0, n0=len(dU), kt=kt, nt=len(dt), LB=LB, UB=UB, ratio=r))
        best = max(best, r)
    return (best, det) if return_detail else best

def pvalue(Y0, Yt, M0, rng_seed, **kw):
    """p-value for H0: M* <= M0, by bisection on alpha (same split, fixed seed)."""
    lo, hi = 1e-12, 1.0
    if lower_bound(Y0, Yt, np.random.default_rng(rng_seed), alpha=hi, **kw) <= M0:
        return 1.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if lower_bound(Y0, Yt, np.random.default_rng(rng_seed), alpha=mid, **kw) > M0:
            hi = mid
        else:
            lo = mid
    return hi

def lower_bound_mixture(Y0, parts, rng, alpha=0.05, grid=GRID, mS=400, fracT=0.4):
    """Gate-agnostic bound: the time-t population is an unknown mixture of sorted fractions `parts`.
    Since nu(B) >= min_j nu_j(B) for every mixture, M* >= max_k min_j LB_jk / UB_k (Bonferroni over (J+1)K events)."""
    p = rng.permutation(len(Y0)); mS = min(mS, len(Y0) // 3)
    S = Y0[p[:mS]]; rest = p[mS:]; nT = int(round(fracT * len(rest)))
    dT, dU = nn_dist(Y0[rest[:nT]], S), nn_dist(Y0[rest[nT:]], S)
    dj = [nn_dist(P, S) for P in parts]
    K = len(grid); a = alpha / ((len(parts) + 1) * K); best = 0.0
    for q in grid:
        s = np.quantile(dT, q, method="inverted_cdf") if q < 1 else dT.max()
        UB = cp_upper(int((dU > s).sum()), len(dU), a)
        LB = min(cp_lower(int((d > s).sum()), len(d), a) for d in dj)
        best = max(best, LB / UB)
    return best
