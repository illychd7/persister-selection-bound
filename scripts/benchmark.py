#!/usr/bin/env python3
"""Head-to-head benchmark on one question: is 'selection + one shared response' sufficient?

Ground truth: semi-synthetic populations built from REAL PC9 day-0 cells (disjoint halves of one
library) with known clone-level selection g. The oracle budget B = true sup g / E g (exact for the
sampling scheme). Every method returns a statistic T on the scale of selective enrichment and
rejects 'selection suffices' when T > B (or, for pure two-sample tests, when p < 0.05).

Null scenarios (correct answer: do not reject): selection only; selection + common response
(moderate and strong selection). Alternatives: clone-intrinsic shifts (5%, 30% of response energy),
state-dependent shifts (30%), cell-level heterogeneity of response.

Methods (all given the same FIT/EVAL split and, where they accept one, the same common-response
adjustment learned on FIT cells):
  ours        lower confidence bound M_L(nu | mu0 + delta)              (finite-sample, alpha = 0.05)
  ours_unadj  M_L(nu | mu0)                                             (no response adjustment)
  clf_ratio   classifier density ratio r = p/(1-p) * n0/nt, max and q99 over EVAL cells
  c2st        classifier two-sample test on held-out accuracy (binomial p < 0.05)
  meld        MELD sample-associated likelihood -> density ratio, max and q99
  milo_like   kNN-neighbourhood abundance ratio, max over neighbourhoods significant at FDR 0.1
  uot         unbalanced entropic OT (WOT-like) growth estimate on unadjusted data, max and q99
"""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import sys, json, time, numpy as np, pandas as pd
from scipy.stats import binomtest, binom
pass; pass
from seltest import lower_bound
from decomposition import within_clone_shift
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
import ot, meld

Q = 20
z = np.load(PC9 + "/pc9_pp.npz", allow_pickle=True)
Yall = z["pc"][:, :Q].astype(float); samp = z["sample"]; clone = z["clone"]
lab0 = (samp == "0") & (clone != ""); Y0, c0 = Yall[lab0], clone[lab0]
dreal, _ = within_clone_shift(Yall[samp == "0"], clone[samp == "0"], Yall[samp == "3_rep1"], clone[samp == "3_rep1"])
cl = np.unique(c0); cidx = {c: i for i, c in enumerate(cl)}; ci = np.array([cidx[c] for c in c0])
cm = np.array([Y0[ci == i].mean(0) for i in range(len(cl))])
E_delta = dreal @ dreal

def make(seed, sigma_g=0.5, common=True, phi=0.0, kind=None, cell_noise=0.0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y0)); ref, src = perm[: len(perm) // 2], perm[len(perm) // 2:]
    g = np.exp(sigma_g * rng.standard_normal(len(cl)))
    w = g[ci[src]]; B = float(w.max() / w.mean())                       # exact sup of the selection density
    nt = int(1.5 * len(src)); pick = src[rng.choice(len(src), nt, replace=True, p=w / w.sum())]
    Yt = Y0[pick] + 1e-6 * rng.standard_normal((nt, Q))
    if common: Yt += dreal
    if phi > 0:
        if kind == "intrinsic": r = rng.standard_normal((len(cl), Q))
        else:
            U = rng.standard_normal((Q, Q)) / np.sqrt(Q); W = rng.standard_normal((Q, Q))
            r = np.tanh((cm - cm.mean(0)) @ U / cm.std()) @ W.T
        wt = np.bincount(ci[pick], minlength=len(cl)) / nt; r = r - wt @ r
        r *= np.sqrt(phi / (1 - phi) * E_delta / (wt @ (r ** 2).sum(1))); Yt += r[ci[pick]]
    if cell_noise > 0: Yt += cell_noise * rng.standard_normal(Yt.shape)
    X0 = Y0[ref]
    f0 = rng.random(len(X0)) < 0.3; ft = rng.random(len(Yt)) < 0.3
    return dict(X0f=X0[f0], X0e=X0[~f0], Xtf=Yt[ft], Xte=Yt[~ft], B=B)

def ratio_stats(r):
    return float(np.max(r)), float(np.quantile(r, 0.99))

def run_methods(d, rng):
    X0f, X0e, Xtf, Xte, B = d["X0f"], d["X0e"], d["Xtf"], d["Xte"], d["B"]
    delta = Xtf.mean(0) - X0f.mean(0)
    out, tm = {}, {}
    t = time.time(); out["ours"] = lower_bound(X0e + delta, Xte, np.random.default_rng(0)); tm["ours"] = time.time() - t
    out["ours_unadj"] = lower_bound(X0e, Xte, np.random.default_rng(0))
    # classifier density ratio and C2ST (response-adjusted baseline vs treated)
    t = time.time()
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    Xf = np.vstack([X0f + delta, Xtf]); yf = np.r_[np.zeros(len(X0f)), np.ones(len(Xtf))]
    clf.fit(Xf, yf)
    Xe = np.vstack([X0e + delta, Xte]); ye = np.r_[np.zeros(len(X0e)), np.ones(len(Xte))]
    p = np.clip(clf.predict_proba(Xe)[:, 1], 1e-6, 1 - 1e-6)
    prior = len(Xtf) / len(X0f)
    r = (p / (1 - p)) / prior
    out["clf_ratio_max"], out["clf_ratio_q99"] = ratio_stats(r[ye == 1])
    acc_n = int(((p > 0.5) == (ye == 1)).sum()); maj = max(ye.mean(), 1 - ye.mean())
    out["c2st_p"] = float(binomtest(acc_n, len(ye), maj, alternative="greater").pvalue); tm["clf"] = time.time() - t
    # MELD on response-adjusted baseline + treated (EVAL cells)
    t = time.time()
    lab = np.r_[np.zeros(len(X0e)), np.ones(len(Xte))]
    dens = meld.MELD(verbose=0, knn=10, beta=60).fit_transform(Xe, sample_labels=np.where(lab == 1, "t", "c"))
    dens = dens.div(dens.sum(axis=0), axis=1) if hasattr(dens, "div") else dens / dens.sum(0)   # per-sample normalised densities
    dens = np.asarray(dens)
    cols = list(np.unique(np.where(lab == 1, "t", "c")))
    rm = dens[:, cols.index("t")] / np.maximum(dens[:, cols.index("c")], 1e-12)
    out["meld_max"], out["meld_q99"] = ratio_stats(rm[lab == 1]); tm["meld"] = time.time() - t
    # Milo-like neighbourhood abundance (k = 30, 10% index cells), binomial test with BH FDR 0.1
    t = time.time()
    k = 30; nn = NearestNeighbors(n_neighbors=k).fit(Xe); idx = rng.choice(len(Xe), max(50, len(Xe) // 10), replace=False)
    nb = nn.kneighbors(Xe[idx], return_distance=False)
    nt_, n0_ = (lab == 1).sum(), (lab == 0).sum(); p0 = nt_ / (nt_ + n0_)
    ct = lab[nb].sum(1); cc = k - ct
    fc = ((ct + .5) / nt_) / ((cc + .5) / n0_)
    pv = binom.sf(ct - 1, k, p0)
    order = np.argsort(pv); m = len(pv); bh = np.zeros(m, bool)
    passed = np.where(pv[order] <= 0.1 * np.arange(1, m + 1) / m)[0]
    if len(passed): bh[order[: passed.max() + 1]] = True
    out["milo_sig_max"] = float(fc[bh].max()) if bh.any() else 1.0
    out["milo_frac_sig"] = float(bh.mean()); tm["milo"] = time.time() - t
    # unbalanced OT (WOT-like) on UNADJUSTED data: mass created at baseline points = growth
    t = time.time()
    a_i = rng.choice(len(X0e), min(1500, len(X0e)), replace=False); b_i = rng.choice(len(Xte), min(1500, len(Xte)), replace=False)
    A, Bt = X0e[a_i], Xte[b_i]
    C = ot.dist(A, Bt); C /= np.median(C)
    a = np.full(len(A), 1 / len(A)); b = np.full(len(Bt), 1 / len(Bt))
    P = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=0.05, reg_m=1.0, numItermax=2000)
    gr = P.sum(1) / a; gr = gr / gr.mean()
    out["uot_max"], out["uot_q99"] = ratio_stats(gr); tm["uot"] = time.time() - t
    out["B"] = B; out["time"] = tm
    return out

SC = [("N1 selection only", dict(common=False)), ("N2 selection + common", dict()), ("N3 strong selection + common", dict(sigma_g=1.0)),
      ("A1 + clone-intrinsic 5%", dict(phi=0.05, kind="intrinsic")), ("A2 + clone-intrinsic 30%", dict(phi=0.30, kind="intrinsic")),
      ("A3 + state-dependent 30%", dict(phi=0.30, kind="state")), ("A4 + cell-level heterogeneity", dict(cell_noise=3.0))]

if __name__ == "__main__":
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    res = []
    for name, kw in SC:
        for s in range(nseed):
            d = make(100 + s, **kw); o = run_methods(d, np.random.default_rng(s)); o.update(scenario=name, seed=s); res.append(o)
            print(name, s, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items() if k not in ("time", "scenario", "seed")}, flush=True)
            json.dump(res, open(RESULTS + "/benchmark_results.json", "w"), indent=1)
    print("done")
