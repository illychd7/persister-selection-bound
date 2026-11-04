#!/usr/bin/env python3
"""Pooled Watermelon lines (GSE150949): naive vs day-10 cycling / non-cycling persisters, per line and experiment."""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import sys, json, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import PCA
pass
from preprocess import thin
from seltest import lower_bound, lower_bound_mixture

D = pathlib.Path(PC9); T, Q, THETA, HVG = 1000, 20, 100.0, 2000
rng = np.random.default_rng(11)
m = pd.read_csv(D / "GSE150949_pooled_watermelon.metadata.matrix.csv.gz")
X = sp.load_npz(D / "pooled_counts.npz").tocsr(); g = np.array(open(D / "pooled_genes.txt").read().split("\n"))
mt = np.array([x.startswith("MT-") for x in g]); nuc = np.asarray(X[:, ~mt].sum(1)).ravel()
keep = (nuc >= T) & (m["percent.mt"].values < 20)
rng2 = np.random.default_rng(2027)
WGRID = np.linspace(0.05, 0.95, 19)
def wtrace(A): return float(np.var(A, 0, ddof=1).sum())
out = {}
for line in ["A375_SKIN", "HT29_LARGE_INTESTINE", "PC9_LUNG"]:
    sel = keep & (m.cell_line.values == line)
    mm = m[sel].reset_index(drop=True); Xt = thin(X[sel][:, ~mt], T, rng)
    pools = mm.sample_pool.values
    fit = np.zeros(len(mm), bool)
    for s in np.unique(pools):
        ii = np.where(pools == s)[0]; fit[rng.choice(ii, min(800, int(.3 * len(ii))), replace=False)] = True
    nb = min((fit & (pools == s)).sum() for s in np.unique(pools))
    bal = np.concatenate([rng.choice(np.where(fit & (pools == s))[0], nb, replace=False) for s in np.unique(pools)])
    p = np.asarray(Xt[bal].sum(0)).ravel() / (T * len(bal)); ok = np.where(p > 0)[0]
    def resid(C, cols):
        M = T * p[cols]; R = (C[:, cols].toarray() - M) / np.sqrt(M + M**2 / THETA); c = np.sqrt(len(bal)); return np.clip(R, -c, c)
    rv = np.zeros(len(p))
    for k in range(0, len(ok), 3000): rv[ok[k:k + 3000]] = resid(Xt[bal], ok[k:k + 3000]).var(0)
    hv = np.argsort(rv)[::-1][:HVG]
    Z = np.vstack([resid(Xt[i:i + 5000], hv) for i in range(0, Xt.shape[0], 5000)])
    pc = PCA(n_components=Q, svd_solver="full", random_state=0).fit(Z[bal]).transform(Z)
    ev = ~fit; out[line] = {}
    for ex in sorted(mm.experiment.unique()):
        S = lambda typ: pc[ev & (mm.experiment.values == ex) & (mm.sample_type.values == typ)]
        Y0, Yc, Yn = S("naive"), S("cycling"), S("non-cycling")
        if min(len(Y0), len(Yc), len(Yn)) < 100: continue
        r = dict(n0=len(Y0), ncyc=len(Yc), nnon=len(Yn),
                 ML_cycling=lower_bound(Y0, Yc, np.random.default_rng(0)), ML_noncycling=lower_bound(Y0, Yn, np.random.default_rng(0)),
                 ML_mixture=lower_bound_mixture(Y0, [Yc, Yn], np.random.default_rng(0)),
                 ML_loss_cyc=lower_bound(Yc, Y0, np.random.default_rng(0)), ML_loss_non=lower_bound(Yn, Y0, np.random.default_rng(0)))
        h = rng.permutation(len(Y0)); r["control_naive_halves"] = lower_bound(Y0[h[:len(h)//2]], Y0[h[len(h)//2:]], np.random.default_rng(0))
        # ---- added: selection + common response (translation estimated on FIT cells, bound on EVAL cells)
        F = lambda typ: pc[fit & (mm.experiment.values == ex) & (mm.sample_type.values == typ)]
        F0, Fc, Fn = F("naive"), F("cycling"), F("non-cycling")
        dc, dn = Fc.mean(0) - F0.mean(0), Fn.mean(0) - F0.mean(0)
        # one common translation for the whole persister population; the cycling fraction w is unknown -> min over a grid (conservative)
        mix = [lower_bound_mixture(Y0 + (w * dc + (1 - w) * dn), [Yc, Yn], np.random.default_rng(0)) for w in WGRID]
        r["common_ML_mixture_min_over_w"] = float(min(mix)); r["common_w_argmin"] = float(WGRID[int(np.argmin(mix))])
        r["common_ML_mixture_by_w"] = [float(v) for v in mix]
        # fraction-specific translations (weaker null: each sorted fraction gets its own response)
        r["fracspec_ML_cycling"] = lower_bound(Y0 + dc, Yc, np.random.default_rng(0)); r["fracspec_ML_noncycling"] = lower_bound(Y0 + dn, Yn, np.random.default_rng(0))
        r["|delta_cyc|"] = float(np.linalg.norm(dc)); r["|delta_non|"] = float(np.linalg.norm(dn)); r["cos_delta"] = float(dc @ dn / np.linalg.norm(dc) / np.linalg.norm(dn))
        # population variance expansion (no barcodes: includes selection), split along / orthogonal to the response axis
        u = (dc + dn); u = u / np.linalg.norm(u); P = np.eye(len(u)) - np.outer(u, u)
        for nm, Yt in [("cyc", Yc), ("non", Yn)]:
            r[f"var_ratio_{nm}"] = wtrace(Yt) / wtrace(Y0); r[f"var_ratio_{nm}_along"] = float(np.var(Yt @ u, ddof=1) / np.var(Y0 @ u, ddof=1))
            r[f"var_ratio_{nm}_orth"] = wtrace(Yt @ P) / wtrace(Y0 @ P)
        out[line][ex] = r
        print(line, ex, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "common_ML_mixture_by_w"}, flush=True)
json.dump(out, open(RESULTS + "/pooled_decomp.json", "w"), indent=1)
