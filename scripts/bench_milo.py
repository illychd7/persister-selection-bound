#!/usr/bin/env python3
"""Benchmark with the official Milo implementation (pertpy 1.4.0, scverse; default PyDESeq2
negative-binomial GLM, spatial FDR) using replicate samples.

Design per simulated dataset: 3 baseline replicates (random partition of the baseline half) and
3 independently drawn treated replicate cultures (same selection g, same response and plasticity).
Milo tests nhood abundance with design ~condition across the 6 samples on response-adjusted EVAL
cells (baseline translated by the FIT-estimated common response). Statistic: largest fold change
2^logFC among nhoods with SpatialFDR < 0.1, compared with the oracle selection strength B.
Our bound is recomputed on the same pooled cells.
"""
from config import RESULTS  # data/results/figure folders, see config.py
import sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
import anndata as ad, scanpy as sc, pertpy as pt
pass; pass
from seltest import lower_bound
import benchmark as b

def make_reps(seed, sigma_g=0.5, common=True, phi=0.0, kind=None, cell_noise=0.0, nrep=3):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(b.Y0)); ref, src = perm[: len(perm) // 2], perm[len(perm) // 2:]
    g = np.exp(sigma_g * rng.standard_normal(len(b.cl))); w = g[b.ci[src]]; B = float(w.max() / w.mean())
    r = None
    if phi > 0:
        if kind == "intrinsic": r = rng.standard_normal((len(b.cl), b.Q))
        else:
            U = rng.standard_normal((b.Q, b.Q)) / np.sqrt(b.Q); W = rng.standard_normal((b.Q, b.Q))
            r = np.tanh((b.cm - b.cm.mean(0)) @ U / b.cm.std()) @ W.T
        wt = w / w.sum(); wc = np.bincount(b.ci[src], weights=wt, minlength=len(b.cl)); r = r - wc @ r
        r *= np.sqrt(phi / (1 - phi) * b.E_delta / (wc @ (r ** 2).sum(1)))
    X0 = b.Y0[ref]; rep0 = rng.integers(0, nrep, len(X0))
    Xt, rept = [], []
    nt = int(1.5 * len(src) / nrep)
    for k in range(nrep):
        pick = src[rng.choice(len(src), nt, replace=True, p=w / w.sum())]
        Y = b.Y0[pick] + 1e-6 * rng.standard_normal((nt, b.Q))
        if common: Y += b.dreal
        if r is not None: Y += r[b.ci[pick]]
        if cell_noise > 0: Y += cell_noise * rng.standard_normal(Y.shape)
        Xt.append(Y); rept.append(np.full(nt, k))
    Xt, rept = np.vstack(Xt), np.concatenate(rept)
    f0 = rng.random(len(X0)) < 0.3; ft = rng.random(len(Xt)) < 0.3
    return dict(X0f=X0[f0], X0e=X0[~f0], r0=rep0[~f0], Xtf=Xt[ft], Xte=Xt[~ft], rt=rept[~ft], B=B)

def milo_stat(Xe, cond, rep, k=30, prop=0.1, seed=0):
    adata = ad.AnnData(X=Xe.astype(np.float32))
    adata.obs["condition"] = np.where(cond == 1, "treated", "baseline")
    adata.obs["sample"] = [f"{c}_{r}" for c, r in zip(adata.obs["condition"], rep)]
    adata.obsm["X_pca"] = Xe.astype(np.float32)
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=k, random_state=seed)
    milo = pt.tl.Milo(); mdata = milo.load(adata)
    milo.make_nhoods(mdata["rna"], prop=prop, seed=seed)
    mdata = milo.count_nhoods(mdata, sample_col="sample")
    milo.da_nhoods(mdata, design="~condition", solver="pydeseq2")
    res = mdata["milo"].var
    sig = res["SpatialFDR"].values < 0.1
    fc = np.power(2.0, res["logFC"].values)
    return dict(milo_sig_max=float(fc[sig].max()) if sig.any() else 1.0, milo_frac_sig=float(sig.mean()),
                milo_n_nhoods=int(len(res)), milo_max_logFC_all=float(np.nanmax(res["logFC"].values)))

def run(d, k=30):
    delta = d["Xtf"].mean(0) - d["X0f"].mean(0)
    X0a = d["X0e"] + delta
    out = dict(B=d["B"], ours=lower_bound(X0a, d["Xte"], np.random.default_rng(0)))
    Xe = np.vstack([X0a, d["Xte"]]); cond = np.r_[np.zeros(len(X0a)), np.ones(len(d["Xte"]))]; rep = np.r_[d["r0"], d["rt"]]
    out.update(milo_stat(Xe, cond, rep, k=k))
    return out

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    res = []
    if mode == "main":
        for name, kw in b.SC:
            for s in range(nseed):
                o = run(make_reps(200 + s, **kw)); o.update(scenario=name, seed=s, k=30); res.append(o)
                print(name, s, {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in o.items() if kk not in ("scenario", "seed")}, flush=True)
                json.dump(res, open(RESULTS + "/milo_official_results.json", "w"), indent=1)
    else:
        for name, kw in [("N2 selection + common", dict()), ("A2 + clone-intrinsic 30%", dict(phi=0.30, kind="intrinsic"))]:
            for s in range(nseed):
                d = make_reps(200 + s, **kw)
                for k in [10, 50, 100]:
                    o = run(d, k=k); o.update(scenario=name, seed=s, k=k); res.append(o)
                    print(name, s, k, round(o["milo_sig_max"], 2), round(o["B"], 2), flush=True)
                json.dump(res, open(RESULTS + "/milo_official_sensitivity.json", "w"), indent=1)
    print("done")
