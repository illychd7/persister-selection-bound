#!/usr/bin/env python3
"""TNBC (GSE291678) and MeRLin in vitro (GSE299711) preprocessed with the conventions of
pc9_preprocess_keep.py: nuclear UMIs >= T, mito fraction < 0.2, thinning of nuclear counts to T,
FIT = min(1000, 30%) per sample, balanced FIT subset, analytic Pearson residuals (theta 100),
2000 HVG by residual variance, sklearn PCA (50 PCs; analyses use 20)."""
from config import GEO, MERLIN, NEWDATA, TNBC  # data/results/figure folders, see config.py
import sys, gzip, pathlib, json
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io
from sklearn.decomposition import PCA
pass
from preprocess import thin
T, MTMAX, HVG, NPC, THETA = 3000, 0.2, 2000, 50, 100.0
OUT = pathlib.Path(NEWDATA); OUT.mkdir(exist_ok=True)

def qc_thin(M, g, rng):
    mt = np.array([x.startswith("MT-") for x in g])
    umi = np.asarray(M.sum(1)).ravel(); nuc = np.asarray(M[:, ~mt].sum(1)).ravel(); mtf = 1 - nuc / np.maximum(umi, 1)
    keep = (nuc >= T) & (mtf < MTMAX)
    return thin(M[keep][:, ~mt], T, rng), keep, g[~mt]

RNG = np.random.default_rng(20261102)

def load_tnbc():
    G = pathlib.Path(GEO + "/raw")
    obs = pd.read_csv(TNBC + "/authors_obs.csv.gz", index_col=0, low_memory=False)
    obs["cellbc"] = obs.index.str.extract(r"^([ACGT]+)-", expand=False).values + "-1"
    names = {"PT-1": "PT1", "Dox": "30HR", "A4": "250_4", "A6": "250_6", "PT-2": "PT2",
             "A1": "250_1", "B2": "400_2", "B5": "400_4", "B6": "400_5"}
    mats, meta, genes = [], [], None
    for code, nm in names.items():
        f = next(G.glob(f"GSM*_{code}_matrix.mtx.gz")); pre = str(f).replace("_matrix.mtx.gz", "")
        M = scipy.io.mmread(gzip.open(f)).T.tocsr()
        bcs = pd.read_csv(pre + "_barcodes.tsv.gz", header=None)[0].values
        g = pd.read_csv(pre + "_features.tsv.gz", header=None, sep="\t")[1].astype(str).values
        genes = g if genes is None else genes
        o = obs[obs["sample-id"] == code]; pos = pd.Series(np.arange(len(bcs)), index=bcs)
        M = M[pos.loc[o["cellbc"]].values]
        single = (o["n_barcodes"] == 1) & (o["barcode-doublet"] == "no")
        cl = np.where(single, o["bgID"].astype(str), "")
        Mt, keep, gn = qc_thin(M, g, RNG); del M
        mats.append(Mt); meta.append(pd.DataFrame({"sample": nm, "batch": str(o["batch"].iloc[0]), "clone": cl[keep],
                                                  "clone_cluster": o["clone-cluster"].astype(str).values[keep]}))
        print(nm, int(keep.sum()), "/", keep.size, flush=True)
    return sp.vstack(mats).tocsr(), gn, pd.concat(meta, ignore_index=True)

def load_merlin():
    D = pathlib.Path(MERLIN); mats, meta = [], []
    for s in ["CTRL", "DT"]:
        M = scipy.io.mmread(D / f"{s}_counts.mtx").T.tocsr()
        g = pd.read_csv(D / f"{s}_genes.txt", header=None)[0].astype(str).values
        md = pd.read_csv(D / f"{s}_meta.csv")
        Mt, keep, gn = qc_thin(M, g, RNG); del M
        mats.append(Mt); meta.append(pd.DataFrame({"sample": s, "batch": "1", "clone": md["Barcode"].fillna("").astype(str).values[keep]}))
        print(s, int(keep.sum()), "/", keep.size, flush=True)
    return sp.vstack(mats).tocsr(), gn, pd.concat(meta, ignore_index=True)

def run(tag, X, genes, m, seed):
    rng = np.random.default_rng(seed)
    Xt = X  # already QC-filtered and thinned per sample (nuclear UMIs)
    keep = np.ones(X.shape[0], bool)
    samp = m["sample"].values
    fit = np.zeros(len(m), bool)
    for s in np.unique(samp):
        ii = np.where(samp == s)[0]; fit[rng.choice(ii, min(1000, int(0.3 * len(ii))), replace=False)] = True
    nbal = min((fit & (samp == s)).sum() for s in np.unique(samp))
    bal = np.concatenate([rng.choice(np.where(fit & (samp == s))[0], nbal, replace=False) for s in np.unique(samp)])
    Cb = Xt[bal]; p = np.asarray(Cb.sum(0)).ravel() / (T * len(bal)); ok = np.where(p > 0)[0]
    def resid(C, cols):
        M = T * p[cols]; R = (C[:, cols].toarray() - M) / np.sqrt(M + M**2 / THETA); c = np.sqrt(len(bal)); return np.clip(R, -c, c)
    rv = np.zeros(len(p))
    for k in range(0, len(ok), 3000):
        cols = ok[k:k + 3000]; rv[cols] = resid(Cb, cols).var(0)
    hv = np.argsort(rv)[::-1][:HVG]
    Z = np.vstack([resid(Xt[i:i + 5000], hv) for i in range(0, Xt.shape[0], 5000)]).astype(np.float32)
    pca = PCA(n_components=NPC, svd_solver="full", random_state=0).fit(Z[bal])
    PC = pca.transform(Z).astype(np.float32)
    extra = {k: m[k].values.astype(str) for k in m.columns if k not in ("sample", "clone")}
    np.savez_compressed(OUT / f"{tag}_pp.npz", pc=PC, sample=samp.astype(str), clone=m["clone"].values.astype(str), eval=~fit, **extra)
    s = dict(n_cells=int(len(m)), n_kept_of=int(keep.size), n_eval={s: int(((samp == s) & ~fit).sum()) for s in np.unique(samp)},
             n_clones=int(len(set(m["clone"]) - {""})), assigned=float((m["clone"] != "").mean()), evr=pca.explained_variance_ratio_[:5].round(4).tolist())
    json.dump(s, open(OUT / f"{tag}_pp_summary.json", "w"), indent=1); print(tag, s, flush=True)

if __name__ == "__main__":
    X, g, m = load_tnbc(); run("tnbc", X, g, m, 20261102)
    X, g, m = load_merlin(); run("merlin", X, g, m, 20261102)
