#!/usr/bin/env python3
"""Preprocess GSE230538 for collective-recurrence analysis.

Per cell line:
  1. union of genes over the four time points (absent genes = 0 counts)
  2. QC: nuclear (non-MT) UMIs >= T and mitochondrial fraction < mt_max
  3. down-sampling (multivariate hypergeometric thinning) of every cell to exactly T nuclear UMIs
     -> identical technical sampling depth across time points
  4. random split of each sample into a FIT set (HVG + PCA estimation) and an EVAL set
     (all recurrence distances are computed on EVAL cells only)
  5. HVG selection on a time-balanced FIT subset, z-scoring, PCA; projection of EVAL cells
  6. curated gene-program scores (Tirosh-style control-corrected module scores)
"""
from config import DATA  # data/results/figure folders, see config.py
import argparse, json, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import PCA

PROGRAMS = {
 "S_phase": "MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 MLF1IP CENPU HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8".split(),
 "G2M": "HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA".split(),
 "melanocytic": "MITF MLANA PMEL TYR TYRP1 DCT SLC45A2 GPR143 RAB27A IRF4 KIT ST3GAL6 CDK2 GPNMB TRPM1 SLC24A5 CAPN3 ERBB3".split(),
 "invasive": "AXL WNT5A EGFR TGFBI CYR61 CCN1 CTGF CCN2 LOXL2 NRP1 PDGFRB VEGFC ZEB1 FOSL2 SPARC COL1A1 FN1 TPM1 FSTL1 THBS1 NGFR".split(),
 "senescence": "CDKN1A CDKN2A CDKN2B GLB1 SERPINE1 IGFBP3 IGFBP7 GDF15 TP53INP1 IL6 CXCL8 MMP1 CCL2 TIMP1 PLAU SERPINB2".split(),
}

def load_sample(work, s):
    X = sp.load_npz(work / f"{s}.npz").tocsr()
    g = pd.read_csv(work / f"{s}_genes.csv", header=None)[0].astype(str).values
    return X, g

def thin(X, T, rng):
    """Multivariate-hypergeometric down-sampling of every row of integer matrix X to total T."""
    X = X.tocsr().astype(np.int64)
    data = np.empty_like(X.data)
    for i in range(X.shape[0]):
        a, b = X.indptr[i], X.indptr[i + 1]
        data[a:b] = rng.multivariate_hypergeometric(X.data[a:b], T, method="marginals")
    Y = sp.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    Y.eliminate_zeros()
    return Y

def module_score(L, genes, gset, rng, nbins=24, nctrl=50):
    """Tirosh et al. (2016) control-corrected module score on log-expression L (cells x genes, dense)."""
    idx = {g: k for k, g in enumerate(genes)}
    target = sorted({idx[g] for g in gset if g in idx})
    if len(target) < 3:
        return np.full(L.shape[0], np.nan), 0
    mu = L.mean(0)
    order = np.argsort(mu, kind="stable")
    bins = np.empty(len(mu), int); bins[order] = np.arange(len(mu)) * nbins // len(mu)
    ctrl = []
    for t in target:
        pool = np.where(bins == bins[t])[0]
        ctrl.extend(rng.choice(pool, size=min(nctrl, len(pool)), replace=False))
    ctrl = np.setdiff1d(np.unique(ctrl), target)
    return L[:, target].mean(1) - L[:, ctrl].mean(1), len(target)

def hvg_dispersion(L, n_top, min_frac=0.01, nbins=20):
    mean = L.mean(0); var = L.var(0, ddof=1)
    frac = (L > 0).mean(0)
    ok = (frac >= min_frac) & (mean > 0)
    disp = np.full(L.shape[1], -np.inf)
    ld = np.log(var[ok] / mean[ok])
    m = mean[ok]
    q = np.quantile(m, np.linspace(0, 1, nbins + 1)); q[-1] += 1e-12
    b = np.clip(np.searchsorted(q, m, side="right") - 1, 0, nbins - 1)
    z = np.zeros_like(ld)
    for k in range(nbins):
        w = b == k
        if w.sum() > 1:
            sd = ld[w].std(ddof=1)
            z[w] = (ld[w] - ld[w].mean()) / (sd if sd > 0 else 1)
    disp[np.where(ok)[0]] = z
    return np.argsort(disp)[::-1][:n_top]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=DATA + "/work")
    ap.add_argument("--meta", default=DATA + "/GSE230538_metadata.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--mt-max", type=float, default=0.25)
    ap.add_argument("--fit-frac", type=float, default=0.3)
    ap.add_argument("--fit-cap", type=int, default=500)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--npc", type=int, default=50)
    ap.add_argument("--method", default="pearson", choices=["pearson","lognorm"])
    ap.add_argument("--theta", type=float, default=100.0)
    ap.add_argument("--no-split", action="store_true", help="fit PCA on all cells (sensitivity)")
    ap.add_argument("--seed", type=int, default=20261025)
    a = ap.parse_args()
    work = pathlib.Path(a.work); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(a.meta)
    rng = np.random.default_rng(a.seed)
    summary = {}
    for line, ml in meta.groupby("cell_line", sort=False):
        ml = ml.sort_values("day")
        mats, gl = [], []
        for s in ml["sample"]:
            X, g = load_sample(work, s); mats.append(X); gl.append(g)
        genes = np.array(sorted(set().union(*map(set, gl))))
        gi = {g: k for k, g in enumerate(genes)}
        is_mt = np.array([g.upper().startswith("MT-") for g in genes])
        blocks, days, qc = [], [], []
        for X, g, (_, r) in zip(mats, gl, ml.iterrows()):
            cols = np.array([gi[x] for x in g])
            P = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), cols)), shape=(len(g), len(genes)))
            Xu = (X @ P).tocsr()
            umi = np.asarray(Xu.sum(1)).ravel()
            nuc = np.asarray(Xu[:, ~is_mt].sum(1)).ravel()
            mtf = 1 - nuc / np.maximum(umi, 1)
            keep = (nuc >= a.T) & (mtf < a.mt_max)
            Xn = Xu[keep][:, ~is_mt]
            Xt = thin(Xn, a.T, rng)
            blocks.append(Xt); days += [int(r["day"])] * Xt.shape[0]
            qc.append(dict(sample=r["sample"], day=int(r["day"]), n_raw=int(X.shape[0]), n_pass=int(keep.sum()),
                           median_nuclear_umi_pass=float(np.median(nuc[keep]))))
        genes_n = genes[~is_mt]
        Xall = sp.vstack(blocks).tocsr(); days = np.array(days)
        L = Xall.astype(np.float32).toarray()
        np.log1p(L * (1e4 / a.T), out=L)
        # split
        fit = np.zeros(len(days), bool)
        for d in np.unique(days):
            ii = np.where(days == d)[0]
            nf = min(a.fit_cap, int(round(a.fit_frac * len(ii))))
            fit[rng.choice(ii, nf, replace=False)] = True
        if a.no_split:
            fit[:] = True
        nbal = min((fit & (days == d)).sum() for d in np.unique(days))
        bal = np.concatenate([rng.choice(np.where(fit & (days == d))[0], nbal, replace=False) for d in np.unique(days)])
        if a.method == "lognorm":
            hv = hvg_dispersion(L[bal], a.hvg)
            Z = L[:, hv]
            mu, sd = Z[bal].mean(0), Z[bal].std(0, ddof=1); sd[sd == 0] = 1
            Z = np.clip((Z - mu) / sd, -10, 10)
        else:
            # analytic Pearson residuals (Lause, Berens & Kobak 2021); all cells have depth T
            Cb = Xall[bal]
            p = np.asarray(Cb.sum(0)).ravel() / (a.T * len(bal))
            ok = np.where(p > 0)[0]
            def resid(C, cols):
                M = a.T * p[cols]
                R = (C[:, cols].toarray() - M) / np.sqrt(M + M**2 / a.theta)
                c = np.sqrt(len(bal)); return np.clip(R, -c, c)
            rv = np.zeros(len(p))
            for k in range(0, len(ok), 4000):
                cols = ok[k:k+4000]; rv[cols] = resid(Cb, cols).var(0)
            hv = np.argsort(rv)[::-1][:a.hvg]
            Z = resid(Xall, hv).astype(np.float32)
        pca = PCA(n_components=a.npc, svd_solver="full", random_state=0).fit(Z[bal])
        PC = pca.transform(Z).astype(np.float32)
        sc = {}; ng = {}
        for k, gs in PROGRAMS.items():
            sc[k], ng[k] = module_score(L, genes_n, gs, rng)
        prog = pd.DataFrame(sc); prog["proliferation"] = (prog["S_phase"] + prog["G2M"]) / 2
        evalmask = ~fit if not a.no_split else np.ones(len(days), bool)
        tag = line.replace("/", "_")
        np.savez_compressed(out / f"{tag}.npz", pc=PC, day=days, eval=evalmask, fit=fit,
                            prog=prog[["proliferation", "melanocytic", "invasive", "senescence", "S_phase", "G2M"]].to_numpy(np.float32),
                            evr=pca.explained_variance_ratio_)
        # top loading genes of PC1/PC2 for interpretation
        load = pd.DataFrame(pca.components_[:5].T, index=genes_n[hv], columns=[f"PC{i+1}" for i in range(5)])
        tops = {c: {"pos": load[c].nlargest(12).index.tolist(), "neg": load[c].nsmallest(12).index.tolist()} for c in load}
        summary[line] = dict(qc=qc, n_genes_union=int(len(genes_n)), n_balanced_fit=int(nbal),
                             n_eval={int(d): int(((days == d) & evalmask).sum()) for d in np.unique(days)},
                             evr_first10=pca.explained_variance_ratio_[:10].round(4).tolist(),
                             program_genes_found=ng, pc_top_genes=tops)
        print(line, summary[line]["n_eval"], "EVR(1-5)", np.round(pca.explained_variance_ratio_[:5], 3), flush=True)
    json.dump(summary, open(out / "preprocess_summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
