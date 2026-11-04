#!/usr/bin/env python3
"""Gene-panel negative control for the snapshot bound M_L on PC9 / osimertinib (GSE150949).

Question: can library-level technical differences (each time point is a separate 10x library)
produce large M_L by themselves? We recompute M_L in feature spaces built from gene panels that
should carry little drug biology, using exactly the cells, thinning, fit/eval split and residual
model of pc9_preprocess.py (manuscript v4.2).

Panels
  HVG2000      the manuscript space (top-2000 residual-variance genes, 20 PCs) - reference
  HK11         a priori housekeeping panel: the 11 most uniformly expressed human genes of
               Eisenberg & Levanon (2013, Trends Genet 29:569)
  STABLE200    data-driven: 200 expressed genes with the smallest pseudobulk change across the
               11 samples, selected on FIT cells only and evaluated on held-out EVAL cells
  MAPK12       positive control: EGFR/MAPK output genes (expected to respond strongly)
  RANDOM(k)    40 random panels per control panel, matched on mean expression (fit cells),
               to show that a small panel still has power when genes do respond
  TECH         technical covariates only: genes detected at 3000 UMIs and mitochondrial fraction
"""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import sys, json, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import PCA
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from seltest import lower_bound

D = pathlib.Path(PC9); OUT = pathlib.Path(RESULTS)
T, THETA, Q = 3000, 100.0, 20
META = D / "GSE150949_metaData_with_lineage.txt.gz"

Xt = sp.load_npz(D / "pc9_thinned.npz").tocsr().astype(np.float32)
spl = np.load(D / "pc9_split.npz", allow_pickle=True)
fit, bal, genes, p = spl["fit"], spl["bal"], spl["genes"].astype(str), spl["p"]
z = np.load(D / "pc9_pp.npz", allow_pickle=True)
samp, ev, PCm = z["sample"], z["eval"], z["pc"]
gidx = {g: i for i, g in enumerate(genes)}
SAMPLES = ["0", "3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_low", "14_rep2_low",
           "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
TREATED = SAMPLES[1:]
REPS = [("3_rep1", "3_rep2"), ("7_rep1", "7_rep2"), ("14_rep1_low", "14_rep2_low"),
        ("14_rep1_med", "14_rep2_med"), ("14_rep1_high", "14_rep2_high")]

HK11 = ["C1orf43", "CHMP2A", "EMC7", "GPI", "PSMB2", "PSMB4", "RAB7A", "REEP5", "SNRPD3", "VCP", "VPS29"]
MAPK = ["DUSP6", "DUSP4", "ETV4", "ETV5", "SPRY4", "SPRY2", "EGR1", "FOSL1", "PHLDA1", "CCND1", "MYC", "IER3"]

def residuals(cols, rows=None):
    C = Xt if rows is None else Xt[rows]
    M = T * p[cols]; R = (C[:, cols].toarray() - M) / np.sqrt(M + M ** 2 / THETA)
    c = np.sqrt(len(bal)); return np.clip(R, -c, c)

def embed(cols):
    """Panel -> feature space, fitted on the balanced FIT cells (as in the manuscript)."""
    cols = np.asarray([c for c in cols if p[c] > 0])
    Z = np.vstack([residuals(cols, np.arange(i, min(i + 8000, Xt.shape[0]))) for i in range(0, Xt.shape[0], 8000)])
    if len(cols) <= Q:
        return Z.astype(float), len(cols)
    pca = PCA(n_components=Q, svd_solver="full", random_state=0).fit(Z[bal])
    return pca.transform(Z).astype(float), len(cols)

def bounds(Y, seeds=(0,), nseed_med=10):
    """M_L for every treated sample vs day 0, and for replicate pairs; EVAL cells only."""
    S = lambda s: Y[ev & (samp == s)]
    res = {}
    for s in TREATED:
        res[f"0->{s}"] = lower_bound(S("0"), S(s), np.random.default_rng(0))
        res[f"0->{s}_med10"] = float(np.median([lower_bound(S("0"), S(s), np.random.default_rng(k)) for k in range(nseed_med)]))
    for a, b in REPS:
        res[f"{a}->{b}"] = lower_bound(S(a), S(b), np.random.default_rng(0))
        res[f"{b}->{a}"] = lower_bound(S(b), S(a), np.random.default_rng(0))
    Y0 = S("0"); r = np.random.default_rng(99).permutation(len(Y0))
    res["0halfA->0halfB"] = lower_bound(Y0[r[: len(r) // 2]], Y0[r[len(r) // 2:]], np.random.default_rng(0))
    return res

def main():
    out = {}
    # ---- 0. reproduce the manuscript space exactly (PCs saved by pc9_preprocess.py)
    out["HVG2000"] = dict(n_genes=2000, **bounds(PCm[:, :Q].astype(float)))
    print("HVG2000", {k: round(v, 2) for k, v in out["HVG2000"].items() if not k.endswith("med10")}, flush=True)

    # ---- fit-cell pseudobulk (per sample) for gene selection and expression matching
    mean_fit = np.vstack([np.asarray(Xt[fit & (samp == s)].mean(0)).ravel() for s in SAMPLES])  # 11 x G
    gmean = mean_fit.mean(0)
    lfc = np.log2((mean_fit[1:] + 1e-3) / (mean_fit[0] + 1e-3))                                 # treated vs day 0
    maxabs = np.abs(lfc).max(0)

    # ---- 1. a priori housekeeping
    hk = [gidx[g] for g in HK11 if g in gidx]
    Y, ng = embed(hk); out["HK11"] = dict(n_genes=ng, genes=[genes[i] for i in hk], **bounds(Y))
    print("HK11", ng, {k: round(v, 2) for k, v in out["HK11"].items() if isinstance(v, float) and not k.endswith("med10")}, flush=True)

    # ---- 2. data-driven stable panel (fit cells only); expressed = mean >= 0.5 UMI/cell at 3000 UMIs
    expressed = np.where(gmean >= 0.5)[0]
    stable = expressed[np.argsort(maxabs[expressed])[:200]]
    Y, ng = embed(stable); out["STABLE200"] = dict(n_genes=ng, max_abs_log2fc_fit=float(maxabs[stable].max()), **bounds(Y))
    print("STABLE200", {k: round(v, 2) for k, v in out["STABLE200"].items() if isinstance(v, float) and not k.endswith("med10")}, flush=True)

    # ---- 3. positive control
    mk = [gidx[g] for g in MAPK if g in gidx]
    Y, ng = embed(mk); out["MAPK12"] = dict(n_genes=ng, genes=[genes[i] for i in mk], **bounds(Y, nseed_med=3))
    print("MAPK12", {k: round(v, 2) for k, v in out["MAPK12"].items() if isinstance(v, float) and not k.endswith("med10")}, flush=True)

    # ---- 4. expression-matched random panels (power reference for the small panels)
    rng = np.random.default_rng(2026)
    order = np.argsort(gmean); rank = np.empty_like(order); rank[order] = np.arange(len(order))
    def matched(panel):
        pick = []
        for g in panel:
            lo, hi = max(rank[g] - 50, 0), min(rank[g] + 50, len(order))
            cand = [c for c in order[lo:hi] if c not in panel and c not in pick]
            pick.append(rng.choice(cand))
        return pick
    for name, panel, nrep in [("HK11", hk, 40), ("STABLE200", list(stable), 20)]:
        vals = {s: [] for s in TREATED}; reps = []
        for r in range(nrep):
            Y, _ = embed(matched(panel))
            S = lambda s: Y[ev & (samp == s)]
            for s in TREATED: vals[s].append(lower_bound(S("0"), S(s), np.random.default_rng(0)))
            reps.append(lower_bound(S("3_rep1"), S("3_rep2"), np.random.default_rng(0)))
        out[f"RANDOM_matched_{name}"] = {s: dict(median=float(np.median(v)), q10=float(np.quantile(v, .1)), q90=float(np.quantile(v, .9)))
                                         for s, v in vals.items()}
        out[f"RANDOM_matched_{name}"]["3_rep1->3_rep2"] = dict(median=float(np.median(reps)))
        print(f"RANDOM_matched_{name}", {s: round(d["median"], 2) for s, d in out[f"RANDOM_matched_{name}"].items()}, flush=True)

    # ---- 5. technical covariates only (standardised): detected genes at 3000 UMIs, mito fraction (pre-thinning)
    X = sp.load_npz(D / "pc9_counts.npz").tocsr(); allg = np.array(open(D / "pc9_genes.txt").read().split("\n"))
    mt = np.array([g.startswith("MT-") for g in allg]); umi = np.asarray(X.sum(1)).ravel(); nuc = np.asarray(X[:, ~mt].sum(1)).ravel()
    keep = (nuc >= T) & (1 - nuc / np.maximum(umi, 1) < 0.2)
    mtf = (1 - nuc / np.maximum(umi, 1))[keep]; ndet = np.diff(Xt.indptr).astype(float)
    F = np.column_stack([ndet, mtf]); F = (F - F[bal].mean(0)) / F[bal].std(0)
    out["TECH2"] = dict(features=["genes_detected_at_3000UMI", "mito_fraction"], **bounds(F))
    out["TECH2"]["per_sample"] = {s: dict(genes_detected_median=float(np.median(ndet[samp == s])), mito_median=float(np.median(mtf[samp == s])))
                                  for s in SAMPLES}
    print("TECH2", {k: round(v, 2) for k, v in out["TECH2"].items() if isinstance(v, float) and not k.endswith("med10")}, flush=True)

    json.dump(out, open(OUT / "gene_panel_control.json", "w"), indent=1)
    print("saved")

if __name__ == "__main__":
    main()
