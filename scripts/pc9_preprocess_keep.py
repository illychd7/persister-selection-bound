#!/usr/bin/env python3
"""PC9 / osimertinib (GSE150949): QC, thinning to T nuclear UMIs, clone calling, held-out Pearson-residual PCA."""
from config import PC9  # data/results/figure folders, see config.py
import sys, json, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import PCA
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from preprocess import thin

T, MTMAX, HVG, NPC, THETA = 3000, 0.2, 2000, 50, 100.0
D = pathlib.Path(PC9)
META = D / "GSE150949_metaData_with_lineage.txt.gz"
rng = np.random.default_rng(20261026)

X = sp.load_npz(D / "pc9_counts.npz").tocsr(); genes = np.array(open(D / "pc9_genes.txt").read().split("\n"))
m = pd.read_csv(META, sep="\t")
mt = np.array([x.startswith("MT-") for x in genes])
umi = np.asarray(X.sum(1)).ravel(); nuc = np.asarray(X[:, ~mt].sum(1)).ravel(); mtf = 1 - nuc / np.maximum(umi, 1)
keep = (nuc >= T) & (mtf < MTMAX)
m = m[keep].copy(); Xn = X[keep][:, ~mt]; genes = genes[~mt]

# ---- clone calling: link two barcodes only if they co-occur in >=2 cells and in >=50% of the rarer one;
#      cells with >=4 barcodes (likely doublets) or barcodes spanning several clones are left unassigned
import collections
bl = m.lineage_barcode.apply(lambda s: s.split(",") if isinstance(s, str) else [])
cnt = collections.Counter(x for l in bl for x in l)
co = collections.Counter()
for l in bl:
    if 2 <= len(l) <= 3:
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                co[tuple(sorted((l[i], l[j])))] += 1
parent = {x: x for x in cnt}
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for (a_, b_), w in co.items():
    if w >= 2 and w >= 0.5 * min(cnt[a_], cnt[b_]):
        ra, rb = find(a_), find(b_)
        if ra != rb: parent[ra] = rb
def assign(l):
    if len(l) == 0 or len(l) >= 4: return None
    roots = {find(x) for x in l}
    if len(roots) == 1: return roots.pop()
    # several roots: keep the cell only if one root carries a strict majority of its barcodes' total counts
    w = collections.Counter()
    for x in l: w[find(x)] += cnt[x]
    (r1, w1), (r2, w2) = w.most_common(2)
    return r1 if w1 >= 3 * w2 else None
m["clone"] = bl.apply(assign)
sizes = m.clone.value_counts()
print("clones:", len(sizes), "largest:", sizes.head(5).tolist(), "assigned frac:", round(m.clone.notna().mean(), 3))

# ---- thinning
Xt = thin(Xn, T, rng)
# ---- fit / eval split, balanced fit subset across the 11 samples
samp = m.sample_name.values
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
np.savez_compressed(D / "pc9_pp.npz", pc=PC, sample=samp.astype(str), time=m.time_point.values.astype(int),
                    clone=m.clone.fillna("").values.astype(str), eval=~fit, cell=m.index.values.astype(str))
Xs = Xt.astype(np.int32); Xs.data = Xs.data.astype(np.int16)
sp.save_npz(D / "pc9_thinned.npz", Xs)
np.savez_compressed(D / "pc9_split.npz", fit=fit, bal=bal, hv=hv, p=p, genes=genes)
load = pd.Series(pca.components_[0], index=genes[hv])
json.dump(dict(n_cells=int(len(m)), n_clones=int(len(sizes)), evr=pca.explained_variance_ratio_[:10].round(4).tolist(),
               n_eval={s: int(((samp == s) & ~fit).sum()) for s in np.unique(samp)},
               pc1_pos=load.nlargest(15).index.tolist(), pc1_neg=load.nsmallest(15).index.tolist()),
          open(D / "pc9_pp_summary.json", "w"), indent=1)
print("done", PC.shape, "EVR", pca.explained_variance_ratio_[:5].round(3))
