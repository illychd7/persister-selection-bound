"""Hyperparameter sensitivity of the plug-in competitors (MELD beta, UOT marginal relaxation reg_m, Milo-like k)
on one null (N2 selection + common) and one alternative (A2 clone-intrinsic 30%), 10 seeds each."""
from config import RESULTS  # data/results/figure folders, see config.py
import json, numpy as np, ot, meld
from scipy.stats import binom
from sklearn.neighbors import NearestNeighbors
import benchmark as b
out = []
for name, kw in [("N2 selection + common", dict()), ("A2 + clone-intrinsic 30%", dict(phi=0.30, kind="intrinsic"))]:
    for s in range(10):
        d = b.make(100 + s, **kw); rng = np.random.default_rng(s)
        X0e, Xte, B = d["X0e"], d["Xte"], d["B"]; delta = d["Xtf"].mean(0) - d["X0f"].mean(0)
        Xe = np.vstack([X0e + delta, Xte]); lab = np.r_[np.zeros(len(X0e)), np.ones(len(Xte))]
        row = dict(scenario=name, seed=s, B=B)
        for beta in [20, 60, 150, 400]:
            dens = np.asarray(meld.MELD(verbose=0, knn=10, beta=beta).fit_transform(Xe, sample_labels=np.where(lab == 1, "t", "c")))
            dens = dens / dens.sum(0); rm = dens[:, 1] / np.maximum(dens[:, 0], 1e-12)
            row[f"meld_beta{beta}"] = float(rm[lab == 1].max())
        a_i = rng.choice(len(X0e), 1500, replace=False); b_i = rng.choice(len(Xte), 1500, replace=False)
        C = ot.dist(X0e[a_i], Xte[b_i]); C /= np.median(C); a = np.full(1500, 1 / 1500)
        for rm_ in [0.03, 0.1, 0.3, 1.0, 3.0]:
            P = ot.unbalanced.sinkhorn_unbalanced(a, a, C, reg=0.05, reg_m=rm_, numItermax=2000)
            gr = P.sum(1) / a; gr = gr / gr.mean(); row[f"uot_regm{rm_}"] = float(gr.max())
        for k in [10, 30, 100]:
            nn = NearestNeighbors(n_neighbors=k).fit(Xe); idx = rng.choice(len(Xe), len(Xe) // 10, replace=False)
            nb = nn.kneighbors(Xe[idx], return_distance=False); nt_, n0_ = lab.sum(), (1 - lab).sum(); p0 = nt_ / (nt_ + n0_)
            ct = lab[nb].sum(1); fc = ((ct + .5) / nt_) / ((k - ct + .5) / n0_); pv = binom.sf(ct - 1, k, p0)
            o = np.argsort(pv); m = len(pv); ps = np.where(pv[o] <= .1 * np.arange(1, m + 1) / m)[0]; sig = np.zeros(m, bool)
            if len(ps): sig[o[: ps.max() + 1]] = True
            row[f"milo_k{k}"] = float(fc[sig].max()) if sig.any() else 1.0
        out.append(row); print(name, s, flush=True)
json.dump(out, open(RESULTS + "/sensitivity.json", "w"), indent=1)
import pandas as pd
df = pd.DataFrame(out); cols = [c for c in df.columns if c.startswith(("meld", "uot", "milo"))]
print((100 * pd.DataFrame({sc: {c: float((g[c] > g.B).mean()) for c in cols} for sc, g in df.groupby("scenario")})).round(0).to_string())
