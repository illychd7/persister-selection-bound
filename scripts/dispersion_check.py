"""Is the within-clone variance increase biological or technical?
(1) split along the common-response direction vs orthogonal (HVG space)
(2) recompute in the drug-stable 200-gene space and in expression-matched random 200-gene spaces
(3) replicate libraries: ratio between replicates at the same time point."""
from config import RESULTS  # data/results/figure folders, see config.py
import sys, json, numpy as np, pandas as pd
pass
import gene_panel_control as g
from decomposition import within_clone_shift
samp, clone = g.samp, g.z["clone"]
TRE = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
def wvar(Y, a, b, proj=None, nmin=4):
    c0, ct = clone[a], clone[b]; s0 = pd.Series(c0[c0 != ""]).value_counts(); st = pd.Series(ct[ct != ""]).value_counts()
    K = [c for c in st.index if c in s0.index and s0[c] >= nmin and st[c] >= nmin]; w = np.array([st[c] for c in K], float); w /= w.sum()
    Y0, Yt = Y[a], Y[b]
    if proj is not None: Y0, Yt = Y0 @ proj, Yt @ proj
    v0 = np.array([np.var(Y0[c0 == c], 0, ddof=1).sum() for c in K]); vt = np.array([np.var(Yt[ct == c], 0, ddof=1).sum() for c in K])
    return float((w @ vt) / (w @ v0))
out = {}
H = g.PCm[:, :20].astype(float); a0 = samp == "0"
for t in TRE:
    at = samp == t
    d, _ = within_clone_shift(H[a0], clone[a0], H[at], clone[at]); u = (d / np.linalg.norm(d))[:, None]
    P = np.eye(20) - u @ u.T
    out[t] = dict(HVG_total=wvar(H, a0, at), HVG_along_delta=wvar(H, a0, at, u), HVG_orthogonal=wvar(H, a0, at, P))
# gene spaces
mean_fit = np.vstack([np.asarray(g.Xt[g.fit & (samp == s)].mean(0)).ravel() for s in g.SAMPLES]); gmean = mean_fit.mean(0)
lfc = np.log2((mean_fit[1:] + 1e-3) / (mean_fit[0] + 1e-3)); maxabs = np.abs(lfc).max(0)
expressed = np.where(gmean >= 0.5)[0]; stable = expressed[np.argsort(maxabs[expressed])[:200]]
Ys, _ = g.embed(stable)
rng = np.random.default_rng(5); order = np.argsort(gmean); rank = np.empty_like(order); rank[order] = np.arange(len(order))
def matched(panel):
    pick = []
    for x in panel:
        lo, hi = max(rank[x] - 50, 0), min(rank[x] + 50, len(order)); cand = [c for c in order[lo:hi] if c not in panel and c not in pick]; pick.append(rng.choice(cand))
    return pick
Yr = [g.embed(matched(list(stable)))[0] for _ in range(5)]
for t in TRE:
    at = samp == t
    out[t]["STABLE200"] = wvar(Ys, a0, at); out[t]["RANDOM200_median"] = float(np.median([wvar(Y, a0, at) for Y in Yr]))
for a, b in [("3_rep1", "3_rep2"), ("7_rep1", "7_rep2"), ("14_rep1_high", "14_rep2_high")]:
    out[f"{b}/{a}"] = dict(HVG_total=wvar(H, samp == a, samp == b), STABLE200=wvar(Ys, samp == a, samp == b))
for k, v in out.items(): print(k, {kk: round(vv, 2) for kk, vv in v.items()})
json.dump(out, open(RESULTS + "/dispersion_check.json", "w"), indent=1)
