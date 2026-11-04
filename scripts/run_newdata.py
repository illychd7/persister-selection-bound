#!/usr/bin/env python3
"""Response-adjusted bounds and clone decomposition on the two new datasets, using the
manuscript's own functions (seltest.lower_bound, decomposition.*). Bounds follow main_table.py:
FIT-estimated maps, EVAL cells, rng(0), familywise alpha within each dataset."""
from config import NEWDATA, RESULTS  # data/results/figure folders, see config.py
import sys, json, numpy as np, pandas as pd
pass; pass
from seltest import lower_bound
from decomposition import clone_decomposition, within_clone_shift, map_affine, Q
OUT = {}

def bounds(Y, samp, ev, ref, t, A):
    Y0f, Ytf = Y[~ev & (samp == ref)], Y[~ev & (samp == t)]
    Y0e, Yte = Y[ev & (samp == ref)], Y[ev & (samp == t)]
    d = Ytf.mean(0) - Y0f.mean(0); f_aff, _ = map_affine(Y0f, Ytf)
    u = d / np.linalg.norm(d)
    k = np.sqrt(np.var((Ytf - Ytf.mean(0)) @ u) / np.var((Y0f - Y0f.mean(0)) @ u))
    stretch = lambda X: Ytf.mean(0) + (X - Y0f.mean(0)) + (k - 1) * np.outer((X - Y0f.mean(0)) @ u, u)
    seeds = lambda f: [lower_bound(f(Y0e), Yte, np.random.default_rng(s), alpha=A) for s in range(5)]
    r = dict(n0=int(len(Y0e)), nt=int(len(Yte)),
             identity=lower_bound(Y0e, Yte, np.random.default_rng(0), alpha=A),
             translation=lower_bound(Y0e + d, Yte, np.random.default_rng(0), alpha=A),
             axis_stretch=lower_bound(stretch(Y0e), Yte, np.random.default_rng(0), alpha=A), stretch_k=float(k),
             affine=lower_bound(f_aff(Y0e), Yte, np.random.default_rng(0), alpha=A),
             loss_identity=lower_bound(Yte, Y0e, np.random.default_rng(0), alpha=A))
    r["translation_5seeds"] = seeds(lambda X: X + d)
    return r

def floor_S1(Y, samp, clone, ref, t, seeds=range(4)):
    """semisynthetic.py scenario S1 (selection + real common response) on this dataset's baseline cells."""
    lab0 = (samp == ref) & (clone != ""); Y0, c0 = Y[lab0], clone[lab0]
    dreal, _ = within_clone_shift(Y[samp == ref], clone[samp == ref], Y[samp == t], clone[samp == t])
    cl = np.unique(c0); ci = np.searchsorted(cl, c0); out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Y0)); rf, src = perm[: len(perm) // 2], perm[len(perm) // 2:]
        g = np.exp(0.5 * rng.standard_normal(len(cl))); pr = g[ci[src]]; pr /= pr.sum(); nt = int(1.5 * len(src))
        pick = src[rng.choice(len(src), nt, replace=True, p=pr)]
        Yt = Y0[pick] + 1e-6 * rng.standard_normal((nt, Y.shape[1])) + dreal
        r = clone_decomposition(Y0[rf], c0[rf], Yt, c0[pick], seed=seed, nmin=3, min_clones=10)
        out.append(r.get("frac_clone_specific", np.nan))
    return out

def decomp(Y, samp, clone, ref, t):
    a0, at = samp == ref, samp == t
    r = clone_decomposition(Y[a0], clone[a0], Y[at], clone[at], seed=1)
    keys = ("n_clones", "skipped", "share_selection", "share_selection_incl_extinction", "frac_baseline_clones_extinct", "frac_common",
            "frac_clone_specific", "frac_specific_state_dependent", "p_clone_specific", "p_state_dependent", "within_clone_var_ratio")
    return {k: r[k] for k in keys if k in r}

# ------------------------------------------------------------------ TNBC
z = np.load(NEWDATA + "/tnbc_pp.npz", allow_pickle=True)
Y = z["pc"][:, :Q].astype(float); samp, clone, ev = z["sample"], z["clone"], z["eval"]
REF = {"30HR": "PT1", "250_4": "PT1", "250_6": "PT1", "250_1": "PT2", "400_2": "PT2", "400_4": "PT2", "400_5": "PT2"}
A = 0.05 / (2 * len(REF) + 2); tn = {}
for t, ref in REF.items():
    tn[t] = dict(reference=ref, **bounds(Y, samp, ev, ref, t, A), clone=decomp(Y, samp, clone, ref, t))
    print("TNBC", t, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tn[t].items() if k not in ("clone", "translation_5seeds")}, tn[t]["clone"], flush=True)
tn["technical"] = {"PT1->PT2": lower_bound(Y[ev & (samp == "PT1")], Y[ev & (samp == "PT2")], np.random.default_rng(0), alpha=A),
                   "PT2->PT1": lower_bound(Y[ev & (samp == "PT2")], Y[ev & (samp == "PT1")], np.random.default_rng(0), alpha=A)}
tn["floor_S1_30HR"] = floor_S1(Y, samp, clone, "PT1", "30HR")
# share of the 30HR clone-specific energy carried by the two ESAM-defined subpopulations
cc = z["clone_cluster"]
print("TNBC technical", tn["technical"], "floor", np.round(tn["floor_S1_30HR"], 3), flush=True)
OUT["TNBC"] = tn
# ------------------------------------------------------------------ MeRLin
z = np.load(NEWDATA + "/merlin_pp.npz", allow_pickle=True)
Y = z["pc"][:, :Q].astype(float); samp, clone, ev = z["sample"], z["clone"], z["eval"]
A = 0.05 / 4
mr = dict(**bounds(Y, samp, ev, "CTRL", "DT", A), clone=decomp(Y, samp, clone, "CTRL", "DT"))
Y0 = Y[ev & (samp == "CTRL")]; h = np.random.default_rng(99).permutation(len(Y0))
mr["control_halves"] = lower_bound(Y0[: len(h) // 2][h[: len(h) // 2] < len(Y0)] if False else Y0[h[: len(h) // 2]], Y0[h[len(h) // 2:]], np.random.default_rng(0), alpha=A)
mr["floor_S1"] = floor_S1(Y, samp, clone, "CTRL", "DT")
# dispersion along / orthogonal to the within-clone response axis (dispersion_check.py logic)
a0, at = samp == "CTRL", samp == "DT"
d, _ = within_clone_shift(Y[a0], clone[a0], Y[at], clone[at]); u = d / np.linalg.norm(d); P = np.eye(Q) - np.outer(u, u)
tr = lambda Z: float(np.var(Z, 0, ddof=1).sum())
mr["pop_var_ratio_along"] = float(np.var(Y[at] @ u, ddof=1) / np.var(Y[a0] @ u, ddof=1))
mr["pop_var_ratio_orth"] = tr(Y[at] @ P) / tr(Y[a0] @ P)
mr["budget"] = {f"d{dd}": [2 ** dd / 0.3, 2 ** dd / 0.2] for dd in range(4)}
print("MeRLin", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in mr.items() if k not in ("clone", "budget", "translation_5seeds")}, mr["clone"], np.round(mr["floor_S1"], 3), flush=True)
OUT["MeRLin"] = mr
json.dump(OUT, open(RESULTS + "/newdata_results.json", "w"), indent=1, default=float)
print("saved")
