"""Semi-synthetic validation: pseudo time-t populations built from REAL day-0 PC9 cells with known
selection, common response and clone-specific plasticity. Pseudo day 0 and the source of pseudo day t
are disjoint halves of the barcoded day-0 cells (same library -> no batch effect)."""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import json, sys, numpy as np, pandas as pd
from decomposition import formal_test, clone_decomposition, within_clone_shift, Q
z = np.load(PC9 + "/pc9_pp.npz", allow_pickle=True)
Yall = z["pc"][:, :Q].astype(float); samp = z["sample"]; clone = z["clone"]
lab0 = (samp == "0") & (clone != ""); Y0, c0 = Yall[lab0], clone[lab0]
# real common response of day 3 (rep 1), used as the true translation
dreal, _ = within_clone_shift(Yall[samp == "0"], clone[samp == "0"], Yall[samp == "3_rep1"], clone[samp == "3_rep1"])
cl = np.unique(c0); cidx = {c: i for i, c in enumerate(cl)}; ci = np.array([cidx[c] for c in c0])
cm = np.array([Y0[ci == i].mean(0) for i in range(len(cl))])          # true baseline clone states
E_delta = dreal @ dreal

def scenario(name, seed, phi=0.0, kind=None, cell_noise=0.0, common=True, sigma_g=0.5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y0)); ref, src = perm[: len(perm) // 2], perm[len(perm) // 2:]
    g = np.exp(sigma_g * rng.standard_normal(len(cl)))
    pr = g[ci[src]]; pr = pr / pr.sum(); nt = int(1.5 * len(src))              # expansion: sampling with replacement
    pick = src[rng.choice(len(src), nt, replace=True, p=pr)]
    Yt = Y0[pick] + 1e-6 * rng.standard_normal((nt, Q))
    if common: Yt += dreal
    if phi > 0:
        if kind == "intrinsic":
            r = rng.standard_normal((len(cl), Q))
        else:                                                            # smooth function of baseline clone state
            U = rng.standard_normal((Q, Q)) / np.sqrt(Q); W = rng.standard_normal((Q, Q))
            r = np.tanh((cm - cm.mean(0)) @ U / cm.std()) @ W.T
        wt = np.bincount(ci[pick], minlength=len(cl)) / nt
        r = r - wt @ r                                                   # zero weighted mean: pure clone-specific
        r *= np.sqrt(phi / (1 - phi) * E_delta / (wt @ (r ** 2).sum(1)))
        Yt += r[ci[pick]]
    if cell_noise > 0: Yt += cell_noise * rng.standard_normal(Yt.shape)
    Y = np.vstack([Y0[ref], Yt]); s = np.array(["0"] * len(ref) + ["t"] * nt); c = np.concatenate([c0[ref], c0[pick]])
    ev = rng.random(len(Y)) > 0.25
    # realised selection strength: max clone enrichment among clones with >=5 pseudo-day-0 cells
    n0 = pd.Series(c0[ref]).value_counts(); ntc = pd.Series(c0[pick]).value_counts()
    big = n0[n0 >= 5].index; enr = (ntc.reindex(big).fillna(0) / ntc.sum()) / (n0[big] / n0.sum())
    A = formal_test(Y, s, ev, c, "t")
    B = clone_decomposition(Y[s == "0"], c[s == "0"], Y[s == "t"], c[s == "t"], seed=seed, nmin=3, min_clones=10)
    return dict(scenario=name, seed=seed, true_phi=phi, kind=kind, cell_noise=cell_noise, max_clone_enrichment=float(enr.max()),
                formal=A, clone=B)

SC = [("S0 selection only", dict(common=False)),
      ("S1 selection + common", dict()),
      ("S2 + clone-intrinsic 5%", dict(phi=0.05, kind="intrinsic")),
      ("S3 + clone-intrinsic 30%", dict(phi=0.30, kind="intrinsic")),
      ("S4 + state-dependent 30%", dict(phi=0.30, kind="state")),
      ("S5 + cell-level noise", dict(cell_noise=3.0))]
out = []
for name, kw in SC:
    for seed in range(4):
        r = scenario(name, seed, **kw); out.append(r)
        f, c = r["formal"], r["clone"]
        if c.get("skipped"): print(name, seed, "skipped", c); continue
        print(f"{name:26s} s{seed} maxenr={r['max_clone_enrichment']:.1f} | id={f['identity']:.2f} trans={f['F_trans']:.2f} aff={f['F_affine']:.2f} clone={f['F_clone']:.2f} | "
              f"selshare={c['share_selection']:.3f} common={c['frac_common']:.3f} spec={c['frac_clone_specific']:.3f} p={c['p_clone_specific']:.3f} "
              f"stateFrac={c['frac_specific_state_dependent']:.2f} pstate={c['p_state_dependent']:.3f} varratio={c['within_clone_var_ratio']:.2f} nK={c['n_clones']}", flush=True)
json.dump(out, open(RESULTS + "/semisynthetic.json", "w"), indent=1)
