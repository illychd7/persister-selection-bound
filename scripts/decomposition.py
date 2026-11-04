#!/usr/bin/env python3
"""Three-term decomposition of drug-induced state change: selection, common response, clone-specific plasticity.

Model (null hierarchy). mu = baseline state distribution, nu = time-t distribution.
  H_S    : nu = g.mu                      (pure selection; tested in manuscript v4.2)
  H_SC   : nu = f#(g.mu), f in F          (selection + a response f shared by all cells)
For injective f, f#(g.mu) has density g(f^{-1}y) w.r.t. f#mu, hence
  M*(nu || f#mu) = M*(g.mu || mu) <= sup g  <=  selection budget          (Prop. "common response")
so a bound M_L(nu || f_hat#mu) above the selection budget rejects H_SC for that response class.

Part A (formal): f_hat estimated on FIT cells only, bound computed on held-out EVAL cells.
  F_trans  : translation by the population mean shift
  F_affine : Gaussian optimal-transport (moment-matching) affine map  - the most generous affine response
  F_clone  : translation by the within-clone mean shift (barcodes; sensitivity analysis, see note)
Part B (clone level, all barcoded cells, descriptive + permutation tests):
  mean shift  = composition (selection between clones) + within-clone response          (exact identity)
  response    = common translation + clone-specific deviation   (energy; split-half, noise-free)
  clone-specific = state-dependent (predicted from neighbouring clones' baseline state) + clone-intrinsic
"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.linalg import sqrtm
pass
from seltest import lower_bound

Q = 20; NMIN = 4

# ----------------------------------------------------------------------------- common-response maps
def map_translation(Y0, Yt):
    d = Yt.mean(0) - Y0.mean(0); return lambda X: X + d, d

def map_affine(Y0, Yt, ridge=1e-6):
    m0, mt = Y0.mean(0), Yt.mean(0)
    S0 = np.cov(Y0.T) + ridge * np.eye(Y0.shape[1]); St = np.cov(Yt.T) + ridge * np.eye(Y0.shape[1])
    r0 = np.real(sqrtm(S0)); ir0 = np.linalg.inv(r0)
    A = ir0 @ np.real(sqrtm(r0 @ St @ r0)) @ ir0                  # Gaussian OT map: A S0 A = St, A symmetric PSD
    return lambda X: mt + (X - m0) @ A.T, A

def within_clone_shift(Y0, c0, Yt, ct, nmin=2):
    s0 = pd.Series(c0[c0 != ""]).value_counts(); st = pd.Series(ct[ct != ""]).value_counts()
    sh = [c for c in st.index if c in s0.index and s0[c] >= nmin and st[c] >= nmin]
    d = np.array([Yt[ct == c].mean(0) - Y0[c0 == c].mean(0) for c in sh]); w = np.array([st[c] for c in sh], float)
    return (w[:, None] * d).sum(0) / w.sum(), len(sh)

# ----------------------------------------------------------------------------- Part A
def formal_test(Y, samp, ev, clone, t, seed=0):
    fitc, evc = ~ev, ev
    Y0f, Ytf = Y[fitc & (samp == "0")], Y[fitc & (samp == t)]
    Y0e, Yte = Y[evc & (samp == "0")], Y[evc & (samp == t)]
    out = {"n0_eval": int(len(Y0e)), "nt_eval": int(len(Yte))}
    out["identity"] = lower_bound(Y0e, Yte, np.random.default_rng(seed))
    f, d = map_translation(Y0f, Ytf); out["F_trans"] = lower_bound(f(Y0e), Yte, np.random.default_rng(seed)); out["|delta_pop|"] = float(np.linalg.norm(d))
    f, A = map_affine(Y0f, Ytf); out["F_affine"] = lower_bound(f(Y0e), Yte, np.random.default_rng(seed))
    ev_ = np.linalg.eigvalsh((A + A.T) / 2); out["affine_eig_range"] = [float(ev_.min()), float(ev_.max())]
    # barcode-based translation (uses all barcoded cells -> overlaps with eval cells; sensitivity analysis)
    a0, at = samp == "0", samp == t
    dw, nsh = within_clone_shift(Y[a0], clone[a0], Y[at], clone[at])
    out["F_clone"] = lower_bound(Y0e + dw, Yte, np.random.default_rng(seed)); out["|delta_within|"] = float(np.linalg.norm(dw)); out["n_clones_shift"] = nsh
    return out

# ----------------------------------------------------------------------------- Part B
def clone_decomposition(Y0, c0, Yt, ct, nsplit=50, k=10, nperm=200, seed=0, nmin=NMIN, min_clones=15):
    rng = np.random.default_rng(seed)
    s0 = pd.Series(c0[c0 != ""]).value_counts(); st = pd.Series(ct[ct != ""]).value_counts()
    K = [c for c in st.index if c in s0.index and s0[c] >= nmin and st[c] >= nmin]
    if len(K) < min_clones: return {"n_clones": len(K), "skipped": True}
    i0 = {c: np.where(c0 == c)[0] for c in K}; it = {c: np.where(ct == c)[0] for c in K}
    n0 = np.array([len(i0[c]) for c in K], float); nt = np.array([len(it[c]) for c in K], float)
    p0, pt = n0 / n0.sum(), nt / nt.sum()
    M0 = np.array([Y0[i0[c]].mean(0) for c in K]); Mt = np.array([Yt[it[c]].mean(0) for c in K])
    # (1) mean shift = composition + within-clone response (exact)
    Delta = pt @ Mt - p0 @ M0; Sel = (pt - p0) @ M0; Resp = pt @ (Mt - M0)
    nD = Delta @ Delta
    res = dict(n_clones=len(K), n_cells_t=int(nt.sum()), mean_shift=float(np.sqrt(nD)),
               share_selection=float(Sel @ Delta / nD), share_response=float(Resp @ Delta / nD),
               norm_selection=float(np.linalg.norm(Sel)), norm_response=float(np.linalg.norm(Resp)))
    # (1b) selection including clone extinction: baseline reweighted by clone frequency ratio (clones absent at t -> 0)
    lab0 = c0 != ""; all0 = pd.Series(c0[lab0]).value_counts(); labt = ct != ""
    ft = pd.Series(ct[labt & np.isin(ct, all0.index)]).value_counts()
    w = np.array([ft.get(c, 0) / all0[c] for c in c0[lab0]]); w = w / w.sum()
    Sel_ext = w @ Y0[lab0] - Y0[lab0].mean(0); Delta_all = Yt[labt].mean(0) - Y0[lab0].mean(0)
    res["share_selection_incl_extinction"] = float(Sel_ext @ Delta_all / (Delta_all @ Delta_all))
    res["frac_baseline_clones_extinct"] = float(np.mean([ft.get(c, 0) == 0 for c in all0.index[all0 >= 3]]))
    # (2)+(3) split-half energies; nulls use the same splits with clone-pairing / baseline-position permutations
    splits = []
    for _ in range(nsplit):
        A0, B0, At, Bt = {}, {}, {}, {}
        for c in K:
            a = rng.permutation(i0[c]); b = rng.permutation(it[c])
            A0[c], B0[c] = a[: len(a) // 2], a[len(a) // 2:]; At[c], Bt[c] = b[: len(b) // 2], b[len(b) // 2:]
        m0a = np.array([Y0[A0[c]].mean(0) for c in K]); m0b = np.array([Y0[B0[c]].mean(0) for c in K])
        mta = np.array([Yt[At[c]].mean(0) for c in K]); mtb = np.array([Yt[Bt[c]].mean(0) for c in K])
        da, db = mta - m0a, mtb - m0b; dela, delb = pt @ da, pt @ db
        D = ((m0a[:, None, :] - m0a[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        splits.append((dela, delb, da - dela, db - delb, np.argsort(D, 1)[:, :k]))
    def stats(perm_pair=None, perm_pos=None):
        Ec, Es, Est = [], [], []
        for dela, delb, ra, rb, nn0 in splits:
            rb_ = rb if perm_pair is None else rb[perm_pair]
            if perm_pos is None: nn = nn0
            else:                                    # neighbours after permuting baseline positions among clones
                inv = np.empty_like(perm_pos); inv[perm_pos] = np.arange(len(perm_pos)); nn = inv[nn0[perm_pos]]
            Ec.append(dela @ delb); Es.append(pt @ (ra * rb_).sum(1)); Est.append(pt @ (ra[nn].mean(1) * rb).sum(1))
        return np.median(Ec), np.median(Es), np.median(Est), Es
    Ec, Es, Est, Es_all = stats()
    nul_s = [stats(perm_pair=rng.permutation(len(K)))[1] for _ in range(nperm)]
    nul_t = [stats(perm_pos=rng.permutation(len(K)))[2] for _ in range(nperm)]
    tot = Ec + max(Es, 0)
    res.update(E_common=float(Ec), E_clone_specific=float(Es), E_state_dependent=float(Est),
               frac_common=float(Ec / tot), frac_clone_specific=float(max(Es, 0) / tot),
               frac_specific_state_dependent=float(np.clip(Est / Es, 0, 1)) if Es > 0 else float("nan"),
               p_clone_specific=float((1 + np.sum(np.array(nul_s) >= Es)) / (1 + nperm)),
               p_state_dependent=float((1 + np.sum(np.array(nul_t) >= Est)) / (1 + nperm)),
               E_spec_q10_q90=[float(np.quantile(Es_all, .1)), float(np.quantile(Es_all, .9))])
    # (4) within-clone dispersion (mean per-clone trace covariance)
    v0 = np.array([np.var(Y0[i0[c]], 0, ddof=1).sum() for c in K]); vt = np.array([np.var(Yt[it[c]], 0, ddof=1).sum() for c in K])
    res["within_clone_var_ratio"] = float((pt @ vt) / (pt @ v0))
    return res
