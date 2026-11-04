#!/usr/bin/env python3
"""Known-response simulation (Section 4.1, Fig. 1).

Bounded population with known state-dependent selection followed by a nonlinear common response.

  baseline       x ~ Uniform([-1, 1]^q)                      (q = 5)
  selection      g(x) = exp(beta <w, x>), w a random unit vector per run
                 true selective enrichment  M_sel = sup g / E g
                   = exp(beta ||w||_1) / prod_j sinh(beta w_j) / (beta w_j)   (exact)
  time-t cells   exact draws from S_g mu by rejection sampling, then
  response       f(x)_j = c_j + a_j x_j + b x_j^3   (a_j > 0, b > 0: injective, nonlinear)
                 |c| = 1 (shift of one unit in a random direction), a_j ~ U(0.8, 1.25), b = 0.2

Three reference models for the bound M_L(nu | mu K) (seltest.lower_bound, alpha = 0.05):
  ignore response     K = identity
  translation         K = x + delta_hat, delta_hat estimated on independent FIT samples
                      (misspecified: f stretches and bends each axis)
  correct response    K = f (the true map)
By Theorem 2.1 with injective f, M*(nu | f#mu) = M_sel, so only the correct model keeps the
lower-bound interpretation.

Usage: python sim_fig1.py  -> sim_fig1.json, sim_fig1.csv
"""
from config import RESULTS  # data/results/figure folders, see config.py
import sys, json, numpy as np, pandas as pd
pass
from seltest import lower_bound

Q, BETA, N0, NT, NFIT, RUNS = 5, 0.35, 4000, 4000, 2000, 120


def m_sel(w, beta=BETA):
    b = beta * w
    norm = np.prod(np.where(np.abs(b) < 1e-12, 1.0, np.sinh(b) / np.where(b == 0, 1, b)))
    return float(np.exp(np.abs(b).sum()) / norm)


def draw_selected(n, w, rng):
    """Exact draws from S_g mu by rejection: accept x with probability g(x) / sup g."""
    out, need = [], n
    lmax = BETA * np.abs(w).sum()
    while need > 0:
        x = rng.uniform(-1, 1, (2 * need + 100, Q))
        keep = rng.random(len(x)) < np.exp(BETA * x @ w - lmax)
        out.append(x[keep][:need]); need -= len(out[-1])
    return np.vstack(out)


def run(seed):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(Q); w /= np.linalg.norm(w)
    u = rng.standard_normal(Q); u /= np.linalg.norm(u)
    c, a, b = 1.0 * u, rng.uniform(0.8, 1.25, Q), 0.2
    f = lambda x: c + a * x + b * x ** 3
    Y0e = rng.uniform(-1, 1, (N0, Q)); Yte = f(draw_selected(NT, w, rng))
    Y0f = rng.uniform(-1, 1, (NFIT, Q)); Ytf = f(draw_selected(NFIT, w, rng))
    delta = Ytf.mean(0) - Y0f.mean(0)
    return dict(seed=seed, true=m_sel(w),
                ignore=lower_bound(Y0e, Yte, np.random.default_rng(seed)),
                translation=lower_bound(Y0e + delta, Yte, np.random.default_rng(seed)),
                correct=lower_bound(f(Y0e), Yte, np.random.default_rng(seed)))


if __name__ == "__main__":
    df = pd.DataFrame([run(s) for s in range(RUNS)])
    df.to_csv(RESULTS + "/sim_fig1.csv", index=False)
    summ = {"runs": RUNS, "median_true": float(df.true.median())}
    for m in ["ignore", "translation", "correct"]:
        summ[m] = dict(median=float(df[m].median()), q05=float(df[m].quantile(.05)), q95=float(df[m].quantile(.95)),
                       frac_exceeding_true=float((df[m] > df.true).mean()))
    json.dump(summ, open(RESULTS + "/sim_fig1.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
