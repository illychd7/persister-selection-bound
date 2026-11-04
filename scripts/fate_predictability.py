"""Is clonal fate readable from the baseline transcriptome?
Target: clone log-enrichment at time t (pseudocount 0.5) for clones with >=5 baseline cells.
Predictor: baseline clone centroid (20 PCs). Ridge, 10-fold CV, R^2; null by permuting targets.
Noise ceiling: split the baseline cells of each clone in halves, so the predictor noise is visible;
a Poisson-noise ceiling for the target is estimated by parametric resampling of counts."""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import json, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
z = np.load(PC9 + "/pc9_pp.npz", allow_pickle=True)
Y = z["pc"][:, :20].astype(float); samp = z["sample"]; clone = z["clone"]
rng = np.random.default_rng(3)
a0 = samp == "0"; c0 = clone[a0]; n0 = pd.Series(c0[c0 != ""]).value_counts(); K = n0[n0 >= 5].index
X = np.array([Y[a0][c0 == c].mean(0) for c in K])
def r2(yhat, y): return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
out = {}
for t in ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_high", "14_rep2_high", "14_rep1_low", "14_rep2_low"]:
    ct = clone[samp == t]; nt = pd.Series(ct[ct != ""]).value_counts()
    cnt = np.array([nt.get(c, 0) for c in K], float); base = n0[K].values.astype(float)
    y = np.log((cnt + .5) / cnt.sum()) - np.log((base + .5) / base.sum())
    m = RidgeCV(alphas=np.logspace(-1, 4, 20)); cv = KFold(10, shuffle=True, random_state=0)
    obs = r2(cross_val_predict(m, X, y, cv=cv), y)
    null = [r2(cross_val_predict(m, X, yp, cv=cv), yp) for yp in (rng.permutation(y) for _ in range(100))]
    # target-noise ceiling: R^2 between two independent Poisson realisations of the same expected counts
    lam_t, lam_0 = (cnt + .5), (base + .5)
    ys = [np.log(rng.poisson(lam_t) + .5) - np.log(rng.poisson(lam_0) + .5) for _ in range(2)]
    ceil = float(np.corrcoef(ys[0], ys[1])[0, 1])
    # projection of the fate signal onto the common-response direction
    out[t] = dict(n_clones=len(K), cv_R2=float(obs), null_q95=float(np.quantile(null, .95)), p=float((1 + np.sum(np.array(null) >= obs)) / 101),
                  frac_extinct=float(np.mean(cnt == 0)), target_reliability=ceil)
    print(t, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[t].items()}, flush=True)
json.dump(out, open(RESULTS + "/fate_predictability.json", "w"), indent=1)
