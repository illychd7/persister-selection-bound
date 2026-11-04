"""Direction check: the novelty bound counts only time-t mass in regions far from baseline.
Down-regulated panels (MAPK outputs under EGFR inhibition) move cells into the dense low-count
region and should show up in the loss direction M_L(mu|nu) instead. Also: a fit-selected
up-regulated 12-gene panel as a small-panel positive control (selected on FIT, tested on EVAL)."""
from config import RESULTS  # data/results/figure folders, see config.py
import sys, json, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gene_panel_control as g   # reuses data loading and embed(); module runs its main block only via exec guard below
from seltest import lower_bound
S = lambda Y, s: Y[g.ev & (g.samp == s)]
mean_fit = np.vstack([np.asarray(g.Xt[g.fit & (g.samp == s)].mean(0)).ravel() for s in g.SAMPLES])
lfc3 = np.log2((mean_fit[1:3].mean(0) + 1e-3) / (mean_fit[0] + 1e-3))
expressed = mean_fit.mean(0) >= 0.5
up = np.where(expressed)[0][np.argsort(-lfc3[expressed])[:12]]
down = np.where(expressed)[0][np.argsort(lfc3[expressed])[:12]]
mk = [g.gidx[x] for x in g.MAPK if x in g.gidx]
out = {"lfc3_MAPK": {g.genes[i]: round(float(lfc3[i]), 2) for i in mk},
       "UP12_genes": [g.genes[i] for i in up], "DOWN12_genes": [g.genes[i] for i in down]}
for name, panel in [("MAPK12", mk), ("UP12_fit", list(up)), ("DOWN12_fit", list(down))]:
    Y, _ = g.embed(panel); r = {}
    for s in ["3_rep1", "3_rep2", "7_rep1", "14_rep1_high"]:
        r[s] = dict(novelty=lower_bound(S(Y, "0"), S(Y, s), np.random.default_rng(0)),
                    loss=lower_bound(S(Y, s), S(Y, "0"), np.random.default_rng(0)))
    out[name] = r
    print(name, {s: (round(v["novelty"], 2), round(v["loss"], 2)) for s, v in r.items()}, flush=True)
print(out["lfc3_MAPK"]); print("UP12", out["UP12_genes"]); print("DOWN12", out["DOWN12_genes"])
json.dump(out, open(RESULTS + "/panel_direction_check.json", "w"), indent=1)
