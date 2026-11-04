"""Main table (v5.3): full held-out population, PC9 time course.
Identity and a common translation estimated on FIT cells only; bounds on EVAL cells.
Familywise 95% over 10 treated samples x 2 reference models (alpha = 0.05/20 per bound).
Also: batch-sensitivity tolerance c* = M_L / budget (Proposition: technical inflation)."""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import sys, json, numpy as np
pass
from seltest import lower_bound
z = np.load(PC9 + "/pc9_pp.npz", allow_pickle=True)
Y = z["pc"][:, :20].astype(float); samp = z["sample"]; ev = z["eval"]
TRE = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_low", "14_rep2_low", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
A = 0.05 / 20
BUD = {3: (5.88, 17.30), 7: (36.41, 204.04), 14: (108.04, 1838.99)}
out = {}
Y0f, Y0e = Y[~ev & (samp == "0")], Y[ev & (samp == "0")]
for t in TRE:
    Ytf, Yte = Y[~ev & (samp == t)], Y[ev & (samp == t)]
    d = Ytf.mean(0) - Y0f.mean(0)
    idt = lower_bound(Y0e, Yte, np.random.default_rng(0), alpha=A)
    tr = lower_bound(Y0e + d, Yte, np.random.default_rng(0), alpha=A)
    day = int(t.split("_")[0]); lo, hi = BUD[day]
    out[t] = dict(n0=int(len(Y0e)), nt=int(len(Yte)), identity=idt, translation=tr, budget_proxy=lo, budget_upper=hi,
                  cstar_identity_upper=idt / hi, cstar_translation_upper=tr / hi, shift_norm=float(np.linalg.norm(d)))
    print(t, {k: round(v, 2) if isinstance(v, float) else v for k, v in out[t].items()}, flush=True)
json.dump(out, open(RESULTS + "/main_table.json", "w"), indent=1)
