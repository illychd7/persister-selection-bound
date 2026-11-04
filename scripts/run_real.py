"""Apply the decomposition to PC9 / osimertinib (manuscript PC space, 20 PCs)."""
from config import PC9, RESULTS  # data/results/figure folders, see config.py
import json, numpy as np
from decomposition import formal_test, clone_decomposition, Q
z = np.load(PC9 + "/pc9_pp.npz", allow_pickle=True)
Y = z["pc"][:, :Q].astype(float); samp = z["sample"]; ev = z["eval"]; clone = z["clone"]
TRE = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_low", "14_rep2_low", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
out = {}
a0 = samp == "0"
for t in TRE:
    A = formal_test(Y, samp, ev, clone, t)
    at = samp == t
    B = clone_decomposition(Y[a0], clone[a0], Y[at], clone[at], seed=1)
    out[t] = dict(formal=A, clone=B)
    print(t, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in A.items() if k in ("identity", "F_trans", "F_affine", "F_clone")},
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in B.items() if k in ("n_clones", "share_selection", "frac_common", "frac_clone_specific",
                                                                                           "frac_specific_state_dependent", "p_clone_specific", "p_state_dependent", "within_clone_var_ratio", "share_selection_incl_extinction", "frac_baseline_clones_extinct")}, flush=True)
# null reference for the formal test: day-0 eval halves vs fit-estimated maps between day-0 fit halves
json.dump(out, open(RESULTS + "/real_results.json", "w"), indent=1)
