#!/usr/bin/env python3
"""Rejection rates of every method per benchmark scenario (rejection_rates.csv, Figs. 3 and S3).
Rejection of "selection plus a shared response": statistic > B, or p < 0.05 for C2ST.
Input: benchmark_results.json (benchmark.py). Output: rejection_rates.csv."""
from config import RESULTS
import json
import pandas as pd

d = pd.DataFrame(json.load(open(RESULTS + "/benchmark_results.json")))
methods = [("ours", "ours (M_L, adjusted)"), ("ours_unadj", "ours unadjusted"),
           ("clf_ratio_max", "classifier ratio, max"), ("clf_ratio_q99", "classifier ratio, q99"),
           ("c2st", "C2ST (p<0.05)"), ("meld_max", "MELD ratio, max"), ("meld_q99", "MELD ratio, q99"),
           ("milo_sig_max", "Milo-like, max sig. FC"), ("uot_max", "UOT growth, max"), ("uot_q99", "UOT growth, q99")]
rows = {}
for col, name in methods:
    rej = d.c2st_p < 0.05 if col == "c2st" else d[col] > d.B
    rows[name] = rej.groupby(d.scenario).mean()
tab = pd.DataFrame(rows).T
tab = tab[sorted(tab.columns)]
tab.to_csv(RESULTS + "/rejection_rates.csv")
print(tab.round(2).to_string())
