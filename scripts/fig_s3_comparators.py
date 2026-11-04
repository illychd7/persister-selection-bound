"""Figure S3: all comparators adapted to the mechanism-specific question (benchmark, default settings).
Inputs: rejection_rates.csv (benchmark.py, 20 replicates) and milo_official_results.json."""
from config import FIGS, RESULTS  # data/results/figure folders, see config.py
import json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
IN, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (RESULTS, FIGS)
C = ["#eb6834", "#c9c8c3", "#e34948", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#2a78d6"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titlesize": 8.5})
tab = pd.read_csv(f"{IN}/rejection_rates.csv", index_col=0)
MO = pd.DataFrame(json.load(open(f"{IN}/milo_official_results.json"))); tab.loc["milo_official"] = (MO.milo_sig_max > MO.B).groupby(MO.scenario).mean()
order = ["N1 selection only", "N2 selection + common", "N3 strong selection + common", "A1 + clone-intrinsic 5%", "A2 + clone-intrinsic 30%",
         "A3 + state-dependent 30%", "A4 + cell-level heterogeneity"]
lab = ["selection\nonly", "selection +\ncommon", "strong sel. +\ncommon", "clone-\nintrinsic 5%", "clone-\nintrinsic 30%", "state-\ndependent 30%", "cell-level\nheterogeneity"]
meths = [("ours (M_L, adjusted)", "response-adjusted bound"), ("ours unadjusted", "bound, unadjusted"), ("milo_official", "Milo (pertpy, 3+3 replicates)"),
         ("Milo-like, max sig. FC", "Milo-like, single sample"), ("MELD ratio, max", "MELD ratio, max"), ("UOT growth, max", "unbalanced OT growth, max"),
         ("classifier ratio, q99", "classifier density ratio, q99"), ("C2ST (p<0.05)", "C2ST (p < 0.05)")]
fig, ax = plt.subplots(figsize=(10, 3.4), dpi=200)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
x = np.arange(len(order)); w = .1
ax.axvspan(-.5, 2.5, color="#eef2f7", lw=0, zorder=0)
for i, ((k, l), c) in enumerate(zip(meths, C)):
    ax.bar(x + (i - 3.5) * w, 100 * tab.loc[k, order].values, w, color=c, label=l, edgecolor="white", lw=.3)
ax.axhline(5, color="#52514e", lw=.6, ls=(0, (3, 2)))
ax.text(1, 106, "mechanism-specific null true", ha="center", fontsize=7, color="#52514e"); ax.text(4.5, 106, "alternatives", ha="center", fontsize=7, color="#52514e")
ax.set_xticks(x, lab, fontsize=6.6); ax.set_ylim(0, 112); ax.set_xlim(-.5, len(order) - .5)
ax.set_ylabel("Rejections of 'selection + shared response' (%)")
ax.legend(fontsize=6.3, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.2))
ax.set_title("Comparators adapted to the mechanism-specific question (20 replicates per scenario, default settings)")
fig.savefig(f"{OUT}/fig_s3_comparators.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_s3_comparators.png", bbox_inches="tight", dpi=200)
