"""Figures 2, 5 and 6 (fig_semisynthetic, fig_budget_adjusted, fig_clone_decomposition) from saved results.
Usage: python make_figs_core.py [results dir] [figure dir]"""
from config import FIGS, RESULTS  # data/results/figure folders, see config.py
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
IN = sys.argv[1] if len(sys.argv) > 1 else RESULTS
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else FIGS) + "/"
os.chdir(IN)  # inputs are read from the results folder
C = dict(b="#2a78d6", o="#eb6834", a="#1baf7a", y="#eda100", m="#e87ba4", v="#6250d6", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", band="#d9d8d3")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.titlesize": 9, "axes.titlelocation": "left", "pdf.fonttype": 42})
def clean(ax, grid="y"):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=C["grid"], lw=.5); ax.set_axisbelow(True)
MT = json.load(open("main_table.json")); R = json.load(open("real_results.json")); S = json.load(open("semisynthetic.json"))
V = json.load(open("dispersion_check.json")); WB = json.load(open("within_vs_between.json")); F = json.load(open("fate_predictability.json"))
P = json.load(open("pooled_decomp.json")); z = np.load("fig1b_digitized.npz")
SL = {"3_rep1": "d3 r1", "3_rep2": "d3 r2", "7_rep1": "d7 r1", "7_rep2": "d7 r2", "14_rep1_low": "d14 low r1", "14_rep2_low": "d14 low r2",
      "14_rep1_med": "d14 med r1", "14_rep2_med": "d14 med r2", "14_rep1_high": "d14 high r1", "14_rep2_high": "d14 high r2"}

# ---------------------------------------------------------------- Fig 2: live-imaging budget + response-adjusted bounds
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=200, gridspec_kw=dict(width_ratios=[1, 2.3]))
last, W = z["last"], int(z["W"]); days = np.linspace(0, 14, 141)
alive = [(last >= max(int(d / 14 * W) - 1, 0)).mean() for d in days]
clean(a1); a1.plot(days, np.array(alive) * 100, color=C["ink"], lw=2)
for d, f in zip(z["days"], [0.463, 0.157, 0.070]):
    a1.scatter(d, f * 100, s=26, color=C["o"], zorder=3, edgecolor="white", lw=.7); a1.text(d + .3, f * 100 + 4, f"{f*100:.0f}%", fontsize=7, color=C["ink2"])
a1.set_xlabel("Days of osimertinib"); a1.set_ylabel("Tracked lineages alive (%)"); a1.set_ylim(0, 105); a1.set_xlim(0, 14.5)
a1.set_title("A  Lineage survival (live imaging, n = 1,135)")
clean(a2); groups = [(3, ["3_rep1", "3_rep2"]), (7, ["7_rep1", "7_rep2"]), (14, ["14_rep1_low", "14_rep2_low", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"])]
x = 0; ticks = []
for day, ss in groups:
    x0 = x
    for s in ss:
        a, b = MT[s]["identity"], MT[s]["translation"]
        a2.plot([x, x], [b, a], color=C["band"], lw=1.3, zorder=1)
        a2.scatter(x, a, s=24, color=C["b"], edgecolor="white", lw=.7, zorder=3); a2.scatter(x, b, s=24, marker="D", color=C["o"], edgecolor="white", lw=.7, zorder=3)
        ticks.append((x, SL[s])); x += 1
    lo, hi = MT[ss[0]]["budget_proxy"], MT[ss[0]]["budget_upper"]
    a2.fill_between([x0 - .45, x - .55], lo, hi, color=C["band"], alpha=.55, lw=0, zorder=0); x += .7
a2.set_yscale("log"); a2.set_ylim(2, 3000); a2.set_xticks([t for t, _ in ticks], [l for _, l in ticks], rotation=40, ha="right", fontsize=6.5)
a2.set_ylabel("Certified enrichment $M_L$ (familywise 95%)")
a2.legend(handles=[Line2D([], [], marker="o", ls="", color=C["b"], label="reference $\\mu_0$ (pure selection)"),
                   Line2D([], [], marker="D", ls="", color=C["o"], label="reference $\\mu_0 K_t$ (selection + common response)"),
                   plt.Rectangle((0, 0), 1, 1, color=C["band"], alpha=.55, label="selection budget (proxy to upper)")], loc="upper left", fontsize=6.5, frameon=False)
a2.set_title("B  After a common response, no bound exceeds the upper selection budget")
fig.tight_layout(); fig.savefig(OUT + "fig_budget_adjusted.pdf"); fig.savefig(OUT + "fig_budget_adjusted.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: semi-synthetic calibration (validity + power)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
order = ["S0 selection only", "S1 selection + common", "S2 + clone-intrinsic 5%", "S3 + clone-intrinsic 30%", "S4 + state-dependent 30%", "S5 + cell-level noise"]
lab = ["selection\nonly", "+ common\nresponse", "+ clone-\nintrinsic 5%", "+ clone-\nintrinsic 30%", "+ state-\ndep. 30%", "+ cell-level\nnoise"]
clean(b1)
for i, sc in enumerate(order):
    rr = [r for r in S if r["scenario"] == sc]; jit = np.linspace(-.12, .12, len(rr))
    b1.scatter(i + jit, [r["formal"]["identity"] for r in rr], s=14, color=C["b"], edgecolor="white", lw=.5, zorder=3)
    b1.scatter(i + jit, [r["formal"]["F_trans"] for r in rr], s=18, marker="D", color=C["o"], edgecolor="white", lw=.5, zorder=3)
    b1.scatter(i + jit, [r["formal"]["F_affine"] for r in rr], s=14, marker="x", color=C["ink2"], lw=.8, zorder=2)
    b1.scatter(np.full(len(rr), i + .3), [r["max_clone_enrichment"] for r in rr], s=10, marker="_", color=C["a"], lw=1.4, zorder=2)
b1.set_yscale("log"); b1.set_ylim(.5, 150); b1.set_xticks(range(6), lab, fontsize=6.2); b1.set_ylabel("$M_L$")
b1.legend(handles=[Line2D([], [], marker="o", ls="", color=C["b"], label="reference $\\mu_0$"), Line2D([], [], marker="D", ls="", color=C["o"], label="common translation"),
                   Line2D([], [], marker="x", ls="", color=C["ink2"], label="affine moment matching"),
                   Line2D([], [], marker="_", ls="", color=C["a"], markersize=8, mew=1.5, label="true max clone enrichment")], loc="upper left", fontsize=6.2, frameon=False, ncol=2)
b1.set_title("A  Validity and power of the response-adjusted bound")
clean(b2)
for i, sc in enumerate(order[1:], start=1):
    rr = [r for r in S if r["scenario"] == sc]; jit = np.linspace(-.12, .12, len(rr))
    b2.scatter(i + jit, [100 * r["clone"]["frac_clone_specific"] for r in rr], s=18, marker="s", color=C["m"], edgecolor="white", lw=.5, zorder=3)
    tp = {"S2 + clone-intrinsic 5%": 5, "S3 + clone-intrinsic 30%": 30, "S4 + state-dependent 30%": 30}.get(sc, 0)
    b2.plot([i - .25, i + .25], [tp, tp], color=C["ink"], lw=1.2)
b2.set_xticks(range(1, 6), lab[1:], fontsize=6.2); b2.set_xlim(.4, 5.6); b2.set_ylabel("Clone-specific share of response (%)")
b2.legend(handles=[Line2D([], [], marker="s", ls="", color=C["m"], label="estimated (split-half)"), Line2D([], [], color=C["ink"], lw=1.2, label="true")], loc="upper right", fontsize=6.5, frameon=False)
b2.set_title("B  Clone-level estimator: 2.9–5.9% floor without plasticity")
fig.tight_layout(); fig.savefig(OUT + "fig_semisynthetic.pdf"); fig.savefig(OUT + "fig_semisynthetic.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: clone decomposition
ss = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
xl = ["d3 r1", "d3 r2", "d7 r1", "d7 r2", "d14 m1", "d14 m2", "d14 h1", "d14 h2"]; xs = np.arange(len(ss))
fig, (c1, c2, c3) = plt.subplots(1, 3, figsize=(10.2, 3.3), dpi=200)
clean(c1); nul = [r["clone"]["frac_clone_specific"] for r in S if r["scenario"] == "S1 selection + common"]
c1.axhspan(100 * min(nul), 100 * max(nul), color=C["band"], alpha=.6, lw=0); c1.text(len(ss) - .5, 100 * max(nul) + .4, "calibrated null", fontsize=6, color=C["ink2"], ha="right")
c1.scatter(xs, [100 * R[s]["clone"]["share_selection_incl_extinction"] for s in ss], s=20, color=C["a"], edgecolor="white", lw=.6, zorder=3, label="selection share of mean shift")
c1.scatter(xs, [100 * R[s]["clone"]["frac_clone_specific"] for s in ss], s=20, marker="s", color=C["m"], edgecolor="white", lw=.6, zorder=3, label="clone-specific share of response")
c1.axhline(0, color=C["ink2"], lw=.6); c1.set_ylim(-3, 10); c1.set_xticks(xs, xl, rotation=45, fontsize=6.2); c1.set_ylabel("%")
c1.legend(fontsize=6, frameon=False, loc="upper left"); c1.set_title("A  Selection moves clones, not the state")
clean(c2); w = .2
c2.bar(xs - 1.5 * w, [WB[s]["within_ratio"] for s in ss], w, color=C["b"], label="within clone (HVG)")
c2.bar(xs - .5 * w, [WB[s]["between_ratio"] for s in ss], w, color=C["v"], label="between clones (HVG)")
c2.bar(xs + .5 * w, [V[s]["RANDOM200_median"] for s in ss], w, color=C["y"], label="within, random 200 genes")
c2.bar(xs + 1.5 * w, [V[s]["STABLE200"] for s in ss], w, color=C["band"], label="within, drug-stable 200 genes")
c2.axhline(1, color=C["ink2"], lw=.7); c2.set_ylim(0, 7.8); c2.set_xticks(xs, xl, rotation=45, fontsize=6.2); c2.set_ylabel("Variance ratio (t / day 0)")
c2.legend(fontsize=5.8, frameon=False, loc="upper left"); c2.set_title("B  State space expands in responsive genes")
clean(c3); fs = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_high", "14_rep2_high"]; xf = np.arange(len(fs))
c3.bar(xf, [F[s]["target_reliability"] for s in fs], .6, color=C["band"], label="reliability ceiling")
c3.bar(xf, [max(F[s]["cv_R2"], 0) for s in fs], .6, color=C["a"], label="CV $R^2$ from baseline state")
c3.set_xticks(xf, ["d3 r1", "d3 r2", "d7 r1", "d7 r2", "d14 h1", "d14 h2"], rotation=45, fontsize=6.2); c3.set_ylim(0, 1); c3.set_ylabel("Clone log-enrichment, $R^2$")
c3.legend(fontsize=6, frameon=False, loc="upper left"); c3.set_title("C  Fate is weakly readable at baseline")
fig.tight_layout(); fig.savefig(OUT + "fig_clone_decomposition.pdf"); fig.savefig(OUT + "fig_clone_decomposition.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: cross-system
rows = [("PC9 time course", SL[s], MT[s]["identity"], MT[s]["translation"]) for s in ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_high", "14_rep2_high"]]
for line, nm in [("A375_SKIN", "A375"), ("HT29_LARGE_INTESTINE", "HT29"), ("PC9_LUNG", "PC9")]:
    for ex, d in P[line].items(): rows.append(("Pooled lines", f"{nm} {ex}", d["ML_mixture"], d["common_ML_mixture_min_over_w"]))
fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200); clean(ax)
x, prev, xt, xlab = 0, None, [], []
for grp, lab_, a, b in rows:
    if prev is not None and grp != prev: x += .8
    ax.plot([x, x], [b, a], color=C["band"], lw=1.3, zorder=1)
    ax.scatter(x, a, s=24, color=C["b"], edgecolor="white", lw=.7, zorder=3); ax.scatter(x, b, s=24, marker="D", color=C["o"], edgecolor="white", lw=.7, zorder=3)
    ax.text(x, a * 1.2, f"{a / b:.0f}×", ha="center", fontsize=6, color=C["ink2"]); xt.append(x); xlab.append(lab_); x += 1; prev = grp
ax.axhline(1, color=C["ink2"], lw=.7, ls=(0, (3, 2))); ax.set_yscale("log"); ax.set_ylim(.6, 300)
ax.set_xticks(xt, xlab, rotation=40, ha="right", fontsize=6.6); ax.set_ylabel("Certified enrichment $M_L$")
ax.text(2.5, 220, "PC9 time course (barcoded)", ha="center", fontsize=7); ax.text(9.3, 220, "Pooled lines, day 10 (gate-agnostic)", ha="center", fontsize=7)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C["b"], label="reference $\\mu_0$"), Line2D([], [], marker="D", ls="", color=C["o"], label="reference $\\mu_0K_t$ (one common translation)")],
          loc="lower left", fontsize=6.6, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig_cross_system.pdf"); fig.savefig(OUT + "fig_cross_system.png"); plt.close(fig)
print("figures written")
