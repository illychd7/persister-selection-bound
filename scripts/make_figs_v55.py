#!/usr/bin/env python3
"""Figures 1, 6, 7 and 8 for manuscript v5.5 (style of make_figs.py v5.3).

Inputs (same folder or paths below)
  sim_fig1.csv                      sim_fig1.py
  gene_panel_control.json           gene_panel_control.py
  panel_direction_check.json        panel_direction_check.py
  milo_official_results.json        bench_milo.py main 20
  milo_official_sensitivity.json    bench_milo.py sens 20   (20 seeds per k)

Fixes relative to the submitted v5.5 figures
  Fig. 6: footnote no longer refers to an internal manuscript version.
  Fig. 8A: legend moved below the axes, so it no longer overlaps the shaded-region labels.
  Fig. 8B: the response-adjusted bound and the Milo points at (0, 100) are both visible.
"""
from config import FIGS, RESULTS  # data/results/figure folders, see config.py
import json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IN = sys.argv[1] if len(sys.argv) > 1 else RESULTS
OUT = sys.argv[2] if len(sys.argv) > 2 else FIGS
C = dict(b="#2a78d6", o="#eb6834", a="#1baf7a", y="#eda100", m="#e87ba4", v="#6250d6", r="#e34948",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", band="#d9d8d3", shade="#eef2f7")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": C["ink2"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.titlesize": 9,
                     "axes.titlelocation": "left", "pdf.fonttype": 42})


def clean(ax, grid="y"):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, color=C["grid"], lw=.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1: known-response simulation
def fig1():
    d = pd.read_csv(f"{IN}/sim_fig1.csv")
    cols, labs = ["ignore", "translation", "correct"], ["ignore response", "translation\n(misspecified)", "correct response"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=200); clean(ax)
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols):
        v = d[c].values
        ax.boxplot(v, positions=[i], widths=.45, showfliers=False, medianprops=dict(color=C["o"], lw=1.4),
                   boxprops=dict(color=C["ink2"]), whiskerprops=dict(color=C["ink2"]), capprops=dict(color=C["ink2"]))
        ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=5, color=C["b"], alpha=.35, lw=0, zorder=3)
        labs[i] += f"\nmedian {np.median(v):.2g}×, {100 * (v > d.true).mean():.0f}% above truth"
    lo, hi = d.true.min(), d.true.max()
    ax.axhspan(lo, hi, color=C["band"], alpha=.6, lw=0, zorder=0)
    ax.axhline(d.true.median(), color=C["ink"], lw=.9, ls=(0, (4, 2)))
    ax.text(-.45, d.true.median() * 1.12, f"true selective enrichment (median {d.true.median():.2f}×; band = range)",
            ha="left", fontsize=6.5, color=C["ink"])
    ax.set_yscale("log"); ax.set_ylim(.5, d["ignore"].max() * 2)
    ax.set_xticks(range(3), labs, fontsize=7); ax.set_ylabel("Finite-sample lower bound $M_L$")
    ax.set_title(f"Known-response simulation ({len(d)} runs)")
    save(fig, "fig_known_response")


# ------------------------------------------------------------------ Fig 6: feature-space controls
def fig6():
    G = json.load(open(f"{IN}/gene_panel_control.json"))
    T = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_low", "14_rep2_low", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
    REPS = [("3_rep1", "3_rep2"), ("7_rep1", "7_rep2"), ("14_rep1_low", "14_rep2_low"), ("14_rep1_med", "14_rep2_med"), ("14_rep1_high", "14_rep2_high")]
    day_col = {3: C["b"], 7: C["o"], 14: C["a"]}
    rows = [("Analysis space\n(2000 HVG, 20 PCs)", "HVG2000", False), ("Random 200 genes\n(expression-matched)", "RANDOM_matched_STABLE200", True),
            ("Drug-stable 200 genes", "STABLE200", False), ("Technical covariates only\n(genes detected, mito fraction)", "TECH2", False),
            ("Housekeeping 11\n(Eisenberg & Levanon)", "HK11", False), ("Random 11 genes\n(expression-matched)", "RANDOM_matched_HK11", True)]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200); clean(ax, grid="x")
    rng = np.random.default_rng(1); fl = lambda v: max(v, .3)
    for i, (lab, key, rnd) in enumerate(rows):
        y = len(rows) - 1 - i; g = G[key]
        for t in T:
            v = g[t]["median"] if rnd else g[f"0->{t}"]
            ax.scatter(fl(v), y + .12 + rng.uniform(-.05, .05), s=16, color=day_col[int(t.split("_")[0])], edgecolor="white", lw=.4, zorder=3)
        reps = [g["3_rep1->3_rep2"]["median"]] if rnd else [g[f"{a}->{b}"] for a, b in REPS] + [g[f"{b}->{a}"] for a, b in REPS]
        ax.scatter([fl(v) for v in reps], y - .14 + rng.uniform(-.05, .05, len(reps)), s=14, marker="s", color=C["ink2"], alpha=.8, lw=0, zorder=3)
    ax.axvline(17.3, color=C["ink2"], lw=.8, ls=(0, (3, 2)))
    ax.text(17.3 * 1.07, len(rows) - .45, "day-3 selection budget\n(upper bound, 17.3)", fontsize=6, color=C["ink2"], va="top")
    ax.set_xscale("log"); ax.set_xlim(.25, 300)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=7); ax.set_ylim(-.6, len(rows) - .4)
    ax.set_xlabel("Lower confidence bound $M_L$ (log scale)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=day_col[3], label="day 3 vs day 0"), Line2D([], [], marker="o", ls="", color=day_col[7], label="day 7 vs day 0"),
                       Line2D([], [], marker="o", ls="", color=day_col[14], label="day 14 vs day 0"), Line2D([], [], marker="s", ls="", color=C["ink2"], label="replicate library vs replicate")],
              loc="lower right", fontsize=6.5, frameon=False)
    ax.set_title("$M_L$ is carried by drug-responsive genes, not by library identity")
    fig.text(0.01, -0.04, "Same cells, thinning (3000 UMIs), held-out evaluation cells and residual model as the main analysis; seed 0.\n"
             f"Random panels: median of 20 (200 genes) or 40 (11 genes) draws. Drug-stable panel selected on FIT cells only "
             f"(max |log2FC| vs day 0 = {G['STABLE200']['max_abs_log2fc_fit']:.2f}). Points below 0.3 drawn at 0.3.", fontsize=5.6, color=C["ink2"])
    save(fig, "fig_gene_panels")


# ------------------------------------------------------------------ Fig 7: directional controls
def fig7():
    P = json.load(open(f"{IN}/panel_direction_check.json"))
    S = ["3_rep1", "3_rep2", "7_rep1", "14_rep1_high"]; xl = ["d3 r1", "d3 r2", "d7 r1", "d14 high r1"]; x = np.arange(len(S))
    spec = [("MAPK12", "MAPK outputs (12)", C["b"], "o"), ("UP12_fit", "12 induced genes (FIT)", C["a"], "s"), ("DOWN12_fit", "12 repressed genes (FIT)", C["v"], "^")]
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=200); clean(ax)
    fl = lambda v: max(v, 1e-3)
    for key, lab, col, mk in spec:
        ax.plot(x, [fl(P[key][s]["novelty"]) for s in S], color=col, marker=mk, lw=1.3, ms=4.5, label=f"{lab}: novelty")
        ax.plot(x, [fl(P[key][s]["loss"]) for s in S], color=col, marker=mk, lw=1.1, ms=4.5, ls=(0, (4, 2)), mfc="white", label=f"{lab}: loss")
    ax.axhline(1, color=C["ink2"], lw=.7, ls=":")
    ax.set_yscale("log"); ax.set_xticks(x, xl); ax.set_ylabel("Directional bound (values below $10^{-3}$ drawn at $10^{-3}$)")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax.set_title("The bound is directional: induction appears as novelty, repression as loss")
    save(fig, "fig_direction")


# ------------------------------------------------------------------ Fig 8: Milo benchmark
def fig8():
    MO = pd.DataFrame(json.load(open(f"{IN}/milo_official_results.json")))
    MS = pd.DataFrame(json.load(open(f"{IN}/milo_official_sensitivity.json")))
    order = ["N1 selection only", "N2 selection + common", "N3 strong selection + common", "A1 + clone-intrinsic 5%",
             "A2 + clone-intrinsic 30%", "A3 + state-dependent 30%", "A4 + cell-level heterogeneity"]
    lab = ["selection\nonly", "selection +\ncommon", "strong sel. +\ncommon", "clone-\nintrinsic 5%", "clone-\nintrinsic 30%", "state-\ndependent 30%", "cell-level\nheterogeneity"]
    MO["ours_rej"] = MO.ours > MO.B; MO["milo_rej"] = MO.milo_sig_max > MO.B; MO["milo_da"] = MO.milo_frac_sig > 0
    rate = MO.groupby("scenario")[["ours_rej", "milo_rej", "milo_da"]].mean().loc[order] * 100
    n = MO.groupby("scenario").size().loc[order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.7), dpi=200, gridspec_kw=dict(width_ratios=[2.3, 1]))
    clean(a1); x = np.arange(len(order)); w = .26
    a1.axvspan(-.5, 2.5, color=C["shade"], lw=0, zorder=0)
    a1.text(1, 108, "mechanism-specific null true", ha="center", fontsize=7, color=C["ink2"])
    a1.text(4.5, 108, "alternatives", ha="center", fontsize=7, color=C["ink2"])
    for j, (k, l, c) in enumerate([("ours_rej", "response-adjusted bound ($M_L > B$)", C["b"]), ("milo_rej", "Milo effect exceeds selection budget", C["o"]),
                                    ("milo_da", "Milo detects differential abundance", C["a"])]):
        a1.bar(x + (j - 1) * w, rate[k].values, w, color=c, label=l, edgecolor="white", lw=.3)
        for xi, v in zip(x, rate[k].values):
            if v == 0: a1.text(xi + (j - 1) * w, 1.5, "0", ha="center", fontsize=5.5, color=c)
    a1.axhline(5, color=C["ink2"], lw=.6, ls=(0, (3, 2)))
    a1.set_xticks(x, lab, fontsize=6.4); a1.set_ylim(0, 115); a1.set_xlim(-.5, len(order) - .5)
    a1.set_ylabel(f"Decision / detection rate (%), n = {int(n.min())} per scenario")
    a1.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.2))
    a1.set_title("A  Mechanism-specific certification and differential abundance are distinct")
    # B: k sensitivity, k = 30 from the main run
    MS["rej"] = MS.milo_sig_max > MS.B
    k30 = MO[MO.scenario.isin(["N2 selection + common", "A2 + clone-intrinsic 30%"])].assign(k=30, rej=lambda d: d.milo_rej)
    allk = pd.concat([MS, k30]); pts = {}
    for k, g in allk.groupby("k"):
        pts[k] = (100 * g[g.scenario == "N2 selection + common"].rej.mean(), 100 * g[g.scenario == "A2 + clone-intrinsic 30%"].rej.mean(),
                  int((g.scenario == "N2 selection + common").sum()))
    clean(a2, grid="both"); ks = sorted(pts)
    a2.plot([pts[k][0] for k in ks], [pts[k][1] for k in ks], color=C["o"], lw=.9, zorder=2)
    a2.scatter([pts[k][0] for k in ks], [pts[k][1] for k in ks], s=26, color=C["o"], edgecolor="white", lw=.6, zorder=3, label="Milo effect-budget wrapper")
    groups = {}
    for k in ks: groups.setdefault(pts[k][:2], []).append(k)
    for (xx, yy), kk in groups.items():
        a2.annotate("k = " + ", ".join(map(str, kk)), (xx, yy), textcoords="offset points", xytext=(10, -3), fontsize=6.3, color=C["o"])
    o = MO[MO.scenario == "N2 selection + common"].ours_rej.mean() * 100, MO[MO.scenario == "A2 + clone-intrinsic 30%"].ours_rej.mean() * 100
    a2.scatter([o[0]], [o[1]], s=150, marker="D", facecolor="none", edgecolor=C["b"], lw=1.4, zorder=4, label="response-adjusted bound (no tuning)")
    a2.annotate("bound", (o[0], o[1]), textcoords="offset points", xytext=(10, 6), fontsize=6.3, color=C["b"])
    a2.set_xlim(-6, 104); a2.set_ylim(-6, 110)
    a2.set_xlabel("Rejections under selection + common (%)"); a2.set_ylabel("Detection, clone-intrinsic 30% (%)")
    a2.legend(fontsize=6.3, frameon=True, framealpha=1, edgecolor=C["grid"], loc="lower right", markerscale=.7)
    a2.set_title(f"B  Milo neighbourhood size (n = {min(p[2] for p in pts.values())} per point)")
    fig.tight_layout(); save(fig, "fig_benchmark")
    return rate, pts


if __name__ == "__main__":
    for f in [fig1, fig6, fig7, fig8]:
        try:
            r = f(); print(f.__name__, "ok", r if r is not None else "")
        except FileNotFoundError as e:
            print(f.__name__, "skipped: missing", e.filename)
