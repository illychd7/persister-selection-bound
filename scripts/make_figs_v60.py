#!/usr/bin/env python3
"""New figures for the Cell Reports Methods version (v6.0).

  fig_overview.pdf        Fig. 1  method overview (schematic) + known-response simulation
  fig_power.pdf           Fig. 4  practical guidance: power vs cells and effect, computing time
  fig_generalization.pdf  Fig. 7  pooled lines, TNBC/doxorubicin, melanoma/BRAFi+MEKi

Inputs: sim_fig1.csv, power.csv, runtime.csv, pooled_decomp.json, newdata_results.json,
selection_budget_barcodeseq.csv, real_results.json, semisynthetic.json
Usage: python make_figs_v60.py <input dir> <output dir>
"""
from config import FIGS, RESULTS  # data/results/figure folders, see config.py
import json, sys, numpy as np, pandas as pd, matplotlib
import matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

IN = sys.argv[1] if len(sys.argv) > 1 else RESULTS
OUT = sys.argv[2] if len(sys.argv) > 2 else FIGS
C = dict(b="#2a78d6", o="#eb6834", a="#1baf7a", y="#eda100", m="#e87ba4", v="#6250d6", r="#e34948",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", band="#d9d8d3", shade="#eef2f7")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": C["ink2"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "axes.titlesize": 9, "axes.titlelocation": "left", "pdf.fonttype": 42})


def clean(ax, grid="y"):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, color=C["grid"], lw=.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ Fig 1
def fig_overview():
    fig = plt.figure(figsize=(10.4, 3.9), dpi=200)
    ax = fig.add_axes([0.0, 0.0, 0.58, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    rng = np.random.default_rng(3)
    P = rng.multivariate_normal([0, 0], [[1, .35], [.35, .6]], 260)
    g = np.exp(1.1 * P[:, 0]); g /= g.max()
    def cloud(cx, cy, pts, s, col, alpha):
        ax.scatter(cx + .33 * pts[:, 0], cy + .33 * pts[:, 1], s=s, color=col, alpha=alpha, lw=0)
    cloud(1.1, 4.3, P, 6, C["ink2"], .55)
    cloud(4.2, 4.3, P, 2 + 22 * g, C["b"], .55)
    Pt = np.c_[P[:, 0] * 1.05 + 1.0 + .25 * P[:, 1] ** 2, P[:, 1] * .9 + .45]
    cloud(7.4, 4.0, Pt, 2 + 22 * g, C["o"], .55)
    for x, t, s in [(1.1, "baseline $\\mu_0$", "untreated cells"), (4.2, "$S_g\\mu_0$", "selected (unobserved)"),
                    (7.75, "treated $\\nu_t$", "observed at time $t$")]:
        ax.text(x, 5.55, t, ha="center", fontsize=9); ax.text(x, 5.2, s, ha="center", fontsize=6.8, color=C["ink2"])
    for x0, x1, lab in [(1.95, 3.3, "selection $g$\n(death, division)"), (5.05, 6.45, "shared response\n$K_t$")]:
        ax.add_patch(FancyArrowPatch((x0, 4.3), (x1, 4.3), arrowstyle="-|>", mutation_scale=12, color=C["ink"], lw=1))
        ax.text((x0 + x1) / 2, 4.5, lab, ha="center", fontsize=6.8)
    box = lambda x, y, w, h, txt, col: (ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                                                    fc="white", ec=col, lw=1.1)), ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2))
    box(.2, 1.35, 4.3, 1.35, "Lower confidence bound\n$M_L(\\nu_t \\mid \\mu_0\\hat K_t)\\ \\leq\\ M^*$\n$\\hat K_t$ fitted on FIT cells, bound on EVAL cells", C["b"])
    box(5.0, 1.35, 4.6, 1.35, "Selection budget\n$B = 2^{d}/f$ from survival and division\n($d$ divisions, surviving fraction $f$)", C["a"])
    ax.text(4.75, 2.02, "vs", ha="center", va="center", fontsize=9, color=C["ink2"])
    ax.text(4.9, .55, "$M_L > B$: selection plus a shared response cannot explain the data\n"
                      "$M_L \\leq B$: compatible; barcodes then split selection from response",
            ha="center", va="center", fontsize=7.2)
    fig.text(0.005, 1.04, "A  Response-adjusted selection test", fontsize=9, ha="left", va="top")
    # B: known-response simulation
    d = pd.read_csv(f"{IN}/sim_fig1.csv")
    b = fig.add_axes([0.66, 0.16, 0.33, 0.72]); clean(b)
    cols = ["ignore", "translation", "correct"]; labs = ["ignore\nresponse", "translation\n(misspecified)", "correct\nresponse"]
    for i, c in enumerate(cols):
        v = d[c].values
        b.boxplot(v, positions=[i], widths=.5, showfliers=False, medianprops=dict(color=C["o"], lw=1.4),
                  boxprops=dict(color=C["ink2"]), whiskerprops=dict(color=C["ink2"]), capprops=dict(color=C["ink2"]))
        b.scatter(i + rng.uniform(-.14, .14, len(v)), v, s=4, color=C["b"], alpha=.35, lw=0, zorder=3)
        labs[i] += f"\n{np.median(v):.2g}×; {100 * (v > d.true).mean():.0f}% > truth"
    b.axhspan(d.true.min(), d.true.max(), color=C["band"], alpha=.6, lw=0, zorder=0)
    b.axhline(d.true.median(), color=C["ink"], lw=.9, ls=(0, (4, 2)))
    b.text(-.45, d.true.median() * 1.13, f"true enrichment ({d.true.median():.2f}×)", fontsize=6.5)
    b.set_yscale("log"); b.set_ylim(.5, d["ignore"].max() * 2); b.set_xticks(range(3), labs, fontsize=6.4)
    b.set_ylabel("Lower bound $M_L$"); b.set_title(f"B  Known-response simulation ({len(d)} runs)")
    save(fig, "fig_overview")


# ------------------------------------------------------------------ Fig 4
def fig_power():
    P = pd.read_csv(f"{IN}/power.csv"); R = pd.read_csv(f"{IN}/runtime.csv")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.2), dpi=200, gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
    cols = {0.0: C["ink2"], 0.05: C["y"], 0.10: C["m"], 0.20: C["v"], 0.30: C["b"]}
    pw = P.groupby(["phi", "n"]).reject.agg(["mean", "size"]).reset_index()
    clean(a1, grid="both")
    for phi, g in pw.groupby("phi"):
        lab = "null (selection + common)" if phi == 0 else f"clone-intrinsic {int(round(100 * phi))}%"
        a1.plot(g.n, 100 * g["mean"], marker="o", ms=4, lw=1.4, color=cols[phi], ls="--" if phi == 0 else "-", label=lab)
    a1.axhline(80, color=C["ink2"], lw=.6, ls=":"); a1.set_xscale("log"); a1.set_xticks(sorted(P.n.unique()), [str(n) for n in sorted(P.n.unique())])
    a1.set_ylim(-4, 104); a1.set_xlabel("Cells per condition (EVAL)"); a1.set_ylabel(f"Rejections (%), n = {int(pw['size'].min())} per point")
    a1.legend(fontsize=6.3, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.2)); a1.set_title("A  Power grows with cells and effect size")
    med = P.groupby(["phi", "n"]).M_L.median().reset_index(); clean(a2, grid="both")
    for phi, g in med.groupby("phi"):
        a2.plot(g.n, g.M_L, marker="o", ms=4, lw=1.4, color=cols[phi], ls="--" if phi == 0 else "-")
    a2.axhspan(P.B.quantile(.1), P.B.quantile(.9), color=C["band"], alpha=.6, lw=0)
    a2.text(205, P.B.quantile(.1) / 1.22, "grey band: oracle budget $B$ (10th–90th pct.)", fontsize=6.1, color=C["ink2"])
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xticks(sorted(P.n.unique()), [str(n) for n in sorted(P.n.unique())])
    a2.set_xlabel("Cells per condition (EVAL)"); a2.set_ylabel("Median $M_L$ after translation"); a2.set_title("B  The bound, not the budget, moves with $n$")
    clean(a3, grid="both")
    a3.plot(R.n_per_condition, R.seconds, marker="o", ms=4, color=C["o"], lw=1.4)
    for x, y in zip(R.n_per_condition, R.seconds): a3.annotate(f"{y:.2g} s", (x, y), textcoords="offset points", xytext=(-6, 7), fontsize=6.2, color=C["ink2"])
    a3.set_xscale("log"); a3.set_yscale("log"); a3.set_xlabel("Cells per condition"); a3.set_ylabel("Seconds per bound (20 dims, 1 core)")
    a3.set_title("C  Computing time is linear in $n$")
    for a_ in (a1, a2): a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout(); save(fig, "fig_power")


# ------------------------------------------------------------------ Fig 7
def fig_generalization():
    Pd = json.load(open(f"{IN}/pooled_decomp.json")); R = json.load(open(f"{IN}/newdata_results.json"))
    SB = pd.read_csv(f"{IN}/selection_budget_barcodeseq.csv").set_index("well")
    RR = json.load(open(f"{IN}/real_results.json")); SS = json.load(open(f"{IN}/semisynthetic.json"))
    fig, ax = plt.subplots(1, 4, figsize=(12.2, 3.5), dpi=200, gridspec_kw=dict(width_ratios=[1.05, 1.35, .9, 1.05]))
    # A pooled lines
    a = ax[0]; clean(a); x = 0; xt, xl = [], []
    for line, nm in [("A375_SKIN", "A375"), ("HT29_LARGE_INTESTINE", "HT29"), ("PC9_LUNG", "PC9")]:
        for ex, d in Pd[line].items():
            u, t = d["ML_mixture"], d["common_ML_mixture_min_over_w"]
            a.plot([x, x], [t, u], color=C["band"], lw=1.3); a.scatter(x, u, s=22, color=C["b"], zorder=3); a.scatter(x, t, s=24, marker="D", color=C["o"], zorder=3)
            xt.append(x); xl.append(f"{nm} {ex}"); x += 1
    a.axhline(1, color=C["ink2"], lw=.6, ls=":"); a.set_yscale("log"); a.set_ylim(.6, 60); a.set_xticks(xt, xl, rotation=40, ha="right", fontsize=6.4)
    a.set_ylabel("Certified enrichment $M_L$"); a.set_title("A  Pooled lines, day 10")
    a.legend(handles=[Line2D([], [], marker="o", ls="", color=C["b"], label="vs untreated"), Line2D([], [], marker="D", ls="", color=C["o"], label="after one common response")],
             fontsize=6.2, frameon=False, loc="upper left")
    # B TNBC
    a = ax[1]; clean(a); T = R["TNBC"]
    order = [("30HR", "30 h\n550 nM", None), ("250_1", "250-1", "A1"), ("250_4", "250-4", "A4"), ("250_6", "250-6", "A6"),
             ("400_2", "400-2", "B2"), ("400_4", "400-4", "B5"), ("400_5", "400-5", "B6")]
    tech = sorted(T["technical"].values())
    a.axhspan(tech[0], tech[1], color=C["band"], alpha=.6, lw=0)
    for i, (k, lab, well) in enumerate(order):
        u, t = T[k]["identity"], T[k]["translation"]
        a.plot([i, i], [t, u], color=C["band"], lw=1.3); a.scatter(i, u, s=22, color=C["b"], zorder=3); a.scatter(i, t, s=24, marker="D", color=C["o"], zorder=3)
        if well: a.scatter(i, SB.loc[well, "max_clone_enrichment"], s=30, marker="*", color=C["a"], zorder=3)
    a.axvline(.5, color=C["ink2"], lw=.5, ls=":"); a.text(.6, 450, "recovered after washout (~50 d)", fontsize=6.2, color=C["ink2"])
    a.set_yscale("log"); a.set_ylim(.5, 800); a.set_xticks(range(len(order)), [o[1] for o in order], fontsize=6.4)
    a.set_title("B  TNBC (MDA-MB-231), doxorubicin")
    a.legend(handles=[Line2D([], [], marker="o", ls="", color=C["b"], label="$M_L$ vs untreated"), Line2D([], [], marker="D", ls="", color=C["o"], label="$M_L$ after translation"),
                      Line2D([], [], marker="*", ls="", color=C["a"], ms=8, label="max clone enrichment\n(barcode-seq)"),
                      plt.Rectangle((0, 0), 1, 1, color=C["band"], alpha=.6, label="technical: replicate\nuntreated libraries")], fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.16))
    # C clone-specific share vs floor
    a = ax[2]; clean(a)
    pc9_floor = [r["clone"]["frac_clone_specific"] for r in SS if r["scenario"] == "S1 selection + common"]
    rows = [("PC9\nday 3", 100 * RR["3_rep1"]["clone"]["frac_clone_specific"], 100 * np.array(pc9_floor)),
            ("TNBC\n30 h", 100 * T["30HR"]["clone"]["frac_clone_specific"], 100 * np.array(T["floor_S1_30HR"])),
            ("melanoma\n1 week", 100 * R["MeRLin"]["clone"]["frac_clone_specific"], 100 * np.array(R["MeRLin"]["floor_S1"]))]
    for i, (lab, v, fl) in enumerate(rows):
        a.add_patch(plt.Rectangle((i - .3, fl.min()), .6, fl.max() - fl.min(), color=C["band"], alpha=.8, lw=0))
        a.scatter(i, v, s=40, marker="s", color=C["m"], zorder=3)
    a.set_xticks(range(3), [r[0] for r in rows], fontsize=6.6); a.set_xlim(-.6, 2.6); a.set_ylim(0, 14)
    a.set_ylabel("Clone-specific share of response (%)"); a.set_title("C  Plasticity vs calibrated floor")
    a.legend(handles=[Line2D([], [], marker="s", ls="", color=C["m"], label="observed"), plt.Rectangle((0, 0), 1, 1, color=C["band"], label="no-plasticity floor")],
             fontsize=6.2, frameon=False, loc="upper left")
    # D melanoma vs budget
    a = ax[3]; clean(a); M = R["MeRLin"]; tr = M["translation_5seeds"]
    for dd, (lo, hi) in enumerate(M["budget"].values()):
        a.axhspan(lo, hi, xmin=.02, xmax=.98, color=[C["shade"], "#dde6f0", "#ccd9e8", "#bccce0"][dd], lw=0, zorder=0)
        a.text(-.45, np.sqrt(lo * hi), f"budget, {dd} division{'s' if dd != 1 else ''}", fontsize=5.9, va="center", ha="left", color=C["ink2"])
    vals = [("vs\nuntreated", [M["identity"]], C["b"], "o"), ("translation\n(5 splits)", tr, C["o"], "D"), ("axis\nstretch", [M["axis_stretch"]], C["v"], "^")]
    for i, (lab, v, col, mk) in enumerate(vals):
        a.scatter(np.full(len(v), i), v, s=26, color=col, marker=mk, zorder=3)
    a.set_yscale("log"); a.set_ylim(2, 100); a.set_xlim(-.5, 2.5); a.set_xticks(range(3), [v[0] for v in vals], fontsize=6.4)
    a.set_ylabel("$M_L$"); a.set_title("D  Melanoma (WM4237-1), BRAFi+MEKi")
    fig.tight_layout(); save(fig, "fig_generalization")


if __name__ == "__main__":
    for f in [fig_overview, fig_power, fig_generalization]:
        f(); print(f.__name__, "ok")
