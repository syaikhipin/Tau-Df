#!/usr/bin/env python3
"""R020: IEEE paper figures for the kernel-learnable paper.

Reads frozen result JSONs from experiments/E1/results and E2/results, writes
PDF (+PNG for inspection) into paper/figures/.

Palette (validated, dataviz method, light mode, --pairs all):
  #2a78d6 blue  |  #eb6834 orange  |  #1baf7a aqua (contrast WARN 2.74:1
  -> relief rule: every aqua mark carries a visible direct label)
Secondary encoding for CVD / grayscale: distinct markers per series and
hatching on the leaf/aqua bars. One axis per panel; no dual axes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "experiments" / "E1" / "results"
E2 = ROOT / "experiments" / "E2" / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#8a8a8a"
CHANNEL_COLORS = {"RBF (chosen)": BLUE, "ARD (learned)": ORANGE, "leaf / tree": AQUA}
CHANNEL_MARKERS = {"RBF (chosen)": "o", "ARD (learned)": "s", "leaf / tree": "^"}

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_W = 3.45   # IEEE single-column width (in)
FULL_W = 7.16  # IEEE double-column width (in)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


# ---------------------------------------------------------------- Fig 1: ledger
def fig1_ledger() -> None:
    d = json.loads((E1 / "m1_seeds.json").read_text())["ledger"]
    chans = ["RBF (chosen)", "ARD (learned)", "leaf / tree"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.15), sharey=True)
    for ax, ds, tag in zip(axes, ["california", "taiwan"], ["(a) California", "(b) Taiwan credit"]):
        sup = [next(r["support_r2_mean"] for r in d[ds] if r["name"] == c) for c in chans]
        qry = [next(r["query_r2_mean"] for r in d[ds] if r["name"] == c) for c in chans]
        sup_s = [next(r["support_r2_std"] for r in d[ds] if r["name"] == c) for c in chans]
        qry_s = [next(r["query_r2_std"] for r in d[ds] if r["name"] == c) for c in chans]
        gap = [next(r["gap_mean"] for r in d[ds] if r["name"] == c) for c in chans]
        x = np.arange(3)
        w = 0.36
        ax.bar(x - w / 2 - 0.01, sup, w, color=BLUE, label="support $R^2$")
        ax.bar(x + w / 2 + 0.01, qry, w, color=ORANGE, label="query $R^2$",
               yerr=qry_s, error_kw=dict(lw=0.7, capsize=1.5))
        for xi, (s, g) in enumerate(zip(sup, gap)):
            ax.annotate(f"$\\Delta$ {g:.2f}", (xi, max(s, 0.1) + 0.13),
                        ha="center", fontsize=6.5, color="#444444")
        ax.set_xticks(x, ["RBF", "ARD", "leaf"], fontsize=7)
        ax.set_title(tag)
        ax.set_ylim(0, 1.13)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("$R^2$ (800/800 split)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.10), fontsize=6.5, handlelength=1.2)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig1_ledger")


# ------------------------------------------------------------- Fig 2: fusion
def fig2_fusion() -> None:
    f = json.loads((E1 / "m1_seeds.json").read_text())["fusion"]["california"]
    names = f["names"]
    w_sure = np.mean(f["in_sample"]["weights"], axis=0)
    w_qry = np.mean(f["query"]["weights"], axis=0)
    rmse_sure = np.mean(f["in_sample"]["test_rmse"])
    rmse_qry = np.mean(f["query"]["test_rmse"])

    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    rows = [("SURE-selected", w_sure, rmse_sure), ("query-selected", w_qry, rmse_qry)]
    ys = [0, 1]
    left = np.zeros(2)
    hatch = {0: None, 1: None, 2: "///"}  # hatch the leaf (aqua) segment
    for ci, ch in enumerate(names):
        vals = [w[ci] for _, w, _ in rows]
        ax.barh(ys, vals, left=left, height=0.52, color=CHANNEL_COLORS[ch],
                edgecolor="white", linewidth=1.2,
                hatch=hatch[ci], label=ch.replace(" (chosen)", "").replace(" / tree", ""))
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.07:  # visible direct label on every wide segment (incl. aqua relief)
                ax.text(l + v / 2, yi, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 0.3 else "#333333")
        left += np.array(vals)
    for (_, _, rm), yi in zip(rows, ys):
        ax.text(1.02, yi, f"RMSE {rm:.3f}", va="center", fontsize=6.5, color="#444444")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fusion weight (mean of 3 seeds, California)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=3,
              handlelength=1.1, columnspacing=0.9)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig2_fusion")


# --------------------------------------------------- Fig 3: tau-df (novelty)
def fig3_tau() -> None:
    rows = json.loads((E2 / "m2_gate.json").read_text())["rows"]
    real = json.loads((E2 / "m3_real_data.json").read_text())
    THR = 20.0

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.25),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.0]))

    # (a) synthetic tau-df, dose-ordered
    ax = axes[0]
    order = [r for r in rows if "fixed" in r["setting"]]
    order += [r for r in rows if "CONTROL" in r["setting"]]
    order += [r for r in rows if "LABEL-FITTED" in r["setting"]]
    labels, vals, cols, hat = [], [], [], []
    for r in order:
        s = r["setting"]
        if "fixed" in s:
            labels.append(s.replace("RBF ell=", "$\\ell$=").replace(" (fixed)", ""))
            cols.append(BLUE)
        elif "CONTROL" in s:
            labels.append("leaf control")
            cols.append(GRAY)
        else:
            labels.append(s.replace("leaf LABEL-FITTED d", "depth ").replace("epth=", " "))
            cols.append(AQUA)
        vals.append(r["tau_df"])
        hat.append(r["df_boot"] - r["tr_S"])
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white", linewidth=0.4)
    ax.axvline(THR, color="#333333", lw=0.9, ls="--")
    ax.text(THR + 2, len(vals) - 0.55, "threshold $\\tau{=}20$", fontsize=6, color="#333333")
    for yi, v in zip(y, vals):
        lab = f"{v:+.1f}" if abs(v) < 5 else f"{v:+.0f}"
        if v >= 0:  # right of the bar tip
            ax.text(v + 3, yi, lab, va="center", ha="left", fontsize=6)
        else:       # right of the zero line (avoids colliding with y-tick labels)
            ax.text(4, yi, lab, va="center", ha="left", fontsize=6)
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_xlabel("$\\tau$-df $= \\widehat{df}_{\\mathrm{boot}} - \\mathrm{tr}\\,S$")
    ax.set_title("(a) synthetic (600 pts)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    # (b) SURE error vs tau-df
    ax = axes[1]
    fam = lambda s: ("fixed" if "fixed" in s else "control" if "CONTROL" in s else "lf")
    for key, col, mk, lab in [("fixed", BLUE, "o", "fixed RBF"),
                              ("control", GRAY, "D", "leaf control"),
                              ("lf", AQUA, "^", "label-fitted leaf")]:
        pts = [(r["tau_df"], r["sure_err_mean"]) for r in rows if fam(r["setting"]) == key]
        xs, ys_ = zip(*pts)
        ax.scatter(xs, ys_, s=16, color=col, marker=mk, zorder=3,
                   edgecolor="white", linewidth=0.4, label=lab)
    lf = [r for r in rows if fam(r["setting"]) == "lf"]
    for r in lf:  # direct labels on the aqua points (contrast relief)
        depth = r["setting"].split("=")[-1]
        ax.annotate(depth, (r["tau_df"], r["sure_err_mean"]),
                    textcoords="offset points", xytext=(3, 2), fontsize=6, color="#1a6b4d")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")
    ax.set_xlabel("$\\tau$-df")
    ax.set_ylabel("|SURE $-$ true risk| (mean over $\\lambda$)")
    ax.set_title("(b) SURE error vs. $\\tau$-df")
    ax.legend(loc="upper left", fontsize=6, handlelength=1.0)
    rho = json.loads((E2 / "m2_gate.json").read_text())["spearman_tau_df_meanerr"]["rho"]
    ax.text(0.97, 0.04, f"Spearman $\\rho = {rho:.2f}$", transform=ax.transAxes,
            ha="right", fontsize=6.5, color="#333333")
    ax.set_axisbelow(True)

    # (c) real data
    ax = axes[2]
    chans = ["RBF (chosen)", "ARD (learned)", "leaf / tree"]
    x = np.arange(2)
    w = 0.24
    for ci, ch in enumerate(chans):
        vals = [real[ds][ci]["tau_df"] for ds in ["california", "taiwan"]]
        ax.bar(x + (ci - 1) * (w + 0.03), vals, w, color=CHANNEL_COLORS[ch],
               label=ch.replace(" (chosen)", "").replace(" / tree", ""),
               hatch="///" if ch == "leaf / tree" else None,
               edgecolor="white", linewidth=0.4)
    ax.axhline(THR, color="#333333", lw=0.9, ls="--", zorder=4)
    ax.axhline(0, color="#999999", lw=0.5)
    bot = min(real[ds][ci]["tau_df"] for ds in ["california", "taiwan"]
              for ci in range(len(chans)))
    ax.set_ylim(bot - 14, 88)
    for ci, ch in enumerate(chans):
        vals = [real[ds][ci]["tau_df"] for ds in ["california", "taiwan"]]
        for xi, v in zip(x + (ci - 1) * (w + 0.03), vals):
            if v >= THR:  # tall bars: label sits inside, clear of the axes edge
                ax.text(xi, v - 3.0, f"{v:+.0f}", ha="center", va="top",
                        fontsize=6.5, color="white", weight="bold")
            elif v >= 0:
                ax.text(xi, v + 1.6, f"{v:+.0f}", ha="center", va="bottom",
                        fontsize=6, color="#444444")
            else:
                ax.text(xi, v - 1.6, f"{v:+.0f}", ha="center", va="top",
                        fontsize=6, color="#444444")
    ax.text(1.63, THR + 1.0, "$\\tau{=}20$", ha="right", va="bottom",
            fontsize=6, color="#333333")
    ax.set_xticks(x, ["California", "Taiwan"])
    ax.set_ylabel("$\\tau$-df")
    ax.set_title("(c) real support folds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              fontsize=6, handlelength=1.0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.0)
    save(fig, "fig3_tau")


# --------------------------------------------------- Fig 4: SURE vs true risk
def fig4_sure_curves() -> None:
    c = json.loads((E2 / "m2_gate.json").read_text())["curves"]
    picks = [("RBF ell=1.0 (fixed)", "(a) fixed kernel: SURE tracks risk"),
             ("leaf LABEL-FITTED depth=3", "(b) label-fitted leaf: SURE under-credits")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.2), sharey=True)
    for ax, (key, title) in zip(axes, picks):
        cu = c[key]
        lams = np.array(cu["lams"])
        ax.semilogx(lams, cu["true"], color=BLUE, marker="o", markevery=2,
                    markersize=3.2, label="true risk")
        ax.semilogx(lams, cu["sure"], color=ORANGE, ls="--", marker="s", markevery=2,
                    markersize=3.2, label="SURE")
        ax.semilogx(lams, cu["insample"], color=AQUA, ls=":", marker="^", markevery=2,
                    markersize=3.2, label="in-sample")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("$\\lambda$")
        ax.set_axisbelow(True)
        # direct labels at the right ends (aqua relief)
        for arr, col in [(cu["true"], BLUE), (cu["sure"], ORANGE), (cu["insample"], "#1a6b4d")]:
            ax.annotate("", (lams[-1], arr[-1]), color=col, fontsize=6)
    axes[0].set_ylabel("risk (MSE)")
    axes[0].set_ylim(bottom=0)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.14), fontsize=6.5, handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_sure_curves")


# ------------------------------------------------------- Fig 5: capacity map
def fig5_capacity() -> None:
    d = json.loads((E1 / "r018_capacity_map.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    ax = axes[0]
    K = [r["K"] for r in d["free"]]
    gap = [r["gap"] for r in d["free"]]
    ax.semilogx(K, gap, color=BLUE, marker="o")
    ax.set_title("(a) free atoms (leaf depth sweep)")
    ax.set_xlabel("atoms $K$")
    for k, g in zip(K, gap):
        ax.annotate(f"{g:.2f}", (k, g), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=5.8, color="#15507f")
    ax = axes[1]
    H = [r["H"] for r in d["banks"]]
    gap = [r["gap"] for r in d["banks"]]
    ax.semilogx(H, gap, color=ORANGE, marker="s")
    ax.set_title("(b) convex bank of $H$ RBFs")
    ax.set_xlabel("banks $H$")
    for h, g in zip(H, gap):
        ax.annotate(f"{g:.2f}", (h, g), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=5.8, color="#a04314")
    axes[0].set_ylabel("support$-$query gap")
    axes[0].set_ylim(0.2, 1.05)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig5_capacity")


if __name__ == "__main__":
    fig1_ledger()
    fig2_fusion()
    fig3_tau()
    fig4_sure_curves()
    fig5_capacity()
    print("all figures written to", OUT)
