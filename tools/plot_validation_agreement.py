#!/usr/bin/env python3
"""The figure "feffgpu against FEFF 8.5L, element by element and edge by edge".

It is drawn from the stored spectra of the random-environment validation (data/validation/agreement.npz and
agreement.json: for every environment the chi(k) of FEFF 8.5L and of feffgpu from the same feff.inp). Nothing is
computed here; to regenerate the spectra themselves, run tools/validate_random.py with a FEFF 8.5L build.

    python tools/plot_validation_agreement.py            # writes docs/validation/agreement_by_element_and_edge.png

Top: the R-factor between the two codes for every environment, labeled by absorbing element and edge. Below: one
panel per environment with the two spectra drawn over each other.
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
EDGE_COLOR = {"K": BLUE, "L2": ORANGE, "L3": ORANGE, "M5": AQUA}
EDGE_GROUP = {"K": "K edge", "L2": "L2 and L3 edges", "L3": "L2 and L3 edges", "M5": "M5 edge"}
EDGE_MARKER = {"K": "o", "L2": "s", "L3": "D", "M5": "^"}
TEST_LIMIT = 2e-5                                           # the limit of tests/test_validation_cases.py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/validation/agreement_by_element_and_edge.png"))
    ap.add_argument("--columns", type=int, default=6)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = json.load(open(ROOT / "data/validation/agreement.json"))["cases"]
    data = np.load(ROOT / "data/validation/agreement.npz")
    order = {"K": 0, "L2": 1, "L3": 2, "M5": 3}
    cases = sorted(meta, key=lambda m: (m["Z"], order[m["edge"]], m["case"]))
    n, ncol = len(cases), a.columns
    nrow = int(np.ceil((n + 1) / ncol))                     # one free slot for the key of the lines
    r = np.array([m["r_factor_r_space"] for m in cases])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": NEUTRAL, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(10.0, 2.9 + 1.55 * nrow), dpi=160)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.25, 1.55 * nrow], hspace=0.20, left=0.065, right=0.985, top=0.918, bottom=0.042)

    # ---- the R-factor of every environment ----------------------------------------------------------------------
    ax = fig.add_subplot(outer[0])
    x = np.arange(n)
    for edge in ("K", "L2", "L3", "M5"):
        sel = [i for i, m in enumerate(cases) if m["edge"] == edge]
        ax.scatter(x[sel], r[sel], s=44, marker=EDGE_MARKER[edge], color=EDGE_COLOR[edge], edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(TEST_LIMIT, color=NEUTRAL, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(n - 0.5, TEST_LIMIT * 1.35, "limit of the regression tests, 2e-5", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    worst = int(np.argmax(r))
    ax.annotate(f"{cases[worst]['absorber']} {cases[worst]['edge']} in a random packing with an ion (ION card): {r[worst]:.1e}",
                xy=(worst, r[worst]), xytext=(worst + 1.2, r[worst] * 0.9), fontsize=7.5, color=MUTED, va="center",
                arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.6))
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e-2)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m['absorber']} {m['edge']}" for m in cases], rotation=90, fontsize=7.5, color=INK)
    ax.set_ylabel("R-factor between the two codes")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], ls="", marker=EDGE_MARKER[e], color=EDGE_COLOR[e], markeredgecolor="white", markersize=7, label=lab)
               for e, lab in (("K", "K edge"), ("L2", "L2 edge"), ("L3", "L3 edge"), ("M5", "M5 edge"))]
    ax.legend(handles=handles, loc="lower right", ncol=4, frameon=False, fontsize=7.5, handletextpad=0.2, columnspacing=1.2, labelcolor=MUTED)
    n_el = len({m["absorber"] for m in cases})
    fig.text(0.065, 0.975, "feffgpu reproduces the chi(k) of FEFF 8.5L", fontsize=12.5, color=INK, weight="bold", va="center")
    fig.text(0.065, 0.949, f"{n} random environments, {n_el} absorbing elements from {cases[0]['absorber']} to {cases[-1]['absorber']}, K, L2, L3 and M5 edges: "
             f"median R-factor {np.median(r):.1e}, {int((r < TEST_LIMIT).sum())} of {n} below 2e-5.\n"
             "R-factor between the two codes: in R space, k from 3 to at most 23 1/Å, R from 1 Å to the path cutoff, k² weight.",
             fontsize=7.8, color=MUTED, va="center", linespacing=1.45)

    # ---- one panel per environment: the two spectra over each other ------------------------------------------------
    grid = outer[1].subgridspec(nrow, ncol, hspace=0.62, wspace=0.10)
    for i, m in enumerate(cases):
        axp = fig.add_subplot(grid[i // ncol, i % ncol])
        k, ref, gpu = data[f"{m['case']}_k"], data[f"{m['case']}_feff"], data[f"{m['case']}_feffgpu"]
        scale = np.abs(k ** 2 * ref).max()
        axp.axhline(0, color=GRID, lw=0.6, zorder=0)
        axp.plot(k, k ** 2 * ref / scale, color=INK, lw=2.3, solid_capstyle="round")
        axp.plot(k, k ** 2 * gpu / scale, color=BLUE, lw=0.85)
        axp.set_xlim(0, float(k[-1]))
        axp.set_ylim(-1.12, 1.12)
        axp.set_yticks([])
        step = 10 if k[-1] > 22 else 5
        axp.set_xticks(np.arange(0, k[-1] + 0.1, step))
        axp.tick_params(labelsize=6.3, length=2, width=0.5, pad=1.5)
        for side in ("top", "right", "left"):
            axp.spines[side].set_visible(False)
        axp.text(0.0, 1.20, f"{m['absorber']} {m['edge']}", transform=axp.transAxes, fontsize=9.2, weight="bold", color=INK, va="bottom")
        axp.text(1.0, 1.21, f"R = {m['r_factor_r_space']:.0e}", transform=axp.transAxes, fontsize=6.8, color=MUTED, va="bottom", ha="right")
        label = f"{m['elements']} {m['structure']}".replace("molecular-dynamics frame", "MD frame")
        axp.text(0.0, 1.04, label, transform=axp.transAxes, fontsize=6.3, color=MUTED, va="bottom")
    key = fig.add_subplot(grid[n // ncol, n % ncol])
    key.axis("off")
    key.plot([0.05, 0.42], [0.72, 0.72], color=INK, lw=2.3, solid_capstyle="round", transform=key.transAxes)
    key.plot([0.05, 0.42], [0.72, 0.72], color=BLUE, lw=0.85, transform=key.transAxes)
    key.text(0.05, 0.50, "black: FEFF 8.5L\nblue, on top: feffgpu", transform=key.transAxes, fontsize=7.5, color=INK, va="top", linespacing=1.5)
    key.text(0.05, 0.10, "each panel: k² chi(k) against k (1/Å),\nscaled to its own maximum;\nMD frame: a frame of a molecular\ndynamics run of bcc W", transform=key.transAxes, fontsize=6.8, color=MUTED, va="top",
             linespacing=1.4)
    fig.savefig(a.out, dpi=160, facecolor="white")
    print(f"written {a.out}: {n} environments, median R {np.median(r):.2e}, worst {r.max():.2e}")


if __name__ == "__main__":
    main()
