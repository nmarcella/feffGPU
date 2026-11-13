#!/usr/bin/env python3
"""The figures of the element-by-element validation of feffgpu against FEFF 8.5L (tools/validate_elements.py).

They are drawn from data/validation/elements.json and elements.npz (for every element and each of its K, L3 and M5
edges: the R-factor between the two codes and the two chi(k)). Nothing is computed here.

    python tools/plot_validation_elements.py

writes into docs/validation/:

  agreement_all_elements.png    the periodic table with the R-factor of each edge of each element, a parity plot of the
                                two codes' spectra, and the R-factor against the atomic number
  elements_K.png, elements_L3.png, elements_M5.png
                                one panel per element: the two spectra drawn over each other
  elements_table.md             the table of all cases
  elements_summary.md           the numbers of the page as a list (both are included by docs/validation/elements.md)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
EDGES = ("K", "L3", "M5")
EDGE_COLOR = {"K": BLUE, "L3": ORANGE, "M5": AQUA}
EDGE_MARKER = {"K": "o", "L3": "D", "M5": "^"}
TEST_LIMIT = 2e-5                                           # the limit of the regression tests
RAMP = ["#eef4fc", "#cfe0f6", "#a3c4ee", "#6ea2e3", "#2a78d6", "#1b56a3", "#0f3568"]      # one hue, light to dark
LOG_LO, LOG_HI = -8.0, -3.0                                 # the range of log10(R) that the ramp covers
# where the label of a case above the limit goes in the lower panel: (shift in Z, factor on R, alignment, text or None for
# "element edge"); a case that is not listed gets its label to the right of its mark
LABELS = {"090_Th_K": (-1.3, 0.80, "right", "Th K, L3"), "090_Th_L3": None}
NO_SCF_NOTE = "overlapped potentials without the self-consistency loop (FEFF 8.5L stops in its own loop for this element)"


def sci(x):
    """1.9e-06 written as 1.9e-6."""
    m, e = f"{x:.1e}".split("e")
    return f"{m}e{int(e)}"


def table_cell(z):
    """(row, column) of an element in the 18-column periodic table; the lanthanides and actinides in two rows below."""
    if z == 1:
        return 1, 1
    if z == 2:
        return 1, 18
    for row, first in ((2, 3), (3, 11)):
        if first <= z <= first + 7:
            i = z - first
            return row, 1 + i if i < 2 else 11 + i
    if 19 <= z <= 36:
        return 4, z - 18
    if 37 <= z <= 54:
        return 5, z - 36
    if z in (55, 56):
        return 6, z - 54
    if 57 <= z <= 71:
        return 8.45, 3 + (z - 57)
    if 72 <= z <= 86:
        return 6, z - 68
    if z in (87, 88):
        return 7, z - 86
    return 9.45, 3 + (z - 89)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/validation"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    sys.path.insert(0, str(ROOT / "src"))
    from feffgpu.io import SYMBOLS
    out = Path(a.out)
    rows = json.load(open(ROOT / "data/validation/elements.json"))["cases"]
    data = np.load(ROOT / "data/validation/elements.npz")
    by = {(r["Z"], r["edge"]): r for r in rows}
    done = [r for r in rows if r["status"] == "compared"]
    failed = [r for r in rows if r["status"] != "compared"]
    rr = np.array([r["r_factor_r_space"] for r in done])
    cmap = LinearSegmentedColormap.from_list("one_hue", RAMP)
    norm = Normalize(LOG_LO, LOG_HI, clip=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": NEUTRAL, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED, "text.color": INK})

    # =================== figure 1: periodic table, parity plot, R-factor against the atomic number ===================
    fig = plt.figure(figsize=(10.0, 9.3), dpi=160)
    n_el = len({r["Z"] for r in done})
    fig.text(0.04, 0.975, "feffgpu reproduces FEFF 8.5L for every element and edge", fontsize=12.5, weight="bold", va="center")
    line2 = (f"{len(done)} cases compared: {n_el} elements, K, L3 and M5 edges. Median R-factor between the two codes {sci(np.median(rr))}; "
             f"{int((rr < TEST_LIMIT).sum())} of {len(done)} below 2e-5; largest {sci(rr.max())}.")
    if failed:
        line2 += f" Not compared: {len(failed)} (see the table)."
    no_scf = sorted({r["Z"] for r in rows if not r.get("self_consistent", True)})
    without = f" ({', '.join(SYMBOLS[z] for z in no_scf)}: without)" if no_scf else ""
    fig.text(0.04, 0.951, line2 + f"\nEach case: fcc cluster of the element (87 atoms), self-consistent potentials{without}, paths to four legs; "
             "R-factor in R space, k 3 to 19 1/Å, R 1 Å to the path cutoff.", fontsize=7.6, color=MUTED, va="center", linespacing=1.45)

    ax = fig.add_axes([0.04, 0.385, 0.92, 0.535])
    ax.set_xlim(0.4, 18.6)
    ax.set_ylim(10.15, 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    sub = 0.29                                              # width of one of the three squares of an element
    for z in range(1, 100):
        row, col = table_cell(z)
        ax.text(col, row - 0.30, SYMBOLS[z], ha="center", va="center", fontsize=7.2, weight="bold", color=INK)
        ax.text(col - 0.43, row - 0.30, str(z), ha="left", va="center", fontsize=4.2, color=MUTED)
        for j, e in enumerate(EDGES):
            x0, y0 = col - 1.5 * sub + j * sub, row - 0.10
            r = by.get((z, e))
            if r is None:                                   # the element has no such edge
                ax.add_patch(Rectangle((x0 + 0.02, y0), sub - 0.04, 0.38, facecolor="none", edgecolor=GRID, lw=0.5))
            elif r["status"] != "compared":
                ax.add_patch(Rectangle((x0 + 0.02, y0), sub - 0.04, 0.38, facecolor="white", edgecolor=NEUTRAL, lw=0.6, hatch="//////"))
            else:
                ax.add_patch(Rectangle((x0 + 0.02, y0), sub - 0.04, 0.38, facecolor=cmap(norm(np.log10(max(r["r_factor_r_space"], 1e-300)))),
                                       edgecolor="white", lw=0.5))
    ax.text(2.45, 8.45, "lanthanides", fontsize=6.5, color=MUTED, va="center", ha="right")
    ax.text(2.45, 9.45, "actinides", fontsize=6.5, color=MUTED, va="center", ha="right")
    # the free corner above the transition metals (columns 3 to 12, rows 1 to 3) holds the key, the color scale and the parity plot
    kx, ky = 3.0, 1.05
    ax.text(kx, ky - 0.52, "the three squares of an element", fontsize=7.0, color=MUTED, ha="left", va="center")
    for j, e in enumerate(EDGES):
        ax.add_patch(Rectangle((kx + j * 0.5 + 0.03, ky - 0.25), 0.44, 0.44, facecolor=RAMP[2], edgecolor="white", lw=0.6))
        ax.text(kx + j * 0.5 + 0.25, ky - 0.03, e, fontsize=6.6, ha="center", va="center", color=INK)
    ax.add_patch(Rectangle((kx + 0.03, ky + 0.50), 0.44, 0.36, facecolor="none", edgecolor=GRID, lw=0.6))
    ax.text(kx + 0.62, ky + 0.68, "no such edge", fontsize=6.6, color=MUTED, va="center")
    if failed:
        ax.add_patch(Rectangle((kx + 0.03, ky + 1.05), 0.44, 0.36, facecolor="white", edgecolor=NEUTRAL, lw=0.6, hatch="//////"))
        ax.text(kx + 0.62, ky + 1.23, "not compared", fontsize=6.6, color=MUTED, va="center")
    # the color scale
    cax = ax.inset_axes([5.75, 1.62, 3.9, 0.24], transform=ax.transData)
    grad = np.linspace(LOG_LO, LOG_HI, 256)[None]
    cax.imshow(grad, aspect="auto", cmap=cmap, norm=norm, extent=[LOG_LO, LOG_HI, 0, 1])
    cax.set_yticks([])
    cax.set_xticks([-8, -7, -6, -5, -4, -3])
    cax.set_xticklabels(["≤1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "≥1e-3"], fontsize=6.3)
    cax.axvline(np.log10(TEST_LIMIT), color=INK, lw=0.8)
    cax.text(np.log10(TEST_LIMIT), 1.3, "2e-5", fontsize=6.3, ha="center", va="bottom", color=INK)
    ax.text(5.75, 1.05, "R-factor between FEFF 8.5L and feffgpu", fontsize=7.0, color=MUTED, va="center", ha="left")
    cax.tick_params(length=2, width=0.5, pad=1.5)
    for sp in cax.spines.values():
        sp.set_visible(False)
    # the parity plot of all spectra
    pax = ax.inset_axes([10.45, 0.95, 1.75, 1.75], transform=ax.transData)
    xs, ys = [], []
    for r in done:
        k, f, g = data[f"{r['case']}_k"], data[f"{r['case']}_feff"], data[f"{r['case']}_feffgpu"]
        sel = k >= 2.0
        sc = np.abs(k[sel] ** 2 * f[sel]).max()
        xs.append(k[sel] ** 2 * f[sel] / sc)
        ys.append(k[sel] ** 2 * g[sel] / sc)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    pax.plot([-1.05, 1.05], [-1.05, 1.05], color=NEUTRAL, lw=0.6, zorder=1)
    pax.scatter(xs, ys, s=1.2, color=BLUE, alpha=0.25, linewidths=0, rasterized=True, zorder=2)
    pax.set_xlim(-1.05, 1.05)
    pax.set_ylim(-1.05, 1.05)
    pax.set_xticks([-1, 0, 1])
    pax.set_yticks([-1, 0, 1])
    pax.tick_params(labelsize=6.0, length=2, width=0.5, pad=1.2)
    pax.set_xlabel("FEFF 8.5L", fontsize=6.6, labelpad=1.0)
    pax.set_ylabel("feffgpu", fontsize=6.6, labelpad=0.0)
    ax.text(11.32, 0.62, f"k² chi(k) of all {len(done)} cases", fontsize=6.6, color=MUTED, va="center", ha="center")
    for side in ("top", "right"):
        pax.spines[side].set_visible(False)

    # the R-factor against the atomic number, one mark per edge
    bx = fig.add_axes([0.085, 0.075, 0.895, 0.265])
    for e in EDGES:
        zz = np.array([r["Z"] for r in done if r["edge"] == e])
        vv = np.array([r["r_factor_r_space"] for r in done if r["edge"] == e])
        bx.scatter(zz, vv, s=22, marker=EDGE_MARKER[e], color=EDGE_COLOR[e], edgecolor="white", linewidth=0.5, zorder=3)
    bx.axhline(TEST_LIMIT, color=NEUTRAL, lw=0.9, ls=(0, (4, 3)), zorder=1)
    bx.text(68.0, TEST_LIMIT * 1.35, "limit of the regression tests, 2e-5", ha="center", va="bottom", fontsize=7.2, color=MUTED)
    bx.set_yscale("log")
    lo = 10 ** np.floor(np.log10(max(rr.min(), 1e-13)))
    hi = 10 ** np.ceil(np.log10(max(rr.max(), 1e-4)))
    bx.set_ylim(lo / 2, hi * 4)
    bx.set_xlim(0, 100)
    ticks = [1, 3, 6, 8, 11, 14, 17, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 53, 55, 57, 60, 63, 66, 69, 72, 74, 76, 78, 80,
             82, 84, 86, 88, 90, 92, 94, 96, 99]
    bx.set_xticks(ticks)
    bx.set_xticklabels([f"{SYMBOLS[z]}\n{z}" for z in ticks], fontsize=6.0, color=INK, linespacing=1.15)
    bx.set_ylabel("R-factor between the two codes")
    bx.set_xlabel("absorbing element and atomic number", labelpad=3)
    bx.grid(axis="y", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.tick_params(length=2.5, width=0.6)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    handles = [Line2D([], [], ls="", marker=EDGE_MARKER[e], color=EDGE_COLOR[e], markeredgecolor="white", markersize=6, label=f"{e} edge") for e in EDGES]
    bx.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=7.5, handletextpad=0.2, columnspacing=1.2, labelcolor=MUTED,
              bbox_to_anchor=(0.0, 1.10))
    worst = done[int(np.argmax(rr))]
    bx.annotate(f"{worst['absorber']} {worst['edge']}: {sci(rr.max())}", xy=(worst["Z"], rr.max()), xytext=(worst["Z"] + 3.5, rr.max()), fontsize=7,
                color=MUTED, va="center", arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.6))
    for r in done:                                          # every other case above the limit is named next to its mark
        if r["r_factor_r_space"] < TEST_LIMIT or r is worst:
            continue
        place = LABELS.get(r["case"], (1.2, 1.0, "left", None))
        if place is None:
            continue
        dz, fy, ha, text = place
        bx.text(r["Z"] + dz, r["r_factor_r_space"] * fy, text or f"{r['absorber']} {r['edge']}", fontsize=6.5, color=MUTED, ha=ha, va="center")
    fig.savefig(out / "agreement_all_elements.png", dpi=160, facecolor="white")
    plt.close(fig)

    # =================== figures 2 to 4: one panel per element, per edge ===========================================
    for e in EDGES:
        cases = sorted([r for r in done if r["edge"] == e], key=lambda r: r["Z"])
        if not cases:
            continue
        ncol = 11
        nrow = int(np.ceil(len(cases) / ncol))
        fig = plt.figure(figsize=(10.0, 0.62 + 0.98 * nrow), dpi=160)
        top = 1 - 0.46 / fig.get_figheight()
        grid = fig.add_gridspec(nrow, ncol, hspace=0.50, wspace=0.08, left=0.012, right=0.988, top=top - 0.30 / fig.get_figheight(), bottom=0.012)
        fig.text(0.012, 1 - 0.17 / fig.get_figheight(), f"{e} edge of every element: FEFF 8.5L (black) and feffgpu (blue, on top)", fontsize=10.5,
                 weight="bold", va="center")
        fig.text(0.012, 1 - 0.38 / fig.get_figheight(), "k² chi(k) from k = 0 to 20 1/Å, each panel scaled to its own maximum; the number is the "
                 "R-factor between the two codes", fontsize=7.2, color=MUTED, va="center")
        for i, r in enumerate(cases):
            axp = fig.add_subplot(grid[i // ncol, i % ncol])
            k, f, g = data[f"{r['case']}_k"], data[f"{r['case']}_feff"], data[f"{r['case']}_feffgpu"]
            s = np.abs(k ** 2 * f).max()
            axp.plot(k, k ** 2 * f / s, color=INK, lw=1.7, solid_capstyle="round")
            axp.plot(k, k ** 2 * g / s, color=BLUE, lw=0.6)
            axp.set_xlim(0, 20)
            axp.set_ylim(-1.12, 1.12)
            axp.axis("off")
            axp.text(0.0, 1.03, r["absorber"], transform=axp.transAxes, fontsize=7.6, weight="bold", va="bottom")
            axp.text(1.0, 1.04, sci(r["r_factor_r_space"]), transform=axp.transAxes, fontsize=5.8, color=MUTED, va="bottom", ha="right")
        fig.savefig(out / f"elements_{e}.png", dpi=160, facecolor="white")
        plt.close(fig)

    # =================== the table =================================================================================
    lines = ["| Z | element | edge | R-factor (R space) | R-factor (k space) | Fermi level difference (hartree) | path counts equal | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["Z"], EDGES.index(r["edge"]))):
        if r["status"] == "compared":
            note = "" if r.get("self_consistent", True) else NO_SCF_NOTE
            lines.append(f"| {r['Z']} | {r['absorber']} | {r['edge']} | {sci(r['r_factor_r_space'])} | {sci(r['r_factor_k_space'])} | "
                         f"{sci(r['fermi_level_difference_hartree'])} | {'yes' if r['same_path_counts'] else 'no'} | {note} |")
        else:
            lines.append(f"| {r['Z']} | {r['absorber']} | {r['edge']} | | | | | {r['status']} |")
    (out / "elements_table.md").write_text("\n".join(lines) + "\n")
    same = sum(1 for r in done if r["same_path_counts"])

    # =================== the numbers of the page, as a list (included by docs/validation/elements.md) ===============
    def diff(r):                                            # degeneracy-weighted path counts, feffgpu minus FEFF 8.5L, per number of legs
        return {k: r["paths_feffgpu"].get(k, 0.0) - r["paths_feff"].get(k, 0.0) for k in sorted(set(r["paths_feff"]) | set(r["paths_feffgpu"]))}

    sc = [r for r in done if r.get("self_consistent", True)]
    efs = np.array([abs(r["fermi_level_difference_hartree"]) for r in sc])
    worst = done[int(np.argmax(rr))]
    above = sorted([r for r in done if r["r_factor_r_space"] >= TEST_LIMIT], key=lambda r: -r["r_factor_r_space"])
    notsame = [r for r in done if not r["same_path_counts"]]
    two_leg = sum(1 for r in notsame if abs(diff(r).get("2", 0.0)) > 0.5)
    largest = max((max(abs(v) for v in diff(r).values()) for r in notsame), default=0.0)
    total = {r["case"]: sum(r["paths_feff"].values()) for r in done}
    text = [f"- **{len(done)} cases compared**: {n_el} elements (Z = {min(r['Z'] for r in done)} to {max(r['Z'] for r in done)}); "
            + ", ".join(f"{sum(1 for r in done if r['edge'] == e)} {e} edges" for e in EDGES) + "."
            + (f" Not compared: {len(failed)} (see the table)." if failed else ""),
            f"- **R-factor between the two codes** (R space, k from 3 to 19 1/Å, R from 1 Å to the path cutoff, k² weight): median "
            f"{sci(np.median(rr))}; {int((rr < 1e-5).sum())} of {len(done)} below 1e-5; {int((rr < TEST_LIMIT).sum())} below 2e-5, the limit that "
            f"the regression tests enforce; the largest is {sci(rr.max())} ({worst['absorber']} {worst['edge']}).",
            "- **By edge**: " + "; ".join(
                f"{e}: median {sci(np.median([r['r_factor_r_space'] for r in done if r['edge'] == e]))}, largest "
                f"{sci(max(r['r_factor_r_space'] for r in done if r['edge'] == e))}" for e in EDGES) + ".",
            f"- **R-factor in k space** (k from 2 1/Å, k² weight, no Fourier transform): median "
            f"{sci(np.median([r['r_factor_k_space'] for r in done]))}, largest {sci(max(r['r_factor_k_space'] for r in done))}.",
            f"- **Paths**: the number of paths, weighted by degeneracy, is identical in {same} of the {len(done)} cases. "
            + (f"In the other {len(notsame)} the two codes differ by at most {largest:.0f} in the weighted count "
               f"(of {min(total[r['case']] for r in notsame):.0f} to {max(total[r['case']] for r in notsame):.0f} paths), "
               + ("never in the single-scattering paths: " if two_leg == 0 else f"in the single-scattering paths in {two_leg} cases: ")
               + "these are three- and four-leg paths whose estimated importance sits at the threshold of the path filter." if notsame else ""),
            f"- **Fermi level** of the self-consistent potentials ({len(sc)} cases): the two codes differ by {sci(np.median(efs))} hartree in the "
            f"median and by {sci(efs.max())} hartree at most."]
    if above:
        text += ["", f"The {len(above)} cases above 2e-5:" if len(above) > 1 else "The one case above 2e-5:", "",
                 "| element and edge | R-factor | Fermi level difference (hartree) | path counts, feffgpu minus FEFF 8.5L |", "|---|---|---|---|"]
        for r in above:
            d = {k: v for k, v in diff(r).items() if abs(v) > 0.5}
            text.append(f"| {r['absorber']} {r['edge']} | {sci(r['r_factor_r_space'])} | {sci(r['fermi_level_difference_hartree'])} | "
                        + (", ".join(f"{v:+.0f} with {k} legs" for k, v in d.items()) if d else "equal") + " |")
    (out / "elements_summary.md").write_text("\n".join(text) + "\n")
    print(f"{len(done)} compared, {len(failed)} not; median R {np.median(rr):.2e}, largest {rr.max():.2e}, below 2e-5: {int((rr < TEST_LIMIT).sum())}; "
          f"path counts equal in {same}; largest Fermi-level difference {max(abs(r['fermi_level_difference_hartree']) for r in done):.1e} Ha")


if __name__ == "__main__":
    main()
