"""Figure of comparisons with EvAX: one row per spectrum with k^2 chi(k) (EvAX's target, EvAX's fit, this package's
fit), the magnitude of the Fourier transform, and the distribution of neighbor distances in the two final structures.
Reads what `tools/compare_with_evax.py` wrote into the run folders.

usage: python tools/plot_evax_comparison.py <out.png> "<run folder>:<spectrum name>:<title>" ...
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from feffgpu.analysis.pairs import distances_in_range
from feffgpu.exafs import FitWindow, SpectrumComparison
from feffgpu.rmc import load_config

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
rows = [item.split(":", 2) for item in sys.argv[2:]]
fig, axes = plt.subplots(len(rows), 3, figsize=(11.5, 2.35 * len(rows)), squeeze=False, gridspec_kw=dict(width_ratios=[1.35, 1, 1.15]))
for r, (folder, edge, title) in enumerate(rows):
    folder = Path(folder)
    z = np.load(folder / f"compare_{edge}.npz")
    k, ours, theirs, target = z["k"], z["ours"], z["evax"], z["target"]
    rep = json.loads((folder / "compare.json").read_text())
    row = [e for e in rep["edges"] if e["name"] == edge][0]
    cfg = load_config(folder / "rmc.toml")
    w = [e for e in cfg.sample[0].edge if e.name == edge][0].window
    sel = (k >= w.kmin) & (k <= w.kmax)
    ax = axes[r, 0]
    ax.plot(k[sel], (target * k ** 2)[sel], color=INK, lw=1.3, label="EvAX's target (the data)")
    ax.plot(k[sel], (theirs * k ** 2)[sel], color=ORANGE, lw=1.1, label="EvAX's fit")
    ax.plot(k[sel], (ours * k ** 2)[sel], color=BLUE, lw=1.1, label="rmc-exafs fit")
    ax.set_ylabel("k² χ(k) (Å⁻²)")
    ax.set_title(f"{title}   misfit: EvAX {row['misfit_evax_own']:.3f}, ours {row['misfit_ours']:.3f}", loc="left", fontsize=9, color=INK)
    cmp = SpectrumComparison("x", k, target, FitWindow(kmin=w.kmin, kmax=w.kmax, dk=1.0, kweight=w.kweight, rmin=w.rmin, rmax=w.rmax), device="cpu")
    ax = axes[r, 1]
    for c, color, lw in ((target, INK, 1.3), (theirs, ORANGE, 1.1), (ours, BLUE, 1.1)):
        ax.plot(cmp.r, np.abs(cmp.chi_r(c)[0].numpy()), color=color, lw=lw)
    ax.set_xlim(0, w.rmax + 1.0); ax.set_ylabel("|χ(R)| (Å⁻³)")
    ax.axvspan(w.rmin, w.rmax, color=GRID, alpha=0.35, lw=0)
    a = np.load(folder / "compare_atoms.npz")
    absorber = [e for e in cfg.sample[0].edge if e.name == edge][0].absorber
    ax = axes[r, 2]
    bins = np.arange(1.2, 6.0 + 0.03, 0.03)
    elements = sorted(set(a["symbols"].tolist()))
    for pos, sym, cell, color in ((a["evax"], a["symbols_evax"], a["cell_evax"], ORANGE), (a["ours"], a["symbols"], a["cell"], BLUE)):
        for j, el in enumerate(elements):
            d, n_centers = distances_in_range(pos[None], sym, cell, absorber, el, 1.2, 6.0)
            if len(d):
                h, _ = np.histogram(d, bins=bins)
                ax.plot(0.5 * (bins[1:] + bins[:-1]), h / max(n_centers, 1) / 0.03, color=color, lw=1.0, ls=("-", "--", ":")[j % 3])
    ax.set_xlim(1.2, 6.0); ax.set_ylim(0, None); ax.set_ylabel("neighbors per Å")
    ax.text(0.02, 0.95, ", ".join(f"{absorber}-{el} {('solid', 'dashed', 'dotted')[j % 3]}" for j, el in enumerate(elements)), transform=ax.transAxes, va="top",
            fontsize=7.5, color=MUTED)
    for x in axes[r]:
        x.grid(color=GRID, lw=0.5); x.set_axisbelow(True)
axes[-1, 0].set_xlabel("k (Å⁻¹)"); axes[-1, 1].set_xlabel("R (Å)"); axes[-1, 2].set_xlabel("distance from the absorber (Å)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / len(rows) - 0.005))
fig.savefig(sys.argv[1], dpi=170)
print("saved", sys.argv[1])
