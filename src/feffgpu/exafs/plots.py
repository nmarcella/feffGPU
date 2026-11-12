"""Figures of spectra: a measured chi(k) against calculated ones, in k space and in R space."""
from __future__ import annotations
import numpy as np
import torch

BLUE, ORANGE, AQUA, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [BLUE, ORANGE, AQUA, NEUTRAL]


def style_axes(ax):
    """The package's plain axis style: a light grid, no top and right spines."""
    ax.grid(color=GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_comparison(comparisons, models, labels=None, out=None, title="", rmax_plot=None):
    """One row per spectrum: ``k^w chi(k)`` in the k window and the magnitude of chi(R), data in black and each model in its color.

    Parameters
    ----------
    comparisons : list of SpectrumComparison
    models : list of dict
        One dict per model: ``{comparison.name: calculated chi (NK,)}`` (unscaled; S0^2 and the shift of the
        comparison are applied here).
    labels : list of str, optional
        One label per model.
    out : str or Path, optional
        File to save (PNG, PDF, SVG by extension). The figure is returned in any case.
    title : str
    rmax_plot : float, optional
        Upper limit of the R axis (A); default the R window's upper limit plus 1 A, at least 6 A.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = labels or [f"model {i + 1}" if len(models) > 1 else "model" for i in range(len(models))]
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(11, 3.4 * len(comparisons)), squeeze=False)
    for i, c in enumerate(comparisons):
        k, r = c.k, c.r
        sel = (k >= c.window[0] - c.dk / 2) & (k <= c.window[1] + c.dk / 2)
        axes[i][0].plot(k[sel], (k ** c.kweight * c.data)[sel], color=INK, lw=1.8, label="data")
        axes[i][1].plot(r, np.abs(c.chi_r(torch.tensor(c.data, dtype=torch.float32))[0].cpu().numpy()), color=INK, lw=1.8, label="data")
        for m, lab, col in zip(models, labels, SERIES):
            with torch.no_grad():
                chi = c.model(torch.as_tensor(np.asarray(m[c.name]), device=c.device, dtype=torch.float32)[None])[0].cpu().numpy()
            axes[i][0].plot(k[sel], (k ** c.kweight * chi)[sel], color=col, lw=1.2, label=lab)
            axes[i][1].plot(r, np.abs(c.chi_r(torch.tensor(chi, dtype=torch.float32))[0].cpu().numpy()), color=col, lw=1.2, label=lab)
        axes[i][0].set_xlabel("k (1/Å)"); axes[i][0].set_ylabel(f"{c.name}: k^{c.kweight} chi(k)")
        axes[i][1].set_xlabel("R (Å)"); axes[i][1].set_ylabel(f"{c.name}: |chi(R)|")
        axes[i][1].set_xlim(0, rmax_plot or max(6.0, c.r_range[1] + 1))
        for x in c.r_range:
            axes[i][1].axvline(x, color=NEUTRAL, lw=0.8, ls=":")
        axes[i][0].legend(fontsize=7, frameon=False); axes[i][1].legend(fontsize=7, frameon=False)
        for ax in axes[i]:
            style_axes(ax)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
