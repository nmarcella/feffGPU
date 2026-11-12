"""Figures of pair distributions."""
from __future__ import annotations
import numpy as np
from ..exafs.plots import SERIES, NEUTRAL, style_axes


def plot_pair_distributions(r, curves, out=None, xlabel="distance (Å)", ylabel="pairs per center per Å", title="", shells=None, xlim=None):
    """Partial pair distributions on one axis per pair.

    Parameters
    ----------
    r : array, shape (nbin,)
        Bin centers (A).
    curves : dict
        ``{pair label: {series label: g (nbin,)}}`` or ``{pair label: g}``; one panel per pair. The first series of
        a panel is drawn filled (a reference), the others as lines.
    out : str or Path, optional
        File to save. The figure is returned in any case.
    shells : dict, optional
        ``{pair label: [(r_min, r_max), ...]}``: ranges to mark with dotted lines (the ranges of a shell table).
    xlim : (float, float), optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(curves)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), squeeze=False)
    flat = axes.ravel()
    for ax, (pair, series) in zip(flat, curves.items()):
        series = series if isinstance(series, dict) else {"": series}
        for j, (lab, g) in enumerate(series.items()):
            if j == 0 and len(series) > 1:
                ax.fill_between(r, 0, g, color=NEUTRAL, alpha=0.35, lw=0, label=lab)
            else:
                ax.plot(r, g, color=SERIES[(j - 1) % len(SERIES)] if len(series) > 1 else SERIES[0], lw=1.3, label=lab or None)
        for lo, hi in (shells or {}).get(pair, []):
            ax.axvline(lo, color=NEUTRAL, lw=0.8, ls=":"); ax.axvline(hi, color=NEUTRAL, lw=0.8, ls=":")
        ax.set_xlabel(f"{pair} {xlabel}"); ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(*xlim)
        if len(series) > 1:
            ax.legend(fontsize=7, frameon=False)
        style_axes(ax)
    for ax in flat[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
