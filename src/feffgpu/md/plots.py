"""Figures of MD-EXAFS results: the averaged spectrum and its parts, convergence, radial distributions, angles.

Colors follow the package: the model in blue, parts in orange, aqua and neutral gray, data in black. R axes are
labeled "R (Å)".
"""
from __future__ import annotations
import numpy as np
import torch
from ..exafs.plots import AQUA, BLUE, GRID, INK, NEUTRAL, ORANGE, style_axes
from ..exafs.transform import ft, hanning_window, r_grid

__all__ = ["plot_average", "plot_convergence", "plot_rdf", "plot_peaks", "plot_angles"]

PART_COLORS = [ORANGE, AQUA, NEUTRAL, "#b5539a", "#7a6a2f", "#4f8fb8"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _chir(k, chi, kmin, kmax, dk, kweight):
    kt = torch.tensor(k, dtype=torch.float64)
    w = hanning_window(kt, torch.tensor([kmin], dtype=torch.float64), torch.tensor([kmax], dtype=torch.float64), dk=dk)[0]
    return np.abs(ft(torch.tensor(np.asarray(chi), dtype=torch.float64)[None], kt, w, kweight=kweight)[0].numpy())


def plot_average(result, out=None, window=None, title="", max_parts=6):
    """The averaged spectrum of one edge: k^w chi(k) with its standard error and spread, and the magnitude of chi(R), with the
    largest parts by path type.

    Parameters
    ----------
    result : dict
        From ``feffgpu.md.average.edge_average``.
    out : str or Path, optional
        File to save; the figure is returned in any case.
    window : dict, optional
        ``kmin``, ``kmax``, ``dk``, ``kweight`` of the transform (default 3, kmax - 1, 1, 2).
    title : str
    max_parts : int
        Parts drawn (the largest by their k-weighted norm).
    """
    plt = _plt()
    k, mean = result["k"], result["mean"]
    win = {"kmin": 3.0, "kmax": float(k[-1]) - 1.0, "dk": 1.0, "kweight": 2}
    win.update({kk: v for kk, v in (window or {}).items() if kk in win})
    kw = int(win["kweight"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    sel = (k >= win["kmin"] - win["dk"] / 2) & (k <= win["kmax"] + win["dk"] / 2)
    y = k ** kw * mean
    for key, alpha, label in (("spread", 0.12, "spread between frames"), ("stderr", 0.35, "standard error")):
        band = k ** kw * result[key]
        if np.all(np.isfinite(band)):
            ax[0].fill_between(k[sel], (y - band)[sel], (y + band)[sel], color=BLUE, alpha=alpha, lw=0, label=label)
    ax[0].plot(k[sel], y[sel], color=BLUE, lw=1.6, label="average")
    r = r_grid().numpy()
    ax[1].plot(r, _chir(k, mean, win["kmin"], win["kmax"], win["dk"], kw), color=BLUE, lw=1.6, label="average")
    parts = {n: v for n, v in result["parts"].items() if n not in ("single scattering", "multiple scattering")}
    order = sorted(parts, key=lambda n: -float(np.sqrt(((k ** kw * parts[n])[sel] ** 2).sum())))[:max_parts]
    for n, col in zip(order, PART_COLORS):
        ax[0].plot(k[sel], (k ** kw * parts[n])[sel], color=col, lw=0.9, label=n)
        ax[1].plot(r, _chir(k, parts[n], win["kmin"], win["kmax"], win["dk"], kw), color=col, lw=0.9, label=n)
    ax[0].set_xlabel("k (1/Å)"); ax[0].set_ylabel(f"k^{kw} chi(k)")
    ax[1].set_xlabel("R (Å)"); ax[1].set_ylabel("|chi(R)|"); ax[1].set_xlim(0, 7)
    for a in ax:
        a.legend(fontsize=7, frameon=False)
        style_axes(a)
    fig.suptitle(title or f"{result['edge']}: average of {result['n_frames']} frames", fontsize=9)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_convergence(spectrum=None, thermo=None, log=None, out=None, title=""):
    """Convergence: the R-factor of the running average against the final one (log scales), and the temperature and
    energies of the thermodynamic log against time.

    Parameters
    ----------
    spectrum : dict, optional
        ``{edge name: spectrum_convergence result}``.
    thermo : dict, optional
        ``thermo_summary`` result.
    log : dict, optional
        ``read_log`` result (the series to draw).
    out : str or Path, optional
    title : str
    """
    plt = _plt()
    panels = (1 if spectrum else 0) + (2 if log else 0)
    if panels == 0:
        raise ValueError("nothing to draw: give spectrum convergence or a log")
    fig, axes = plt.subplots(1, panels, figsize=(4.4 * panels, 3.4), squeeze=False)
    axes = list(axes[0])
    if spectrum:
        a = axes.pop(0)
        for (name, c), col in zip(spectrum.items(), [BLUE] + PART_COLORS):
            n, rf = np.array(c["n_frames"]), np.array(c["r_factor"])
            ok = rf > 0
            a.loglog(n[ok], rf[ok], "o-", color=col, ms=3, lw=1.2, label=f"{name} (split halves {c['split_half_r_factor']:.2g})")
        a.set_xlabel("frames averaged"); a.set_ylabel("R-factor to the final average"); a.legend(fontsize=7, frameon=False)
        style_axes(a)
    if log:
        t = log["time"]
        a = axes.pop(0)
        if log.get("temperature"):
            a.plot(t, log["columns"][log["temperature"]], color=ORANGE, lw=0.8)
            if thermo and "temperature" in thermo:
                a.axhline(thermo["temperature"]["mean"], color=INK, lw=0.8, ls="--")
        a.set_xlabel(f"time ({log['time_unit']})"); a.set_ylabel("temperature (K)")
        style_axes(a)
        a = axes.pop(0)
        present = [n for n in ("potential", "total", "conserved", "kinetic") if n in log["columns"]]
        for name, col in zip(present, [BLUE, INK, AQUA, NEUTRAL]):
            y = np.asarray(log["columns"][name])
            a.plot(t, y - y[0], color=col, lw=0.8, label=f"{name} - initial ({log['units'].get(name, '')})")
        a.set_xlabel(f"time ({log['time_unit']})"); a.set_ylabel("energy change"); a.legend(fontsize=7, frameon=False)
        style_axes(a)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_rdf(result, out=None, peaks=(), quantity="g", title=""):
    """Partial RDFs, one panel per pair, with the ranges of the analyzed peaks marked.

    Parameters
    ----------
    result : dict
        ``RDFAccumulator.result()``.
    out : str or Path, optional
    peaks : sequence of dict
        ``{"center", "neighbor", "r_min", "r_max"}`` to mark.
    quantity : {"g", "pairs_per_center_per_A"}
    title : str
    """
    from ..analysis.plots import plot_pair_distributions
    curves = {}
    for label, p in result["pairs"].items():
        y = p["g"] if (quantity == "g" and p["g"] is not None) else p["pairs_per_center_per_A"]
        curves[label] = y
    shells = {}
    for pk in peaks:
        shells.setdefault(f"{pk['center']}-{pk['neighbor']}", []).append((pk["r_min"], pk["r_max"]))
    ylabel = "g(r)" if quantity == "g" and all(p["g"] is not None for p in result["pairs"].values()) else "pairs per center per Å"
    return plot_pair_distributions(result["r"], curves, out=out, xlabel="distance (Å)", ylabel=ylabel, title=title, shells=shells,
                                   xlim=(0, result["r_max"]))


def plot_peaks(r, densities, analyses, out=None, title=""):
    """Each analyzed peak: the pair density (black) and the fitted forms.

    Parameters
    ----------
    r : array
        Bin centers (A).
    densities : list of array
        Pairs per center per A of each peak's pair.
    analyses : list of dict
        ``analyze_peak`` results with ``label``.
    """
    plt = _plt()
    n = max(1, len(analyses))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.3), squeeze=False)
    for a, dens, an in zip(axes[0], densities, analyses):
        lo, hi = an["r_min"], an["r_max"]
        pad = 0.25 * (hi - lo)
        m = (r >= lo - pad) & (r <= hi + pad)
        a.plot(r[m], dens[m], color=INK, lw=1.4, label="pair density")
        for form, col in (("gaussian", BLUE), ("skewed", ORANGE)):
            f = an.get(form)
            if f and "fitted" in f:
                a.plot(f["r_fit"], f["fitted"], color=col, lw=1.1, ls="--", label=f"{form} fit")
        a.axvline(lo, color=NEUTRAL, lw=0.8, ls=":"); a.axvline(hi, color=NEUTRAL, lw=0.8, ls=":")
        integ = an["integration"]
        a.set_title(f"{an.get('label', '')}: N {integ['N']:.2f}, R {integ['R']:.3f} Å, MSRD {integ['sigma2']:.4f} Å²", fontsize=8)
        a.set_xlabel("distance (Å)"); a.set_ylabel("pairs per center per Å"); a.legend(fontsize=7, frameon=False)
        style_axes(a)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_angles(results, out=None, title=""):
    """Bond-angle distributions (angles per center atom per configuration in each bin)."""
    plt = _plt()
    n = max(1, len(results))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False)
    for a, res in zip(axes[0], results):
        a.plot(res["angle"], res["per_center"], color=BLUE, lw=1.2)
        a.set_xlabel(f"{res['label']} angle (degrees)"); a.set_ylabel("angles per center"); a.set_xlim(0, 180)
        a.grid(color=GRID)
        style_axes(a)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
