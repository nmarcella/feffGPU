"""Which distances make which features of a calculated spectrum: the spectrum of a structure split into the single
scattering of distance components (one element, one range of distances) and the multiple-scattering remainder, and
the share of each part in R bands of chi(R).

The parts interfere, so the moduli of their chi(R) do not add. The share of a part in a band is
``Re <part(R), total(R)> / <total(R), total(R)>`` summed over the band (real and imaginary parts): the shares of a
band add up to one, interference included, and a part with a negative share cancels signal there.

Used by RMC-EXAFS (``rmc-exafs signal-map``) and usable for any configurations (frames of a trajectory) with a
``feffgpu.exafs.edge.EdgeModel``.
"""
from __future__ import annotations
import numpy as np
import torch

__all__ = ["component_spectra", "band_shares", "plot_signal_map"]


def component_spectra(edge, positions, components):
    """The single-scattering chi(k) of distance components of configurations.

    Parameters
    ----------
    edge : feffgpu.exafs.edge.EdgeModel
    positions : array, shape (S, n, 3)
        Configurations (A); the parts are averaged over them.
    components : list of (str, float, float)
        ``(element, r_min, r_max)``: the pairs of the absorber with that element at ``r_min <= r < r_max`` (A).

    Returns
    -------
    parts : dict
        ``{"element r_min-r_max A": chi (NK,)}``, unscaled (S0^2 = 1, at the shift ``edge.e0``).
    info : dict
        ``{name: {"element", "r_min", "r_max", "N", "R", "g": pairs per absorber per A on the table grid, "r": grid}}``.
    """
    with torch.no_grad():
        hist = edge.static_histograms(positions).mean(0)                                     # (T, nr)
        grid = edge.r0 + edge.dr * torch.arange(edge.nr, device=hist.device)
        parts, info = {}, {}
        for el, lo, hi in components:
            types = [t for t, ty in enumerate(edge.types) if ty.element == el]
            if not types:
                raise ValueError(f"{edge.name}: no pairs with {el} in the model (elements {edge.elements})")
            mask = ((grid >= lo) & (grid < hi)).float()
            h = sum(hist[t] for t in types) * mask
            table = edge.pairs.tables[edge.ipot_of_symbol[el]]
            name = f"{el} {lo:g}-{hi:g} A"
            parts[name] = (h @ table / edge.nA).cpu().numpy()
            n = float(h.sum() / edge.nA)
            info[name] = {"element": el, "r_min": float(lo), "r_max": float(hi), "N": n,
                          "R": float((h * grid).sum() / h.sum()) if n > 1e-9 else float("nan"),
                          "g": (h / edge.nA / edge.dr).cpu().numpy(), "r": grid.cpu().numpy()}
    return parts, info


def band_shares(comparison, parts, total, bands):
    """The share of every part in R bands of chi(R) of ``total``.

    Parameters
    ----------
    comparison : feffgpu.exafs.SpectrumComparison
        Gives the window and the k weight of the transform (and S0^2 and the shift: the parts and the total are
        compared as the model is, scaled and shifted).
    parts : dict
        ``{name: chi (NK,)}``, unscaled calculated spectra.
    total : array, shape (NK,)
        The whole calculated spectrum, unscaled.
    bands : list of (float, float)
        R bands (A).

    Returns
    -------
    dict
        ``{name: {"lo-hi": share, ..., "peak_R": R of the largest |chi(R)| of the part}}``.
    """
    r = comparison.r

    def cr(chi):
        with torch.no_grad():
            m = comparison.model(torch.as_tensor(np.asarray(chi), device=comparison.device, dtype=torch.float32)[None])
            return comparison.chi_r(m)[0].cpu().numpy()
    ct = cr(total)
    out = {}
    for name, chi in parts.items():
        c = cr(chi)
        row = {}
        for lo, hi in bands:
            sel = (r >= lo) & (r <= hi)
            den = float(np.real(np.vdot(ct[sel], ct[sel])))
            row[f"{lo:g}-{hi:g}"] = float(np.real(np.vdot(c[sel], ct[sel])) / den) if den > 0 else float("nan")
        i = int(np.argmax(np.abs(c)))
        row["peak_R"] = float(r[i])
        out[name] = row
    return out


def plot_signal_map(comparison, parts, info, total, data=None, out=None, title="", rmax_plot=None):
    """Two rows: the pair distribution of every component (against the distance), and the magnitude of chi(R) of every
    part with the total (blue) and the data (black). R is the distance axis of the Fourier transform: a feature appears
    below the distance that makes it, by the scattering phase (about 0.3 to 0.5 A). Returns the matplotlib figure (saved
    to ``out`` if given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..exafs.plots import SERIES, INK, BLUE, NEUTRAL, style_axes
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    rmax_plot = rmax_plot or max(6.0, comparison.r_range[1] + 1)
    for j, (name, d) in enumerate(info.items()):
        axes[0].fill_between(d["r"], 0, d["g"], color=SERIES[j % len(SERIES)], alpha=0.6, lw=0, label=f"{name}: N {d['N']:.2f}")
    axes[0].set_ylabel("pairs per absorber per Å")
    axes[0].set_xlabel("distance (Å)")
    axes[0].set_xlim(0, rmax_plot)
    axes[0].legend(fontsize=7, frameon=False)
    r = comparison.r

    def cr(chi, scaled=True):
        with torch.no_grad():
            m = torch.as_tensor(np.asarray(chi), device=comparison.device, dtype=torch.float32)[None]
            if scaled:
                m = comparison.model(m)
            return np.abs(comparison.chi_r(m)[0].cpu().numpy())
    if data is not None:
        axes[1].plot(r, cr(data, scaled=False), color=INK, lw=1.8, label="data")
    axes[1].plot(r, cr(total), color=BLUE, lw=1.8, label="model")
    for j, (name, chi) in enumerate(parts.items()):
        axes[1].plot(r, cr(chi), color=SERIES[(j + 1) % len(SERIES)] if name != "multiple scattering" else NEUTRAL, lw=1.0,
                     ls=":" if name == "multiple scattering" else "-", label=name)
    axes[1].set_xlim(0, rmax_plot)
    axes[1].set_xlabel("R (Å)")
    axes[1].set_ylabel(f"{comparison.name}: |chi(R)|")
    axes[1].legend(fontsize=7, frameon=False)
    for ax in axes:
        style_axes(ax)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
