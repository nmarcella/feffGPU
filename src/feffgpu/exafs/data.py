"""Measured spectra: reading chi(k) files and putting them on the model's k grid.

The model lives on the uniform grid of ``feffgpu.exafs.k_grid`` (from zero, step 0.05 1/A). A measured chi(k) is
read as two columns (k in 1/A, chi), interpolated onto that grid, and set to zero where the data do not reach; the
k window of the comparison must lie inside the measured range.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .transform import ETOK


def read_chi(path, k_column=0, chi_column=1):
    """Read a measured chi(k) from a text file.

    Lines that start with ``#``, ``%``, ``;`` or ``!`` and lines that do not parse as numbers are skipped, so the
    ``.chi`` files written by Athena and by xraylarch are read as they are.

    Parameters
    ----------
    path : str or Path
    k_column, chi_column : int
        Zero-based columns of k (1/A) and chi.

    Returns
    -------
    k, chi : numpy.ndarray
        Sorted by k.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        t = line.replace(",", " ").split()
        if not t or t[0][0] in "#%;!":
            continue
        try:
            rows.append((float(t[k_column]), float(t[chi_column])))
        except (ValueError, IndexError):
            continue
    if len(rows) < 2:
        raise ValueError(f"{path}: no two-column chi(k) data found")
    a = np.array(rows, float)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1]


def to_grid(k, k_data, chi_data):
    """A measured spectrum on the model's grid ``k``: linear interpolation, zero outside the measured range."""
    k = np.asarray(k, float)
    out = np.zeros_like(k)
    inside = (k >= k_data[0]) & (k <= k_data[-1])
    out[inside] = np.interp(k[inside], k_data, chi_data)
    return out


def check_window_inside_data(k_data, kmin, kmax, dk=0.0, name="the spectrum"):
    """Raise ``ValueError`` if the k window (with its sills) reaches outside the measured range."""
    lo, hi = kmin - dk / 2, kmax + dk / 2
    if lo < k_data[0] - 1e-6 or hi > k_data[-1] + 1e-6:
        raise ValueError(f"{name}: the k window {lo:.2f} to {hi:.2f} 1/A (sills included) reaches outside the measured range "
                         f"{k_data[0]:.2f} to {k_data[-1]:.2f} 1/A")


def shifted_target(k, kd, cd, s02=1.0, de0=0.0):
    """The measured chi ``(kd, cd)`` divided by S0^2 and read at ``k' = sqrt(k^2 - dE0 ETOK)`` on the grid ``k``
    (EvAX's convention, in which the data carry the amplitude and the energy shift); zero where the data do not
    reach. The fits of this package keep S0^2 and the energy shift on the model instead (``feffgpu.exafs.EdgeModel``);
    this function is for inputs prepared the EvAX way."""
    target = np.zeros_like(k)
    e = k ** 2 - de0 * ETOK
    ks = np.sign(e) * np.sqrt(np.abs(e))
    inside = (ks >= kd[0]) & (ks <= kd[-1])
    target[inside] = np.interp(ks[inside], kd, cd) / s02
    return target


def target_on_grid(k, kd, cd):
    """A spectrum that is already on a 0.05 1/A grid (for example EvAX's ``ipolEXAFS.dat``) placed on ``k``; zero
    elsewhere."""
    step = k[1] - k[0]
    idx = np.round(kd / step).astype(int)
    if not np.allclose(idx * step, kd, atol=1e-6):
        raise ValueError("the spectrum must be on the same k grid as the model (use to_grid to interpolate)")
    target = np.zeros_like(k)
    ok = (idx >= 0) & (idx < len(k))
    target[idx[ok]] = cd[ok]
    return target


def evax_window(k, kmin, kmax, ftw):
    """EvAX's k window: 1 strictly inside ``(kmin + ftw, kmax - ftw)``, cos^2 tapers to the ends, 0 outside. In
    xraylarch's terms this is the Hanning window with ``kmin + ftw/2``, ``kmax - ftw/2`` and ``dk = ftw``."""
    w = np.zeros_like(k)
    inside = (k >= kmin - 1e-9) & (k <= kmax + 1e-9)
    w[inside] = 1.0
    lo = inside & (k <= kmin + ftw)
    w[lo] = np.cos(((k[lo] - kmin) / ftw - 1.0) * np.pi / 2) ** 2
    hi = inside & (k >= kmax - ftw)
    w[hi] = np.cos(((kmax - k[hi]) / ftw - 1.0) * np.pi / 2) ** 2
    return w
