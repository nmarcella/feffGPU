"""Comparing two chi(k): the R-space R-factor with a Hanning-tapered k window (no external dependency)."""
import math
import numpy as np


def hanning(k, kmin, kmax, dk=1.0):
    """The Hanning window with sills centered on ``kmin`` and ``kmax`` and ``dk`` wide; ``dk = 0`` gives the rectangular
    window from ``kmin`` to ``kmax`` (as ``feffgpu.exafs.hanning_window`` does)."""
    k = np.asarray(k, float)
    if dk <= 0:
        return ((k >= kmin) & (k <= kmax)).astype(float)
    up = 0.5 * (1 - np.cos(np.pi * np.clip((k - (kmin - dk / 2)) / dk, 0, 1)))
    down = 0.5 * (1 + np.cos(np.pi * np.clip((k - (kmax - dk / 2)) / dk, 0, 1)))
    return up * down


def r_factor(k, data, fit, kmin=3.0, kmax=23.0, rmin=1.0, rmax=6.0, kweight=2, nfft=2048, dk=1.0):
    """sum |FT(fit) - FT(data)|^2 / sum |FT(data)|^2 over rmin..rmax; k uniform, FT of k^kweight chi with a Hanning window
    whose sills are centered on kmin and kmax and are dk wide (xraylarch's convention; default 1 A^-1)."""
    k, data, fit = (np.asarray(v, float) for v in (k, data, fit))
    step = k[1] - k[0]
    win = hanning(k, kmin, kmax, dk)
    f = lambda c: np.fft.fft(c * k ** kweight * win, n=nfft)[: nfft // 2] * step / math.sqrt(math.pi)
    r = np.arange(nfft // 2) * math.pi / (nfft * step)
    keep = (r >= rmin) & (r <= rmax)
    d, m = f(data)[keep], f(fit)[keep]
    return float(np.sum(np.abs(m - d) ** 2) / np.sum(np.abs(d) ** 2))
