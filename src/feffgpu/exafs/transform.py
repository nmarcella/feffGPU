"""The conventions that every tool in this package shares: the k grid, the k window, the Fourier transform to R space
and the R-factor. They are xraylarch's (and so Athena's and Artemis's):

- spectra live on a uniform k grid that starts at zero, step ``KSTEP`` (0.05 1/A);
- the k window is a Hanning window given by ``kmin``, ``kmax`` and ``dk``: the sills are centered on ``kmin`` and
  ``kmax`` and are ``dk`` wide, so the window is flat between ``kmin + dk/2`` and ``kmax - dk/2``;
- chi(R) is the FFT of ``k**kweight * chi(k) * window`` with xraylarch's ``xftf`` normalization, on
  ``R = RSTEP * arange(NFFT / 2)``;
- the misfit of a model to data is the R-factor on the R window ``rmin .. rmax``:
  ``sum |chi_R(model) - chi_R(data)|**2 / sum |chi_R(data)|**2`` over the real and imaginary parts.

``FitWindow`` carries the six numbers; the torch functions are batched and differentiable, the numpy function
``r_factor`` needs nothing but numpy.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import numpy as np
import torch
from ..compare import hanning, r_factor  # noqa: F401  (the numpy forms; re-exported)

ETOK = 0.2624684236075176          # eV -> 1/A^2 (2 m_e / hbar^2): k^2 = ETOK * (E - E0)
KSTEP = 0.05                       # 1/A
NFFT = 2048
RSTEP = math.pi / (KSTEP * NFFT)   # 0.03068 A


@dataclass(frozen=True)
class FitWindow:
    """The window of a comparison between a model and a spectrum, in xraylarch's terms.

    Attributes
    ----------
    kmin, kmax : float
        Centers of the two sills of the Hanning window (1/A).
    dk : float
        Width of each sill (1/A).
    kweight : int
        Power of k that multiplies chi(k) before the transform.
    rmin, rmax : float
        Range of R over which the R-factor is summed (A).
    """
    kmin: float = 3.0
    kmax: float = 14.0
    dk: float = 1.0
    kweight: int = 2
    rmin: float = 1.0
    rmax: float = 6.0

    def __post_init__(self):
        if not (self.kmax > self.kmin >= 0.0):
            raise ValueError(f"the k window needs 0 <= kmin < kmax (got kmin {self.kmin}, kmax {self.kmax})")
        if not (self.rmax > self.rmin >= 0.0):
            raise ValueError(f"the R window needs 0 <= rmin < rmax (got rmin {self.rmin}, rmax {self.rmax})")
        if self.dk < 0:
            raise ValueError(f"dk must not be negative (got {self.dk})")

    def as_dict(self):
        return asdict(self)

    @property
    def independent_points(self):
        """Stern's number of independent points, 2 (kmax - kmin) (rmax - rmin) / pi + 2 (xraylarch's ``nidp``)."""
        return 2.0 * (self.kmax - self.kmin) * (self.rmax - self.rmin) / math.pi + 2.0


def k_grid(kmax=24.0, kstep=KSTEP):
    """The uniform k grid ``0, kstep, ..., kmax`` (numpy)."""
    return kstep * np.arange(int(round(kmax / kstep)) + 1)


def hanning_window(k, kmin, kmax, dk=1.0):
    """Hanning window, batched and on the device of ``k``.

    Parameters
    ----------
    k : torch.Tensor, shape (NK,)
    kmin, kmax : torch.Tensor, shape (B,)
        Centers of the sills.
    dk : float
        Sill width.

    Returns
    -------
    torch.Tensor, shape (B, NK)
        1 between ``kmin + dk/2`` and ``kmax - dk/2``, 0 outside ``kmin - dk/2 .. kmax + dk/2``.
    """
    k = k[None, :]
    lo, hi = kmin[:, None], kmax[:, None]
    if dk <= 0:
        return ((k >= lo) & (k <= hi)).to(k.dtype)
    up = 0.5 * (1 - torch.cos(math.pi * ((k - (lo - dk / 2)) / dk).clamp(0, 1)))
    down = 0.5 * (1 + torch.cos(math.pi * ((k - (hi - dk / 2)) / dk).clamp(0, 1)))
    return up * down


def window_tensor(k, window: FitWindow):
    """The Hanning window of a ``FitWindow`` on the grid ``k`` (NK,) as a tensor (NK,)."""
    dev = k.device
    return hanning_window(k, torch.tensor([window.kmin], device=dev, dtype=k.dtype), torch.tensor([window.kmax], device=dev, dtype=k.dtype),
                          dk=window.dk)[0]


def ft(chi, k, window, kweight=2):
    """Complex chi(R) on ``R = RSTEP * arange(NFFT / 2)``.

    Parameters
    ----------
    chi : torch.Tensor, shape (B, NK)
    k : torch.Tensor, shape (NK,)
        Must be the uniform grid from zero with step ``KSTEP``.
    window : torch.Tensor, shape (NK,) or (B, NK)
    kweight : int

    Returns
    -------
    torch.Tensor, shape (B, NFFT / 2), complex
    """
    x = chi * k[None, :] ** kweight * window
    padded = torch.zeros(x.shape[0], NFFT, dtype=x.dtype, device=x.device)
    padded[:, : x.shape[1]] = x
    return (KSTEP / math.sqrt(math.pi)) * torch.fft.fft(padded)[:, : NFFT // 2]


def r_grid(device=None):
    """The R grid of ``ft`` (A), shape (NFFT / 2,)."""
    return RSTEP * torch.arange(NFFT // 2, device=device, dtype=torch.float32)


def r_factor_torch(model, data, k, window: FitWindow):
    """The R-factor of every row of ``model`` (B, NK) against ``data`` (NK,), differentiable in ``model``.

    Both spectra are on the uniform grid ``k`` (NK,). The transform is taken in double precision.

    Returns
    -------
    torch.Tensor, shape (B,)
    """
    kd = k.double()
    w = window_tensor(kd, window)
    r = r_grid(k.device)
    keep = (r >= window.rmin) & (r <= window.rmax)
    d = ft(data.double()[None], kd, w, kweight=window.kweight)[0][keep]
    m = ft(model.double(), kd, w, kweight=window.kweight)[:, keep]
    return ((m - d[None]).abs() ** 2).sum(1) / (d.abs() ** 2).sum()
