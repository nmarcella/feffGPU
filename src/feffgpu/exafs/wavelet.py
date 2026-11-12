"""Wavelet residuals: optional objectives in place of the R-factor.

The default misfit of this package is xraylarch's R-factor on chi(R) (``feffgpu.exafs.transform``). Two other objectives
compare a calculated spectrum with the data on a wavelet transform, which resolves the signal in k and in R at once:

``Wavelet``
    EvAX's wavelet transform and residual (Timoshenko and Kuzmin, Comput. Phys. Commun. 180, 920 (2009); EvAX's
    ``Space w``). With ``x(z) = k^w chi(k) * window`` on the points of the k range,
    ``WT(r, b) = sqrt(a) sum_z x(z) exp(-0.1 a^2 (z - b)^2) exp(2 i r (z - b))`` with ``a = r / r0``; the residual is
    ``sum |WT_model - WT_data|^2 / sum |WT_data|^2`` over the R range and the k range.
``CauchyWavelet``
    xraylarch's Cauchy wavelet transform (``larch.xafs.cauchy_wavelet``, after Munoz, Argoul and Farges,
    Am. Mineral. 88, 694 (2003)), reproduced step by step in torch so that it is batched and differentiable; the
    residual is the same ratio over the rows of the R range and the columns of the k range.

Both objects have the interface that ``feffgpu.exafs.SpectrumComparison`` expects of its ``wavelet`` attribute:
``set_target(data)``, ``residual(model)`` (one value per spectrum), ``transform(model)`` and ``norm``.
"""
from __future__ import annotations
import numpy as np
import torch
from .transform import FitWindow, hanning_window

__all__ = ["Wavelet", "CauchyWavelet", "make_wavelet"]


class Wavelet:
    """EvAX's wavelet transform of chi(k) and the residual on it.

    Parameters
    ----------
    k : array, shape (NK,)
        The uniform k grid of the spectra (1/A).
    kw : array, shape (NK,)
        The weight applied to chi before the transform: ``k^kweight`` times the k window.
    kmin, kmax : float
        The k range of the transform (1/A): the points ``z`` and the centers ``b`` run over the grid points inside it.
    rmin, rmax : float
        The R range of the transform (A).
    nr : int, optional
        Number of points in R (default: as many as there are k points in the range).
    r0 : float, optional
        EvAX's ``R0_for_WT`` (A): the distance at which the wavelet has its reference width (default: the middle of
        the R range).
    device : str

    Attributes
    ----------
    r : torch.Tensor, shape (nR,)
        The R points (A).
    b : torch.Tensor, shape (nB,)
        The k points (1/A).
    norm : float
        ``sum |WT_data|^2`` after ``set_target``.
    """

    def __init__(self, k, kw, kmin, kmax, rmin, rmax, nr=None, r0=None, device="cuda"):
        k = torch.as_tensor(np.asarray(k, float), device=device, dtype=torch.float64)
        self.sel = (k >= kmin - 1e-9) & (k <= kmax + 1e-9)
        z = k[self.sel]
        nz = int(z.shape[0])
        r = torch.linspace(rmin, rmax, nr or nz, device=device, dtype=torch.float64)
        r0 = 0.5 * (rmin + rmax) if r0 is None else float(r0)
        a = r / r0                                                                      # (nR,)
        d = z[None, :] - z[:, None]                                                     # (nB, nZ): z - b
        g = torch.exp(-0.1 * a[:, None, None] ** 2 * d[None] ** 2) * torch.sqrt(a)[:, None, None]
        self.K = (g * torch.exp(2j * r[:, None, None] * d[None])).to(torch.complex64)    # (nR, nB, nZ)
        self.kw = torch.as_tensor(np.asarray(kw, float).reshape(-1), device=device, dtype=torch.float32)
        self.r, self.b, self.r0 = r.float(), z.float(), r0
        self.norm = 1.0

    @classmethod
    def for_window(cls, k, window: FitWindow, r0=None, device="cuda"):
        """The wavelet of a ``FitWindow``: weight ``k^kweight`` times the Hanning window, k range from ``kmin - dk/2``
        to ``kmax + dk/2`` (the window's support, which is EvAX's ``k_min`` to ``k_max``), R range ``rmin`` to ``rmax``."""
        kt = torch.as_tensor(np.asarray(k, float), dtype=torch.float32)
        win = hanning_window(kt, torch.tensor([window.kmin]), torch.tensor([window.kmax]), dk=window.dk)[0].numpy()
        kw = np.asarray(k, float) ** window.kweight * win
        return cls(k, kw, window.kmin - window.dk / 2, window.kmax + window.dk / 2, window.rmin, window.rmax, r0=r0, device=device)

    def transform(self, chi):
        """chi (S, NK) -> WT (S, nR, nB), complex."""
        x = (chi * self.kw[None])[:, self.sel].to(torch.complex64)
        return torch.einsum("sz,rbz->srb", x, self.K)

    def set_target(self, target):
        """Transform the data (NK,) and keep it as the reference of ``residual``."""
        self.target_wt = self.transform(torch.as_tensor(np.asarray(target, float), device=self.K.device, dtype=torch.float32)[None])[0]
        self.norm = float((self.target_wt.abs() ** 2).sum())

    def residual(self, chi):
        """(S,): ``sum |WT - WT_target|^2 / sum |WT_target|^2`` for every spectrum of ``chi`` (S, NK)."""
        return ((self.transform(chi) - self.target_wt[None]).abs() ** 2).sum((1, 2)) / self.norm


class CauchyWavelet:
    """xraylarch's Cauchy wavelet transform and the residual on it, batched and differentiable.

    The steps are those of ``larch.xafs.cauchy_wavelet``: ``chi k^kweight`` on the grid from k = 0, zero-padded to
    ``nfft / 2`` points, an FFT with ``2 nfft`` points, the product with the Cauchy wavelet of every R row, the inverse
    FFT, and the first ``len(k)`` points. No k window is applied (xraylarch applies none). The residual compares the
    rows with ``rmin <= R <= rmax`` and the columns with ``kmin <= k <= kmax``.

    Parameters
    ----------
    k : array, shape (NK,)
        The uniform k grid from zero (1/A).
    kweight : int
    kmin, kmax : float
        The k columns of the residual (1/A).
    rmin, rmax : float
        The R rows of the residual (A).
    nfft : int
    rmax_out : float
        Largest R of the transform (A), as in xraylarch.
    device : str
    """

    def __init__(self, k, kweight, kmin, kmax, rmin, rmax, nfft=2048, rmax_out=10.0, device="cuda"):
        k = np.asarray(k, float)
        self.device = torch.device(device)
        kstep = np.round(1000. * (k[1] - k[0])) / 1000.0
        rstep = (np.pi / 2048) / kstep
        nrpts = int(np.round((rmax_out - 1.e-7) / rstep))
        self.nkout, self.nfft, self.NFT = len(k), int(nfft), int(nfft / 2)
        freq = (1.0 / kstep) * np.arange(nfft) / (2 * nfft)
        omega = 2 * np.pi * freq
        r = np.linspace(0, rmax_out, nrpts)
        r[0] = 1.e-19
        a = nrpts / (2 * r)
        cauchy_sum = np.log(2 * np.pi) - np.log(1.0 + np.arange(nrpts)).sum()
        rows = np.nonzero((r >= rmin - 1e-9) & (r <= rmax + 1e-9))[0]
        G = np.zeros((len(rows), nfft), complex)
        for j, i in enumerate(rows):
            aom = a[i] * omega
            aom[np.where(aom == 0)] = 1.e-19
            G[j] = np.conj(np.exp(cauchy_sum + nrpts * np.log(aom) - aom))
        self.G = torch.as_tensor(G, device=self.device, dtype=torch.complex128)
        self.r = r[rows]
        self.kcols = torch.as_tensor(np.nonzero((k >= kmin - 1e-9) & (k <= kmax + 1e-9))[0], device=self.device)
        self.kweight = int(kweight)
        self.kpow = torch.as_tensor(k ** kweight if kweight != 0 else np.ones_like(k), device=self.device, dtype=torch.float64)
        self.norm = 1.0

    @classmethod
    def for_window(cls, k, window: FitWindow, device="cuda"):
        """The Cauchy wavelet over the k range ``kmin .. kmax`` and the R range of a ``FitWindow``."""
        return cls(k, window.kweight, window.kmin, window.kmax, window.rmin, window.rmax, device=device)

    def full(self, chi):
        """chi (S, NK) -> (S, nR, NK) complex128: the rows of the R range at every k."""
        x = torch.as_tensor(chi, device=self.device).to(torch.float64) * self.kpow[None]
        if self.nkout < self.NFT:
            x = torch.nn.functional.pad(x, (0, self.NFT - self.nkout))
        else:
            x = x[:, : self.NFT]
        tff = torch.fft.fft(x, n=2 * self.nfft)[:, : self.nfft]
        return torch.fft.ifft(self.G[None] * tff[:, None, :], n=2 * self.nfft)[..., : self.nkout]

    def transform(self, chi):
        """chi (S, NK) -> (S, nR, nK): the rows of the R range and the columns of the k range."""
        return self.full(chi)[..., self.kcols]

    def set_target(self, target):
        """Transform the data (NK,) and keep it as the reference of ``residual``."""
        with torch.no_grad():
            self.target_wt = self.transform(torch.as_tensor(np.asarray(target, float), device=self.device)[None])[0]
            self.norm = float((self.target_wt.abs() ** 2).sum())

    def residual(self, chi):
        """(S,): ``sum |W - W_target|^2 / sum |W_target|^2`` for every spectrum of ``chi`` (S, NK)."""
        return ((self.transform(chi) - self.target_wt[None]).abs() ** 2).sum((1, 2)) / self.norm


def make_wavelet(kind, k, window: FitWindow, r0=None, device="cuda"):
    """The wavelet objective named ``kind``: ``"wavelet"`` (EvAX's) or ``"cauchy_wavelet"`` (xraylarch's)."""
    if kind == "wavelet":
        return Wavelet.for_window(k, window, r0=r0, device=device)
    if kind == "cauchy_wavelet":
        return CauchyWavelet.for_window(k, window, device=device)
    raise ValueError(f"unknown wavelet objective {kind!r}: use 'wavelet' (EvAX's) or 'cauchy_wavelet' (xraylarch's)")
