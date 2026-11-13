"""Comparing a calculated spectrum with a measured one: the window, S0^2 and the E0 shift on the model, the R-factor.

``SpectrumComparison`` holds one measured chi(k) on the model's k grid and the window of the comparison. The
calculated spectrum is FEFF's (S0^2 = 1, no Debye-Waller factor). Following xraylarch's ``feffit``, the amplitude
factor and the energy shift act on the model, never on the data::

    model(k) = S0^2 * chi_calc(k~),    k~ = sqrt(k^2 - ETOK * delta)

where ``delta`` is an extra energy shift (eV) on top of the shift ``e0`` at which the spectrum was calculated; the
edge's total shift is ``dE0 = e0 + delta``. The shift is applied by cubic interpolation on the uniform k grid, which
is differentiable, so S0^2 and the shift can be fitted (``feffgpu.exafs.amplitude``).

The misfit is the R-factor on the R window: ``sum |chi_R(model) - chi_R(data)|^2 / sum |chi_R(data)|^2``.

MD-EXAFS uses this class directly; the RMC-EXAFS edge model builds on it.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as Fnn
from .transform import ETOK, KSTEP, FitWindow, ft, hanning_window, r_grid


class SpectrumComparison:
    """One measured spectrum, its fit window, and S0^2 and the E0 shift on the model.

    Parameters
    ----------
    name : str
        Label of the spectrum (usually the absorbing element or the edge). Spectra with the same name share S0^2
        and the E0 shift in a joint fit.
    k : array, shape (NK,)
        The model's uniform k grid from zero, step 0.05 1/A (``feffgpu.exafs.k_grid``).
    data : array, shape (NK,)
        The measured chi(k) on that grid (``feffgpu.exafs.to_grid``), as measured: never divided by S0^2.
    window : FitWindow
        The k window, the k weight and the R window.
    s02 : float
        Amplitude factor: the start value if ``fit_s02``, else the held value.
    e0 : float
        Energy shift (eV) at which the calculated spectra that will be compared were evaluated.
    fit_s02, fit_e0 : bool
        Whether a fit may change S0^2 and the shift.
    device : str

    Attributes
    ----------
    s02 : float
        Current amplitude factor.
    delta_e0 : float
        Current extra shift (eV); ``de0`` is ``e0 + delta_e0``.
    n_ind : float
        Number of independent points of the window (xraylarch's ``nidp``).
    """

    def __init__(self, name, k, data, window: FitWindow, s02=1.0, e0=0.0, fit_s02=False, fit_e0=False, device="cuda"):
        self.name, self.device = str(name), torch.device(device)
        self.k = np.asarray(k, float)
        self.NK = len(self.k)
        if abs((self.k[1] - self.k[0]) - KSTEP) > 1e-9 or abs(self.k[0]) > 1e-12:
            raise ValueError(f"the k grid must start at zero with a step of {KSTEP} 1/A (feffgpu.exafs.k_grid)")
        self.data = np.asarray(data, float)
        if self.data.shape != self.k.shape:
            raise ValueError("data must be on the k grid (use feffgpu.exafs.to_grid)")
        self.fit_window = window
        self.window, self.r_range, self.kweight, self.dk = (window.kmin, window.kmax), (window.rmin, window.rmax), int(window.kweight), float(window.dk)
        self.s02 = self.s02_init = float(s02)
        self.e0 = float(e0)
        self.delta_e0 = 0.0
        self.fit_s02, self.fit_e0 = bool(fit_s02), bool(fit_e0)
        self.kt = torch.tensor(self.k, device=self.device, dtype=torch.float32)
        self.win = hanning_window(self.kt, torch.tensor([window.kmin], device=self.device), torch.tensor([window.kmax], device=self.device), dk=self.dk)[0]
        self.kw = self.kt ** self.kweight * self.win
        r = r_grid(self.device)
        self.rmask = (r >= window.rmin) & (r <= window.rmax)
        self.wavelet = None                                                  # an optional other objective: an object with residual(model) and transform(model), norm
        self.n_ind = window.independent_points
        self.set_data(self.data)

    def set_data(self, data):
        """Replace the measured spectrum (NK,) and recompute its transform and norm."""
        self.data = np.asarray(data, float)
        with torch.no_grad():
            d = torch.as_tensor(self.data, device=self.device, dtype=torch.float64)[None]
            self.data_r = ft(d, self.kt.double(), self.win.double(), kweight=self.kweight)[0, self.rmask]
            self.data_norm = (self.data_r.abs() ** 2).sum()
        if float(self.data_norm) <= 0.0:
            self.data_norm = torch.ones((), device=self.device, dtype=torch.float64)       # an empty target: residuals are absolute
        if self.wavelet is not None:
            self.wavelet.set_target(self.data)

    # -- S0^2 and the energy shift ------------------------------------------------------------------------------
    @property
    def target(self):
        """The data divided by the current S0^2: what an unscaled model is compared with."""
        return self.data / self.s02

    def set_s02(self, value):
        self.s02 = float(value)

    def set_de0(self, delta):
        self.delta_e0 = float(delta)

    @property
    def de0(self):
        """The total energy shift (eV): ``e0`` plus the fitted extra shift."""
        return self.e0 + self.delta_e0

    def shift_matrix(self, delta):
        """The matrix M(delta), (NK, NK), with ``chi @ M.T`` the spectrum at ``k~ = sqrt(k^2 - ETOK delta)``.

        Catmull-Rom cubic interpolation on the uniform grid, zero where k~ is not real. It is built from constant
        one-hot matrices and polynomial weights, so it is twice differentiable in ``delta``."""
        kt = self.kt.to(torch.float32)
        q = kt ** 2 - ETOK * delta
        ok = (q > 0).to(torch.float32)
        u = torch.sqrt(torch.clamp(q, min=1e-12)) / KSTEP
        i = torch.floor(u.detach()).long().clamp(0, self.NK - 2)
        t = u - i.to(u.dtype)                                                # not clamped: at delta 0 every k sits on a grid point and a clamp would remove the gradient
        t2, t3 = t * t, t * t * t
        ws = (-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2)
        M = 0
        for m, w in enumerate(ws):
            E = Fnn.one_hot((i + m - 1).clamp(0, self.NK - 1), self.NK).to(torch.float32)   # (NK_out, NK_in), constant
            M = M + (w * ok)[:, None] * E
        return M

    def shift(self, chi, delta=None):
        """``chi`` (S, NK) read at ``k~ = sqrt(k^2 - ETOK delta)``: the model with an extra energy shift ``delta`` (eV;
        a tensor for a differentiable fit; default the current extra shift)."""
        delta = self.delta_e0 if delta is None else delta
        if not torch.is_tensor(delta) and abs(float(delta)) < 1e-12:
            return chi
        M = self.shift_matrix(delta)
        return chi @ M.t().to(chi.dtype)

    def model(self, chi, s02=None, de0=None):
        """The calculated spectra ``chi`` (S, NK) as they are compared with the data: ``s02 * chi`` at the shifted k."""
        s02 = self.s02 if s02 is None else s02
        return s02 * self.shift(chi, de0)

    # -- the misfit ---------------------------------------------------------------------------------------------
    def residual(self, chi, s02=None, de0=None):
        """The R-factor of every calculated spectrum ``chi`` (S, NK) -> (S,).

        ``s02`` and ``de0`` (the extra shift) may be tensors for a differentiable fit; by default the current values
        are used."""
        m = self.model(chi, s02, de0)
        if self.wavelet is not None:
            return self.wavelet.residual(m).double()            # double, as the R-factor below (the misfit sums are double)
        cr = ft(m.to(torch.float64), self.kt.double(), self.win.double(), kweight=self.kweight)[:, self.rmask]
        return ((cr - self.data_r[None]).abs() ** 2).sum(1) / self.data_norm

    def r_factor(self, chi):
        """The R-factor of one calculated spectrum (NK,) with the current S0^2 and shift, as a float."""
        chi = torch.as_tensor(np.asarray(chi) if not torch.is_tensor(chi) else chi, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            return float(self.residual(chi[None])[0])

    @torch.no_grad()
    def distance(self, chi_a, chi_b):
        """The same ratio between two calculated spectra (NK,) each, relative to the data's norm."""
        d = self.model((chi_a - chi_b)[None])                                # the shift and the scale are linear
        if self.wavelet is not None:
            return float((self.wavelet.transform(d).abs() ** 2).sum() / self.wavelet.norm)
        cd = ft(d.to(torch.float64), self.kt.double(), self.win.double(), kweight=self.kweight)[:, self.rmask]
        return float((cd.abs() ** 2).sum() / self.data_norm)

    def chi_r(self, chi):
        """chi(R) (S, NR) complex of spectra ``chi`` (S, NK) or (NK,), with this comparison's k weight and window, on
        the grid ``self.r``."""
        chi = torch.as_tensor(chi, device=self.device, dtype=torch.float32)
        if chi.dim() == 1:
            chi = chi[None]
        return ft(chi.to(torch.float64), self.kt.double(), self.win.double(), kweight=self.kweight)

    @property
    def r(self):
        """The R grid of ``chi_r`` (A), numpy."""
        return r_grid().numpy()

    def describe(self):
        """One line: the window and the current S0^2 and shift."""
        return (f"{self.name}: k {self.window[0]}-{self.window[1]} (dk {self.dk}, k^{self.kweight}), R {self.r_range[0]}-{self.r_range[1]}; "
                f"S0^2 {self.s02:.4g}{' (fitted)' if self.fit_s02 else ' (held)'}, dE0 {self.de0:.3f} eV{' (fitted)' if self.fit_e0 else ' (held)'}; "
                f"{self.n_ind:.0f} independent points")
