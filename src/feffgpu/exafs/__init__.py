"""EXAFS conventions and the forward model of atomic configurations, shared by MD-EXAFS and RMC-EXAFS.

- ``feffgpu.exafs.transform``: the k grid, the Hanning window, the Fourier transform, the R-factor (xraylarch's
  conventions) and ``FitWindow``.
- ``feffgpu.exafs.data``: reading measured chi(k) and placing it on the model's grid.
- ``feffgpu.exafs.forward``: ``PairTables`` (exact single scattering of whole configurations), ``MultipleScattering``
  and ``PathSpectra`` (FEFF's path finder and path stage on lists of absorbers).
- ``feffgpu.exafs.comparison``: ``SpectrumComparison`` (a measured spectrum, its window, S0^2 and the E0 shift on the
  model, the R-factor).
- ``feffgpu.exafs.amplitude``: ``fit_shared`` and ``shared_intervals`` (S0^2 and the E0 shift, one value per edge,
  shared between samples, with intervals).
- ``feffgpu.exafs.plots``: ``plot_comparison``.
"""
from .transform import ETOK, KSTEP, NFFT, RSTEP, FitWindow, k_grid, hanning_window, window_tensor, ft, r_grid, r_factor, r_factor_torch
from .data import read_chi, to_grid, check_window_inside_data, shifted_target, target_on_grid, evax_window
from .forward import PathSpectra, PairTables, MultipleScattering
from .comparison import SpectrumComparison
from .amplitude import fit_shared, shared_intervals, likelihood_weights
from .plots import plot_comparison

__all__ = ["ETOK", "KSTEP", "NFFT", "RSTEP", "FitWindow", "k_grid", "hanning_window", "window_tensor", "ft", "r_grid", "r_factor", "r_factor_torch",
           "read_chi", "to_grid", "check_window_inside_data", "shifted_target", "target_on_grid", "evax_window",
           "PathSpectra", "PairTables", "MultipleScattering", "SpectrumComparison", "fit_shared", "shared_intervals", "likelihood_weights",
           "plot_comparison"]
