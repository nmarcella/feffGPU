"""Peaks of a radial distribution: plain integration and fits of a Gaussian or a skewed Gaussian.

The input is a pair density ``n(r)``: the number of neighbors per center atom per A on a grid of bin centers (the
``pairs_per_center_per_A`` of ``feffgpu.analysis.rdf``), whose integral over a range is a coordination number. In a
stated range ``r_min <= r < r_max``:

Integration (fit-free, preferred)
    N = the integral of n(r), R = the mean distance, sigma2 = the variance (MSRD), C3 = the third cumulant. Computed
    best from the distances themselves (``feffgpu.analysis.shell_table``, or the exact sums of ``RDFAccumulator``);
    ``integrate_peak`` gives the same from the histogram (with the binning's resolution).
Fits (as in ``md_exafs`` 2.0)
    ``"gaussian"``: ``n(r) = N0 exp(-(r - R)^2 / (2 s^2)) / (s sqrt(2 pi))``.
    ``"skewed"``: the skew-normal density ``n(r) = N0 (2 / w) phi((r - xi) / w) Phi(a (r - xi) / w)``, whose mean,
    variance and third cumulant are analytic.
    Both are fitted to n(r) itself, not to g(r): the moments of the fitted density are then the mean distance and the
    variance of the pairs (Sukhomlinov and Mueser 2016). The fitted N is the area of the fitted form inside the
    range. Standard errors come from the covariance of the least-squares fit (``scipy.optimize.curve_fit``, scaled by
    the residual scatter) propagated to N, R, sigma2 and C3 through the derivatives of those quantities. They say how
    well the form describes the histogram; they do not include the correlation between frames of a trajectory.
"""
from __future__ import annotations
import math
import numpy as np

__all__ = ["gaussian_density", "skewnormal_density", "skewnormal_moments", "integrate_peak", "fit_peak", "analyze_peak"]

SQ2PI = math.sqrt(2 * math.pi)


def gaussian_density(r, n0, mean, sd):
    """A Gaussian of area ``n0``, center ``mean`` (A) and standard deviation ``sd`` (A)."""
    sd = abs(sd) + 1e-12
    return n0 * np.exp(-0.5 * ((r - mean) / sd) ** 2) / (sd * SQ2PI)


def skewnormal_density(r, n0, xi, omega, alpha):
    """A skew-normal density of area ``n0``: location ``xi`` (A), scale ``omega`` (A), shape ``alpha``."""
    from scipy.special import erf
    omega = abs(omega) + 1e-12
    z = (r - xi) / omega
    return n0 * (2.0 / omega) * np.exp(-0.5 * z * z) / SQ2PI * 0.5 * (1.0 + erf(alpha * z / math.sqrt(2.0)))


def skewnormal_moments(xi, omega, alpha):
    """Mean (A), variance (A^2) and third cumulant (A^3) of a skew-normal distribution."""
    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    b = delta * math.sqrt(2.0 / math.pi)
    mean = xi + omega * b
    var = omega * omega * (1.0 - b * b)
    c3 = (4.0 - math.pi) / 2.0 * (omega * b) ** 3
    return mean, var, c3


def integrate_peak(r, density, r_min, r_max, bin_width=None):
    """Shell numbers by integration of a binned pair density over ``r_min <= r < r_max`` (bin centers).

    Returns
    -------
    dict
        ``N``, ``R`` (A), ``sigma2`` (A^2), ``c3`` (A^3); the variance includes the binning's ``bin_width^2 / 12``
        (prefer the exact numbers from the distances when they are available).
    """
    r = np.asarray(r, float)
    n = np.asarray(density, float)
    dr = float(bin_width) if bin_width else float(r[1] - r[0])
    m = (r >= r_min) & (r < r_max)
    w = n[m] * dr
    N = float(w.sum())
    if N <= 0:
        return {"N": 0.0, "R": math.nan, "sigma2": math.nan, "c3": math.nan}
    R = float((r[m] * w).sum() / N)
    var = float(((r[m] - R) ** 2 * w).sum() / N)
    c3 = float(((r[m] - R) ** 3 * w).sum() / N)
    return {"N": N, "R": R, "sigma2": var, "c3": c3}


def _derived(form, p, r_min, r_max):
    """(N inside the range, mean, variance, third cumulant) of fitted parameters."""
    from scipy.stats import norm, skewnorm
    if form == "gaussian":
        n0, mean, sd = p
        sd = abs(sd)
        inside = norm.cdf(r_max, mean, sd) - norm.cdf(r_min, mean, sd)
        return np.array([n0 * inside, mean, sd * sd, 0.0])
    n0, xi, omega, alpha = p
    omega = abs(omega)
    inside = skewnorm.cdf(r_max, alpha, xi, omega) - skewnorm.cdf(r_min, alpha, xi, omega)
    mean, var, c3 = skewnormal_moments(xi, omega, alpha)
    return np.array([n0 * inside, mean, var, c3])


def fit_peak(r, density, r_min, r_max, form="gaussian"):
    """Fit one form to a peak of a pair density.

    Parameters
    ----------
    r : array, shape (nb,)
        Bin centers (A).
    density : array, shape (nb,)
        Pairs per center per A.
    r_min, r_max : float
        The range of the peak (A).
    form : {"gaussian", "skewed"}

    Returns
    -------
    dict
        ``form``; ``N``, ``R``, ``sigma2``, ``c3`` (N is the fitted form's area inside the range; R, sigma2 and C3 the
        moments of the fitted form) and ``N_se``, ``R_se``, ``sigma2_se``, ``c3_se`` (standard errors from the fit
        covariance); ``parameters`` and ``parameters_se``; ``r_squared``; ``r_fit`` and ``fitted`` (the fitted density
        on the range's bins); ``method`` (one sentence). On failure: ``form`` and ``error``.
    """
    from scipy.optimize import curve_fit
    r = np.asarray(r, float)
    n = np.asarray(density, float)
    m = (r >= r_min) & (r < r_max)
    x, y = r[m], n[m]
    npar = 3 if form == "gaussian" else 4
    if len(x) < npar + 2 or not np.any(y > 0):
        return {"form": form, "error": "too few non-empty bins in the range for a fit"}
    dr = float(r[1] - r[0])
    area = float(y.sum() * dr)
    mean0 = float((x * y).sum() / y.sum())
    sd0 = max(float(np.sqrt(((x - mean0) ** 2 * y).sum() / y.sum())), dr)
    width = r_max - r_min
    try:
        if form == "gaussian":
            f = gaussian_density
            p0 = [area, mean0, sd0]
            bounds = ([0.0, r_min, 1e-4], [10.0 * area + 1e-9, r_max, 2.0 * width])
        elif form == "skewed":
            f = skewnormal_density
            p0 = [area, mean0 - 0.5 * sd0, 1.2 * sd0, 0.5]
            p0[1] = min(max(p0[1], r_min), r_max)
            bounds = ([0.0, r_min - width, 1e-4, -15.0], [10.0 * area + 1e-9, r_max + width, 2.0 * width, 15.0])
        else:
            raise ValueError(f"unknown form {form!r} (\"gaussian\" or \"skewed\")")
        popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as err:
        return {"form": form, "error": f"the fit did not converge: {err}"}
    fitted = f(x, *popt)
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    q = _derived(form, popt, r_min, r_max)
    J = np.zeros((4, npar))                                                       # derivatives of the derived quantities
    for j in range(npar):
        h = 1e-6 * max(abs(popt[j]), 1e-3)
        up, dn = popt.copy(), popt.copy()
        up[j] += h
        dn[j] -= h
        J[:, j] = (_derived(form, up, r_min, r_max) - _derived(form, dn, r_min, r_max)) / (2 * h)
    if np.all(np.isfinite(pcov)):
        cov = J @ pcov @ J.T
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        pse = np.sqrt(np.clip(np.diag(pcov), 0, None))
    else:
        se = np.full(4, math.nan)
        pse = np.full(npar, math.nan)
    names = ["area", "mean", "sd"] if form == "gaussian" else ["area", "location", "scale", "shape"]
    return {"form": form, "N": float(q[0]), "R": float(q[1]), "sigma2": float(q[2]), "c3": float(q[3]),
            "N_se": float(se[0]), "R_se": float(se[1]), "sigma2_se": float(se[2]), "c3_se": float(se[3]),
            "parameters": dict(zip(names, map(float, popt))), "parameters_se": dict(zip(names, map(float, pse))),
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0, "r_fit": x, "fitted": fitted,
            "method": "least-squares fit to the pair density in the range; standard errors from the fit covariance scaled by the residual "
                      "scatter, propagated through the derivatives of N, R, sigma2 and C3; they do not include the correlation between "
                      "frames"}


def analyze_peak(r, density, r_min, r_max, fit="both", exact=None):
    """Integration numbers and fits of one peak, side by side.

    Parameters
    ----------
    r, density : arrays
        Bin centers (A) and pairs per center per A.
    r_min, r_max : float
    fit : {"gaussian", "skewed", "both", "none"}
    exact : dict, optional
        Shell numbers from the distances themselves (``shell_table`` or ``RDFAccumulator``); used as the integration
        numbers when given, else ``integrate_peak`` of the histogram.

    Returns
    -------
    dict
        ``r_min``, ``r_max``, ``integration`` (with ``source``: ``"distances"`` or ``"histogram"``), and ``gaussian``
        and/or ``skewed`` (from ``fit_peak``).
    """
    out = {"r_min": float(r_min), "r_max": float(r_max)}
    if exact is not None:
        integ = {k: exact[k] for k in ("N", "R", "sigma2", "c3") if k in exact}
        integ["source"] = "distances"
    else:
        integ = integrate_peak(r, density, r_min, r_max)
        integ["source"] = "histogram"
    out["integration"] = integ
    forms = {"both": ("gaussian", "skewed"), "gaussian": ("gaussian",), "skewed": ("skewed",), "none": ()}[fit]
    for form in forms:
        out[form] = fit_peak(r, density, r_min, r_max, form)
    return out
