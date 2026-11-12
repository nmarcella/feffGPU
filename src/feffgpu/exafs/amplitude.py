"""Fitting S0^2 and the E0 shift of calculated spectra to measured ones, shared between spectra of the same edge.

The calculated spectra are fixed (they come from atom positions); the only free numbers are the amplitude factor
S0^2 and the energy shift of each edge. Spectra with the same ``name`` (the same absorption edge of the same
compound, measured on several samples, temperatures or trajectories) share one S0^2 and one shift.

The fit minimizes::

    sum over groups g of  sample_weight_g * sum over spectra s in g of  weight_s * R_s(S0^2, delta)

with R_s the R-factor of spectrum s (``SpectrumComparison.residual``). A group is one sample (its edges).

Uncertainty. The measurement noise is not known, so the misfit at the optimum is taken as its own noise level: the
log-likelihood of group g is ``W_g * (its weighted R-factor sum)`` with ``W_g = N_ind,g / (2 M_g)``, where N_ind is the
number of independent points of the group's windows and M_g a misfit floor (by default the group's misfit at the
optimum). The standard deviations are the square roots of the diagonal of the inverse Hessian of the summed
log-likelihood at the optimum, and the reported intervals are ``+- z`` standard deviations (z = 1.645: 90 %), on
log S0^2 and on the shift. This states how well the two numbers are determined by the spectra for the given
structure; it does not include the uncertainty of the structure itself.
"""
from __future__ import annotations
import math
import torch

__all__ = ["shared_names", "fit_shared", "shared_intervals", "likelihood_weights"]


def shared_names(groups):
    """The names with a free S0^2 and the names with a free shift, in order of first appearance."""
    s02_names, e0_names = [], []
    for group in groups:
        for cmp, _chi, _w in group:
            if cmp.fit_s02 and cmp.name not in s02_names:
                s02_names.append(cmp.name)
            if cmp.fit_e0 and cmp.name not in e0_names:
                e0_names.append(cmp.name)
    return s02_names, e0_names


def _first(groups, name):
    for group in groups:
        for cmp, _chi, _w in group:
            if cmp.name == name:
                return cmp
    raise KeyError(name)


def _check(groups, e0_names):
    for name in e0_names:                                                    # a shared extra shift means the same dE0 only on the same base shift
        e0s = {round(float(cmp.e0), 6) for group in groups for cmp, _c, _w in group if cmp.name == name}
        if len(e0s) > 1:
            raise ValueError(f"{name}: the spectra were calculated at different energy shifts {sorted(e0s)}; give the same e0 for all of them")


def _params(groups, s02_names, e0_names, device):
    q = [math.log(_first(groups, n).s02) for n in s02_names] + [_first(groups, n).delta_e0 for n in e0_names]
    return torch.tensor(q, device=device, dtype=torch.float32)


def _terms(groups, p, s02_names, e0_names):
    """Per group: the weighted sum of its R-factors and the R-factors themselves, as functions of the parameters."""
    ns = len(s02_names)
    out = []
    for group in groups:
        ratios, weights = [], []
        for cmp, chi, w in group:
            s02 = torch.exp(p[s02_names.index(cmp.name)]) if cmp.fit_s02 else None
            de0 = p[ns + e0_names.index(cmp.name)] if cmp.fit_e0 else None
            c = chi if chi.dim() == 2 else chi[None]
            ratios.append(cmp.residual(c, s02, de0)[0])
            weights.append(float(w))
        ratios = torch.stack(ratios)
        out.append(((ratios * torch.as_tensor(weights, device=ratios.device)).sum(), ratios))
    return out


def _write_back(groups, p, s02_names, e0_names):
    ns = len(s02_names)
    q = p.detach()
    for group in groups:
        for cmp, _chi, _w in group:
            if cmp.fit_s02:
                cmp.set_s02(float(torch.exp(q[s02_names.index(cmp.name)])))
            if cmp.fit_e0:
                cmp.set_de0(float(q[ns + e0_names.index(cmp.name)]))


def fit_shared(groups, sample_weights=None, s02_bounds=(0.2, 5.0), de0_bounds=(-15.0, 15.0), iters=60):
    """Fit S0^2 and the extra E0 shift, one value per name, and write them into the comparisons.

    Parameters
    ----------
    groups : list of list of (SpectrumComparison, torch.Tensor, float)
        One list per sample; each entry is a comparison, the calculated spectrum (NK,) (unscaled, at the comparison's
        base shift ``e0``) and the weight of that spectrum inside its sample. Only comparisons with ``fit_s02`` /
        ``fit_e0`` set are changed; the others keep their values.
    sample_weights : list of float, optional
        Weight of each group in the objective (default: equal weights that sum to 1).
    s02_bounds : (float, float)
        Allowed range of S0^2. A safety range, not a prior: a result on a bound is not a fit.
    de0_bounds : (float, float)
        Allowed range of the extra shift (eV), relative to each comparison's ``e0``.
    iters : int
        L-BFGS iterations.

    Returns
    -------
    dict
        ``residual`` (the objective), ``ratios`` (per group, the R-factor of each spectrum), ``s02`` and ``de0``
        (``{name: value}`` of the fitted quantities; ``de0`` is the total shift ``e0 + delta``), ``s02_names``,
        ``de0_names``, ``at_bound`` (names whose fitted value sits on a bound).
    """
    s02_names, e0_names = shared_names(groups)
    _check(groups, e0_names)
    device = groups[0][0][0].device
    sw = [1.0 / len(groups)] * len(groups) if sample_weights is None else [float(v) for v in sample_weights]
    ns = len(s02_names)
    lo_s, hi_s = math.log(s02_bounds[0]), math.log(s02_bounds[1])
    p = _params(groups, s02_names, e0_names, device).requires_grad_(True)

    def clamp_(q):
        if ns:
            q[:ns].clamp_(lo_s, hi_s)
        if len(e0_names):
            q[ns:].clamp_(float(de0_bounds[0]), float(de0_bounds[1]))

    def objective(q):
        terms = _terms(groups, q, s02_names, e0_names)
        total = 0.0
        for w, (f, _r) in zip(sw, terms):
            total = total + w * f
        return total, [r for _f, r in terms]

    if p.numel():
        opt = torch.optim.LBFGS([p], lr=0.5, max_iter=int(iters), line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            with torch.no_grad():
                clamp_(p)
            f, _ = objective(p)
            f.backward()
            return f
        opt.step(closure)
    with torch.no_grad():
        clamp_(p)
        f, ratios = objective(p)
    _write_back(groups, p, s02_names, e0_names)
    q = p.detach()
    at_bound = [n for j, n in enumerate(s02_names) if min(abs(float(q[j]) - lo_s), abs(float(q[j]) - hi_s)) < 1e-6]
    at_bound += [n for j, n in enumerate(e0_names) if min(abs(float(q[ns + j]) - de0_bounds[0]), abs(float(q[ns + j]) - de0_bounds[1])) < 1e-6]
    return {"residual": float(f), "ratios": [[float(v) for v in r] for r in ratios],
            "s02": {n: _first(groups, n).s02 for n in s02_names}, "de0": {n: _first(groups, n).de0 for n in e0_names},
            "s02_names": list(s02_names), "de0_names": list(e0_names), "at_bound": sorted(set(at_bound))}


def likelihood_weights(groups, floors=None):
    """``W_g = N_ind,g / (2 M_g)`` per group: N_ind the sum of the independent points of the group's windows, M_g the
    misfit floor (default: the group's current weighted R-factor sum, i.e. the misfit at the optimum taken as the
    noise level)."""
    out = []
    for g, group in enumerate(groups):
        n_ind = sum(cmp.n_ind for cmp, _c, _w in group)
        if floors is not None:
            m = float(floors[g])
        else:
            with torch.no_grad():
                m = float(sum(float(w) * float(cmp.residual(chi if chi.dim() == 2 else chi[None])[0]) for cmp, chi, w in group))
        out.append(n_ind / (2.0 * max(m, 1e-30)))
    return out


def shared_intervals(groups, W=None, z=1.645):
    """Intervals of the fitted S0^2 and shifts from the curvature of the summed log-likelihood at the current values.

    Parameters
    ----------
    groups : as in ``fit_shared`` (call after the fit; the comparisons hold the fitted values).
    W : list of float, optional
        Likelihood weight per group (default ``likelihood_weights(groups)``).
    z : float
        Half-width of the interval in standard deviations (1.645: 90 %).

    Returns
    -------
    dict
        ``positive_definite`` (False: a flat or wrong-signed direction, the intervals are infinite or meaningless),
        ``W``, ``s02_sd_log``, ``s02_lo``, ``s02_hi``, ``de0_sd``, ``de0_lo``, ``de0_hi`` (each ``{name: value}``),
        ``de0_s02_correlation`` (``{name: correlation}``, when positive definite), ``z`` and ``method`` (one sentence
        that says how the intervals were obtained).
    """
    s02_names, e0_names = shared_names(groups)
    device = groups[0][0][0].device
    W = likelihood_weights(groups) if W is None else [float(v) for v in W]
    ns = len(s02_names)
    p0 = _params(groups, s02_names, e0_names, device)

    def f(q):
        total = 0.0
        for w, (fi, _r) in zip(W, _terms(groups, q, s02_names, e0_names)):
            total = total + w * fi
        return total

    n = int(p0.shape[0])
    corr, ok = None, True
    if n == 0:
        sd = torch.zeros(0, device=device)
    else:
        H = torch.autograd.functional.hessian(f, p0)
        H = 0.5 * (H + H.t())
        try:
            cov = torch.linalg.inv(H)
            sd = torch.sqrt(torch.clamp(torch.diag(cov), min=0))
            ok = bool((torch.linalg.eigvalsh(H) > 0).all())
            corr = cov / torch.sqrt(torch.diag(cov)[:, None] * torch.diag(cov)[None, :]) if ok else None
        except Exception:
            sd, ok = torch.full((n,), float("inf"), device=device), False
    i_s = {name: j for j, name in enumerate(s02_names)}
    i_e = {name: ns + j for j, name in enumerate(e0_names)}
    out = {"positive_definite": ok, "W": [float(w) for w in W], "z": float(z),
           "s02_sd_log": {name: float(sd[i_s[name]]) for name in s02_names},
           "s02_lo": {name: _first(groups, name).s02 * math.exp(-z * float(sd[i_s[name]])) for name in s02_names},
           "s02_hi": {name: _first(groups, name).s02 * math.exp(z * float(sd[i_s[name]])) for name in s02_names},
           "de0_sd": {name: float(sd[i_e[name]]) for name in e0_names},
           "de0_lo": {name: _first(groups, name).de0 - z * float(sd[i_e[name]]) for name in e0_names},
           "de0_hi": {name: _first(groups, name).de0 + z * float(sd[i_e[name]]) for name in e0_names},
           "method": ("curvature of the summed log-likelihood W x R-factor at the optimum, W = N_independent / (2 x misfit floor); "
                      f"intervals are +- {z:g} standard deviations; the structure is held fixed")}
    if corr is not None:
        out["de0_s02_correlation"] = {name: float(corr[i_e[name], i_s[name]]) for name in e0_names if name in i_s}
    return out
