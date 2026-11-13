"""Gradient refinement of atomic positions against measured spectra.

``GpuRmc`` refines copies of a cell of one element against one spectrum; ``EdgeRefinement`` (at the end of this
module) does the same for any elements and several spectra, on the edge models of the search. What follows describes
``GpuRmc``; the scheme of ``EdgeRefinement`` is the same.

An ensemble of C copies of a periodic cell of one element (every atom an absorber) is refined so that its
ensemble-averaged spectrum matches the target in R space. Single scattering is exact at every step for every pair
within the cutoff (``feffgpu.exafs.PairTables``). Multiple scattering (three- and four-leg paths) is exact too: FEFF's
path finder enumerates the paths of every absorber of every copy from the current positions (found again every
``refresh_every`` steps), the path stage evaluates them, and the multiple-scattering spectrum is the mean over all
absorbers of their last evaluation. Its gradient, too costly for every absorber at every step, comes from a rotating
subset of absorbers accumulated into a running average whose memory is one cycle through all absorbers, after one
exact pass at the start: in expectation the exact gradient.

The objective is the R-space misfit plus a hard core below ``core_distance``; nothing else: no lattice, no
coordination number, no potential. Steepest descent with the step normalized to a fixed root-mean-square move per
atom moves every coordinate at once, and a step is taken only if the misfit does not rise (Adam is available for
comparison). The start is the only prior; the dependence of the result on it is the method's uncertainty (repeat
with other starts).

The target follows EvAX's convention: the measured chi divided by S0^2, compared with the model at the energy shift
``e0`` (``feffgpu.exafs.shifted_target`` prepares data the EvAX way).
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from ..cell import as_cell
from ..exafs import PairTables, MultipleScattering

__all__ = ["GpuRmc", "EdgeRefinement"]


class GpuRmc:
    """Gradient refinement of the positions of C copies of a one-element cell.

    Parameters
    ----------
    phase : PhaseData or str
        The phase file (its scattering potential 1 is the element).
    k : array, shape (NK,)
        The uniform k grid from zero (1/A).
    target : array, shape (NK,)
        The measured chi(k) divided by S0^2, on ``k``.
    e0 : float
        Energy shift of the model (eV).
    box : float, (3,) or (3, 3)
        The periodic cell (A).
    window : (float, float)
        ``kmin``, ``kmax`` of the Hanning window (1/A).
    r_range : (float, float)
        R window of the misfit (A).
    rmax : float
        Pair and path cutoff (A).
    critpw : float
        Importance filter of the multiple-scattering paths (percent).
    chunk : int
        Paths per batch of the path stage.
    r_min : float
        Hard core (A): the energy ``0.5 core (r_min - r)^2`` per pair closer than ``r_min``.
    core : float
        Stiffness of the hard core (eV/A^2).
    ms : bool
        False: single scattering only.
    criteria_rule : str
        ``"feff"``: the multiple-scattering paths are chosen by FEFF's path finder on the structure as it is, again
        every ``refresh_every`` steps. ``"evax"``: they are chosen once on the lattice ``sites`` by EvAX's rule
        (``feffgpu.exafs.MultipleScattering.set_lattice``).
    sites : array, shape (n, 3), optional
        The lattice sites of one copy (A), for ``criteria_rule = "evax"``.
    group_tolerance : float
        For ``criteria_rule = "evax"``: the distance (A) within which the atoms of two paths must lie for the paths to
        belong to one group (EvAX's ``Pathfinder_robustness``).
    kweight : int
    dk : float
        Sill width of the Hanning window (1/A).
    device : str
    """

    def __init__(self, phase, k, target, e0, box, window=(3.0, 23.0), r_range=(1.0, 6.0), rmax=7.0, critpw=2.0, chunk=1024,
                 r_min=2.2, core=20.0, ms=True, kweight=2, dk=1.0, device="cuda", criteria_rule="feff", sites=None, group_tolerance=0.5):
        self.pairs = PairTables(phase, k, target, e0, window=window, r_range=r_range, box=box, rmax=rmax, ms=False, chunk=chunk, device=device,
                                kweight=kweight, dk=dk)
        self.ms = MultipleScattering(phase, k, e0, box, rmax=rmax, critpw=critpw, chunk=chunk, device=device, criteria_rule=criteria_rule) if ms else None
        if self.ms is not None and criteria_rule == "evax":
            if sites is None:
                raise ValueError("criteria_rule = 'evax' chooses the paths on the lattice: give its sites (n, 3)")
            self.ms.set_lattice(sites, kweight=kweight, tolerance=group_tolerance)
        self.box, self.r_min, self.core = as_cell(box, device), float(r_min), float(core)
        self.device = torch.device(device)

    # -- forward model ---------------------------------------------------------------------------------------------
    def misfit(self, chi):
        """The R-space misfit of a spectrum (NK,) to the target."""
        return self.pairs.base.misfit(chi)

    def core_energy(self, pos):
        """The hard-core energy (eV) of positions (C, n, 3)."""
        r, _ = self.pairs.pair_distances(pos)
        v = torch.clamp(self.r_min - r, min=0.0)
        return 0.5 * self.core * (v ** 2).sum() / 2            # ordered pairs: every unordered pair appears twice

    @torch.no_grad()
    def spectrum_only(self, pos, enumerate_paths=True):
        """The exact spectrum (NK,) (every absorber, every path of the list) of positions (C, n, 3) and its misfit."""
        pos = torch.as_tensor(pos if torch.is_tensor(pos) else np.asarray(pos), device=self.device, dtype=torch.float32)
        chi = self.pairs.single_scattering(pos)
        if self.ms is not None:
            if enumerate_paths or self.ms.lists is None:
                self.ms.enumerate(pos)
            chi = chi + self.ms.full_pass(pos.reshape(-1, 3)).mean(0)
        return chi, float(self.misfit(chi))

    # -- refinement ------------------------------------------------------------------------------------------------
    def refine(self, pos0, steps, lr=0.003, ms_absorbers=128, ms_alpha=1.0, value_every=10, refresh_every=250, optimizer="sgd",
               lr_final=0.1, seed=0, log_every=50, verbose=True, callback=None, log=print):
        """Gradient descent on the positions of every copy.

        Parameters
        ----------
        pos0 : array, shape (C, n, 3)
            The start (A).
        steps : int
        lr : float
            ``"sgd"``: the root-mean-square move per atom per step (A); ``"adam"``: Adam's step.
        ms_absorbers : int
            Absorbers per step whose exact multiple-scattering gradient enters the running average.
        ms_alpha : float
            Memory of the running average in cycles through all absorbers.
        value_every : int
            Steps between full passes of every absorber's multiple-scattering spectrum (its value in the misfit).
        refresh_every : int
            Steps between new searches of the path lists.
        optimizer : str
            ``"sgd"`` (steepest descent with a normalized step and an accepted-step rule) or ``"adam"``.
        lr_final : float
            The step decays on a cosine to ``lr x lr_final``.
        seed : int
            Seed of the order in which the absorbers rotate.
        log_every : int
        verbose : bool
        callback : callable, optional
            ``callback(step, positions, misfit)`` at every log.
        log : callable
            Receives the progress lines.

        Returns
        -------
        positions : numpy.ndarray, shape (C, n, 3)
        history : list of dict
        """
        pos = torch.tensor(np.asarray(pos0), device=self.device, dtype=torch.float32, requires_grad=True)
        start = pos.detach().clone()
        C, n, _ = pos.shape
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        opt = torch.optim.Adam([pos], lr=lr) if optimizer == "adam" else None
        history, t0 = [], time.time()
        timing = {"enumerate": 0.0, "full_pass": 0.0, "full_gradient": 0.0, "subset": 0.0}
        step_size, n_halved, n_rejected, accept_tol = lr, 0, 0, 1e-3
        ms = self.ms
        sync = (lambda: torch.cuda.synchronize()) if self.device.type == "cuda" else (lambda: None)
        counts = {}
        if ms is not None:
            A = min(int(ms_absorbers), C * n)
            alpha = ms_alpha * A / (C * n)
            perm, ptr = torch.randperm(C * n, device=self.device, generator=gen), 0
            G = None
        for step in range(steps + 1):
            frac = lr_final + (1 - lr_final) * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))
            if ms is not None and step % refresh_every == 0:
                t = time.time()
                counts = ms.enumerate(pos.detach())
                sync()
                timing["enumerate"] += time.time() - t
            if ms is not None and (step % value_every == 0 or step % refresh_every == 0):
                t = time.time()
                ms.full_pass(pos.detach().reshape(-1, 3))
                sync()
                timing["full_pass"] += time.time() - t
                if verbose and step == 0:
                    log(f"   paths: {counts} for {C * n} absorbers ({timing['enumerate']:.0f} s to find, {timing['full_pass']:.0f} s for a full pass)")
            chi_ss = self.pairs.single_scattering(pos)
            if ms is not None:
                if ptr + A > C * n:
                    perm, ptr = torch.randperm(C * n, device=self.device, generator=gen), 0
                subset = perm[ptr:ptr + A]
                ptr += A
                t = time.time()
                pos2 = pos.detach().clone().requires_grad_(True)
                chi_sub = ms.chi_absorbers(pos2.reshape(-1, 3), subset)
                with torch.no_grad():
                    ms.rows[subset] = chi_sub.detach()
                sub_mean = chi_sub.mean(0)
                chi = chi_ss + ms.mean.detach() + (sub_mean - sub_mean.detach())          # value: every absorber; gradient: the subset
            else:
                chi = chi_ss
            M = self.misfit(chi)
            Ec = self.core_energy(pos)
            if step % log_every == 0 or step == steps:
                with torch.no_grad():
                    d = pos - start
                    rms, dmax = float(d.pow(2).mean().sqrt()), float(d.abs().max())
                history.append({"step": step, "misfit": float(M.detach()), "core": float(Ec.detach()), "rms_move": rms, "max_move": dmax, "step_size": step_size * frac,
                                "halvings": n_halved, "seconds": time.time() - t0})
                if verbose:
                    log(f"   step {step:5d}: misfit {float(M.detach()):.4f}, core {float(Ec.detach()):.2e} eV, rms move {rms:.4f} A (max {dmax:.3f}), step "
                        f"{step_size * frac:.5f} A ({n_halved} halvings so far), {time.time() - t0:.0f} s")
                if callback is not None:
                    callback(step, pos.detach(), float(M.detach()))
            if step == steps:
                break
            if ms is not None:
                g_ss, g_sub = torch.autograd.grad(M + Ec, [pos, pos2])
                sync()
                timing["subset"] += time.time() - t
                if G is None:                                                             # one exact pass to start the average
                    t = time.time()
                    loss = lambda chi_ms: self.misfit(chi_ss.detach() + chi_ms)          # noqa: E731
                    G = ms.full_gradient(pos.detach().reshape(-1, 3), loss).reshape(C, n, 3)
                    sync()
                    timing["full_gradient"] += time.time() - t
                else:
                    G = (1 - alpha) * G + alpha * g_sub.reshape(C, n, 3)
                g = g_ss + G
            else:
                (g,) = torch.autograd.grad(M + Ec, pos)
            with torch.no_grad():
                if opt is not None:
                    pos.grad = g
                    for pg in opt.param_groups:
                        pg["lr"] = lr * frac
                    opt.step()
                else:
                    # steepest descent with an accepted-step rule: the move is taken only if the misfit (single scattering
                    # live, multiple scattering as held) does not rise; otherwise the step halves and the same gradient is
                    # tried again
                    direction = g / (g.pow(2).mean().sqrt() + 1e-30)
                    M_here = float(M.detach()) + float(Ec.detach())
                    ms_held = ms.mean if ms is not None else 0.0
                    accepted, s = False, step_size
                    for _ in range(8):
                        trial = pos - s * frac * direction
                        M_try = float(self.misfit(self.pairs.single_scattering(trial) + ms_held)) + float(self.core_energy(trial))
                        if M_try <= M_here * (1 + accept_tol):
                            pos.copy_(trial)
                            accepted = True
                            step_size = min(s * 1.2, lr)
                            break
                        s *= 0.5
                        n_halved += 1
                    if not accepted:                                                      # the direction itself is wrong: skip, halve once
                        n_rejected += 1
                        step_size = max(step_size * 0.5, lr * 1e-3)
        self.timing = timing
        self.steps_info = {"halvings": n_halved, "rejected_steps": n_rejected, "final_step_size": step_size if opt is None else lr}
        return pos.detach().cpu().numpy(), history


class EdgeRefinement:
    """Gradient refinement of the atoms of a structure with any elements against one or several spectra.

    The same scheme as ``GpuRmc``, built on the edge models of the search (``feffgpu.exafs.EdgeModel``): single
    scattering of every pair with its gradient at every step; multiple scattering with its gradient from a rotating
    subset of absorbers, accumulated into a running average after one exact pass at the start, and its value from
    every absorber's last evaluation (all renewed every ``value_every`` steps). Each spectrum has its own misfit (the
    edge's ``objective``: the R-factor or a wavelet residual, with the edge's S0^2 and energy shift on the model); the
    refinement lowers their sum. A step moves every coordinate at once by a fixed root-mean-square distance and is
    taken only if the sum is not predicted to rise: single scattering recomputed for the moved atoms, the change of
    the multiple scattering to first order from its gradient. At every full pass the exact sum is compared with that
    of the pass before; if it rose, the step is halved.

    The multiple-scattering gradient is what the gradient steps of the population search leave out. Where multiple
    scattering is strong (metals, linear chains of atoms) atoms that follow the single-scattering gradient alone,
    with the multiple scattering held, do not converge; this refinement does.

    Parameters
    ----------
    edges : list of feffgpu.exafs.EdgeModel
        The spectra, all of the same structure (the same atoms in the same order).
    sites : array, shape (n, 3)
        The start lattice (A): the reference of the displacement limit and, for ``criteria_rule = "evax"``, the
        structure the paths are chosen on.
    symbols : array of str, shape (n,)
    cell : None, float or (3, 3)
    max_displacement : float
        Every atom stays within this distance of its site (A); 0: no limit.
    closest_distance : dict, optional
        ``{"A-B": distance}`` (A): springs act on pairs closer than this.
    core_distance : float
        A closest distance for every pair of elements that ``closest_distance`` does not name (A); 0: none.
    spring : float
        Stiffness of the springs (eV/A^2).
    """

    def __init__(self, edges, sites, symbols, cell, max_displacement=0.0, closest_distance=None, core_distance=0.0, spring=20.0):
        from .constraints import Box, ClosestDistances
        self.edges = list(edges)
        if not self.edges:
            raise ValueError("the refinement needs at least one spectrum")
        self.device = self.edges[0].device
        self.sites = torch.as_tensor(np.asarray(sites, float), device=self.device, dtype=torch.float32)
        self.n = int(self.sites.shape[0])
        self.symbols = np.asarray(symbols).astype(str)
        self.cell = as_cell(cell, self.device)
        for e in self.edges:
            if e.n != self.n:
                raise ValueError(f"spectrum {e.name}: its model has {e.n} atoms, the structure {self.n}")
            if e.ms is not None and e.criteria_rule == "evax":
                e.ms.set_lattice(self.sites[e.keep], e.ipot[e.keep], e.abs_keep, kweight=e.kweight, tolerance=e.group_tolerance)
        self.box = Box(np.asarray(sites, float), max_displacement, cell, device=self.device) if max_displacement and max_displacement > 0 else None
        table = dict(closest_distance or {})
        if core_distance and core_distance > 0:
            elements = sorted(set(self.symbols.tolist()))
            named = {frozenset(str(key).split("-")) for key in table}
            for i, a in enumerate(elements):
                for b in elements[i:]:
                    if frozenset((a, b)) not in named:
                        table[f"{a}-{b}"] = float(core_distance)
        self.closest = ClosestDistances(self.symbols, table, cell, spring=spring, device=self.device)
        self.steps_info, self.timing = {}, {}

    # -- forward model ---------------------------------------------------------------------------------------------
    def _single(self, e, pos):
        """The single-scattering chi (NK,) of edge ``e``, averaged over the copies ``pos`` (C, n, 3); differentiable."""
        return e.spectrum(e.static_histograms(pos).mean(0, keepdim=True))[0]

    def _paths(self, e, pos):
        return e.ms.enumerate(pos[:, e.keep], e.ipot[e.keep], e.abs_keep)

    @torch.no_grad()
    def spectra(self, pos, enumerate_paths=True):
        """The exact spectra of the copies ``pos`` (C, n, 3), one (NK,) per spectrum, unscaled (S0^2 = 1, at the edge's
        energy shift), and their misfits (each edge's objective with its S0^2 and shift on the model)."""
        pos = torch.as_tensor(pos if torch.is_tensor(pos) else np.asarray(pos), device=self.device, dtype=torch.float32)
        chis, misfits = [], []
        for e in self.edges:
            chi = self._single(e, pos)
            if e.ms is not None:
                if enumerate_paths or e.ms.lists is None:
                    self._paths(e, pos)
                rows = e.ms.full_pass(pos[:, e.keep].reshape(-1, 3))
                chi = chi + rows.mean(0).to(chi.dtype)
            chis.append(chi)
            misfits.append(float(e.residual(chi[None].float())[0]))
        return chis, misfits

    def _limit(self, pos):
        if self.box is None:
            return pos
        site_of = torch.arange(self.n, device=self.device)[None].expand(pos.shape[0], -1)
        return self.box.project(pos, site_of)

    # -- refinement ------------------------------------------------------------------------------------------------
    def refine(self, pos0, steps, step_size=0.003, final_fraction=0.1, ms_absorbers=128, ms_memory=1.0, value_every=10, refresh_every=250, seed=0,
               log_every=50, callback=None, log=print):
        """Gradient descent on the positions of every copy.

        Parameters
        ----------
        pos0 : array, shape (C, n, 3)
            The start (A).
        steps : int
        step_size : float
            Root-mean-square move of a coordinate per step (A), decaying on a cosine to ``step_size x final_fraction``.
        ms_absorbers : int
            Absorbers of each spectrum per step whose exact multiple-scattering gradient enters the running average.
        ms_memory : float
            Memory of the running average in cycles through all absorbers.
        value_every : int
            Steps between full passes of every absorber's multiple-scattering spectrum.
        refresh_every : int
            Steps between new searches of the path lists (``criteria_rule = "feff"``; with ``"evax"`` the lists are
            those of the start lattice throughout).
        seed : int
        log_every : int
        callback : callable, optional
            ``callback(step, positions, misfits)`` at every log.
        log : callable or None

        Returns
        -------
        positions : numpy.ndarray, shape (C, n, 3)
        history : list of dict
        """
        log = log or (lambda *a: None)
        pos = self._limit(torch.as_tensor(np.asarray(pos0), device=self.device, dtype=torch.float32)).clone()
        start = pos.clone()
        C = pos.shape[0]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        state = []
        for e in self.edges:
            N = C * e.nA
            A = min(int(ms_absorbers), N)
            state.append(dict(N=N, A=A, alpha=ms_memory * A / N, perm=torch.randperm(N, device=self.device, generator=gen), ptr=0, G=None, rows=None))
        history, t0 = [], time.time()
        timing = {"paths": 0.0, "full_pass": 0.0, "full_gradient": 0.0, "steps": 0.0}
        step, n_halved, n_rejected, n_rose, last_exact = float(step_size), 0, 0, 0, None
        sync = (lambda: torch.cuda.synchronize()) if self.device.type == "cuda" else (lambda: None)
        for it in range(steps + 1):
            frac = final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * it / max(steps, 1)))
            renew = it % refresh_every == 0
            grad = torch.zeros_like(pos)
            parts = []
            exact = all(e.ms is None for e in self.edges)                 # whether the misfits of this step are exact (every absorber fresh)
            t = time.time()
            for e, st in zip(self.edges, state):
                xs = pos.clone().requires_grad_(True)
                chi_ss = self._single(e, xs)
                if e.ms is None:
                    m = e.residual(chi_ss[None].float())[0]
                    (g,) = torch.autograd.grad(m, xs)
                    parts.append(float(m.detach()))
                    grad += g
                    continue
                flat = pos[:, e.keep].reshape(-1, 3)
                if renew:
                    t1 = time.time()
                    counts = self._paths(e, pos)
                    sync()
                    timing["paths"] += time.time() - t1
                    if it == 0:
                        log(f"   {e.name}: paths {counts} for {st['N']} absorbers ({st['A']} per step)")
                if renew or it % value_every == 0 or it == steps:
                    t1 = time.time()
                    with torch.no_grad():
                        st["rows"] = e.ms.full_pass(flat).clone()
                    sync()
                    timing["full_pass"] += time.time() - t1
                    exact = True
                if st["ptr"] + st["A"] > st["N"]:
                    st["perm"], st["ptr"] = torch.randperm(st["N"], device=self.device, generator=gen), 0
                sub = st["perm"][st["ptr"]:st["ptr"] + st["A"]]
                st["ptr"] += st["A"]
                xm = pos.clone().requires_grad_(True)
                chi_sub = e.ms.chi_absorbers(xm[:, e.keep].reshape(-1, 3), e.ms.abs_atoms[sub])
                with torch.no_grad():
                    st["rows"][sub] = chi_sub.detach()
                sub_mean = chi_sub.mean(0)
                held = st["rows"].mean(0).detach()
                chi = chi_ss + (held + (sub_mean - sub_mean.detach())).to(chi_ss.dtype)      # value: every absorber; gradient: the subset
                m = e.residual(chi[None].float())[0]
                parts.append(float(m.detach()))
                g_ss, g_sub = torch.autograd.grad(m, [xs, xm])
                if st["G"] is None:                                                          # one exact pass to start the average
                    t1 = time.time()
                    ss = chi_ss.detach()
                    loss = lambda chi_ms, e=e, ss=ss: e.residual((ss + chi_ms.to(ss.dtype))[None].float())[0]      # noqa: E731
                    e.ms.rows = st["rows"].clone()
                    gk = e.ms.full_gradient(flat, loss)
                    st["G"] = torch.zeros_like(pos)
                    st["G"][:, e.keep] = gk.reshape(C, -1, 3).to(pos.dtype)
                    sync()
                    timing["full_gradient"] += time.time() - t1
                else:
                    st["G"] = (1 - st["alpha"]) * st["G"] + st["alpha"] * g_sub
                grad += g_ss + st["G"]
            E_here = 0.0
            if self.closest.active:
                xc = pos.clone().requires_grad_(True)
                E = self.closest.energy(xc)[0].sum()
                E_here = float(E.detach())
                if E_here > 0:
                    (gc,) = torch.autograd.grad(E, xc)
                    grad += gc
            sync()
            timing["steps"] += time.time() - t
            if exact:                                                     # the exact sum against that of the last full pass
                total = sum(parts) + E_here
                if last_exact is not None and total > last_exact * (1 + 1e-3):
                    step = max(step * 0.5, float(step_size) * 1e-3)
                    n_halved += 1
                    n_rose += 1
                last_exact = total
            if it % log_every == 0 or it == steps:
                with torch.no_grad():
                    d = pos - start
                    rms, dmax = float(d.pow(2).mean().sqrt()), float(d.abs().max())
                history.append({"step": it, "misfit": list(parts), "springs": E_here, "rms_move": rms, "max_move": dmax, "step_size": step * frac,
                                "halvings": n_halved, "seconds": time.time() - t0})
                log(f"   step {it:5d}: misfit " + ", ".join(f"{m:.5f}" for m in parts) + f"; rms move {rms:.4f} A (max {dmax:.3f}), step {step * frac:.5f} A "
                    f"({n_halved} halvings so far), {time.time() - t0:.0f} s")
                if callback is not None:
                    callback(it, pos.detach(), list(parts))
            if it == steps:
                break
            with torch.no_grad():
                direction = grad / (grad.pow(2).mean().sqrt() + 1e-30)
                here = sum(parts) + E_here
                accepted, s = False, step
                for _ in range(8):
                    trial = self._limit(pos - s * frac * direction)
                    value = float(self.closest.energy(trial)[0].sum()) if self.closest.active else 0.0
                    for e, st in zip(self.edges, state):
                        chi = self._single(e, trial)
                        if e.ms is not None:
                            chi = chi + st["rows"].mean(0).to(chi.dtype)
                        value += float(e.residual(chi[None].float())[0])
                        if e.ms is not None:                                 # the change of the multiple scattering, to first order
                            value += float((st["G"] * (trial - pos)).sum())
                    if value <= here * (1 + 1e-3):
                        pos.copy_(trial)
                        accepted = True
                        step = min(s * 1.2, float(step_size))
                        break
                    s *= 0.5
                    n_halved += 1
                if not accepted:                                                             # the direction itself is wrong: skip, halve once
                    n_rejected += 1
                    step = max(step * 0.5, float(step_size) * 1e-3)
        self.timing = timing
        self.steps_info = {"halvings": n_halved, "rejected_steps": n_rejected, "rises_at_full_passes": n_rose, "final_step_size": step}
        return pos.detach().cpu().numpy(), history
