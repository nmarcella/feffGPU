"""The population search under EvAX's rules, for any structure, any number of samples and edges.

A run holds, for every sample, a population of ``states`` states (each ``cells_per_state`` copies of the model cell).
One generation, for every sample (``docs/rmc/search.md`` describes it step by step):

1. Selection (EvAX's ``selectionT``): pairs of parents, each the better of two random states; in half the pairs the
   two children exchange the displacements of the atoms on each site with probability 0.5 (EvAX's crossover, by
   lattice site); a child that violates a constraint is replaced by its unbred parent. A bred child's multiple
   scattering is computed from its own atoms.
2. The move of every state: exchanges and hops (if configured), EvAX's uniform random step on every coordinate,
   then (if configured) FIRE gradient steps on the misfit with the displacement limit and the springs of the closest
   distances, the multiple scattering of the state before the move held.
3. The trial's own multiple scattering (its path lists found again around atoms that changed site), the spectrum
   weights (EvAX's ``adjust_spectra_weights``), and EvAX's acceptance of the trial by every state, cooled to zero at
   ``froze_in``.

Every ``paths_every`` generations all path lists are found again. In mode ``fit``, S0^2 and the E0 shift of every
edge name are refitted jointly over the samples on their best states (``feffgpu.exafs.fit_shared``) every
``refit_every`` generations from ``refit_from`` on, and once at the end.

The random numbers of a sample come from one torch generator seeded with ``seed + 11``, drawn in a fixed order.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from ..cell import wrap
from ..exafs import fit_shared, shared_intervals, likelihood_weights
from .constraints import Box, ClosestDistances, nearest_sites
from .moves import start_population, uniform_step, exchange_atoms, hop_atoms, HopTables, fire_relax, GradientObjective
from .paths import StatePaths
from .rules import EdgeWeights, Annealing, evax_selection, site_crossover
from .structure import ArrangementStatistics
from .outputs import METHOD, FORMAT, write_json, load_continuation, shared_values_of_run
from .config import SearchSettings, AmplitudeSettings, MoveSettings, ConstraintSettings

__all__ = ["RmcSample", "EvaxSearch"]


@dataclass
class RmcSample:
    """One sample of a search: its model cell (with sites) and its edges (``feffgpu.exafs.edge.EdgeModel``, built on the
    model cell's symbols and cell)."""
    label: str
    model: object
    edges: list


def _state_distance(a, b, cell, movable):
    """(P, Q): rms distance per movable atom between the states a (P, C, n, 3) and b (Q, C, n, 3)."""
    d = wrap(a[:, None] - b[None], cell).norm(dim=-1) ** 2
    return ((d * movable).sum((2, 3)) / (movable.sum() * a.shape[1])).sqrt()


class _Best:
    """The evaluation of one state (its cells and its own multiple scattering) with the current S0^2 and E0 shifts."""

    def __init__(self, edges, pos, ms_means, weights):
        self.pos = pos
        self.hists = [e.static_histograms(pos) for e in edges]
        self.ms_means = ms_means
        self.finish(edges, weights)

    def finish(self, edges, weights):
        self.chi = [e.spectrum(h, e.tables, m) for e, h, m in zip(edges, self.hists, self.ms_means)]
        with torch.no_grad():
            self.ratios = [float(e.residual(c.mean(0, keepdim=True))[0]) for e, c in zip(edges, self.chi)]
        self.M = float(np.dot(weights, self.ratios))

    def group(self, edges, weights):
        """The entries of a joint fit of S0^2 and the E0 shift: (edge, spectrum (1, NK), weight)."""
        out = []
        for e, h, m, w in zip(edges, self.hists, self.ms_means, weights):
            out.append((e, e.spectrum(h.mean(0)[None], e.tables, m.mean(0)[None]), float(w)))
        return out

    def record(self, edges):
        return {"misfit": self.M, "ratios": {e.name: r for e, r in zip(edges, self.ratios)},
                "s02": {e.name: e.s02 for e in edges}, "de0": {e.name: e.de0 for e in edges}}


class _State:
    """The search state of one sample (internal)."""


class EvaxSearch:
    """The population search of RMC-EXAFS under EvAX's rules.

    Parameters
    ----------
    samples : list of RmcSample
    search : SearchSettings
    amplitude, moves, constraints : AmplitudeSettings, MoveSettings, ConstraintSettings, optional
    seed : int
    output : str or Path, optional
        Output directory (one subdirectory per sample); None: nothing is written.
    save_every : int
        Generations between checkpoints.
    log : callable
        Receives the progress lines (default ``print``; ``None`` for silence).
    resolved : dict, optional
        The resolved configuration, recorded in the reports.
    """

    def __init__(self, samples, search: SearchSettings, amplitude: AmplitudeSettings = None, moves: MoveSettings = None,
                 constraints: ConstraintSettings = None, seed=1, output=None, save_every=25, log=print, resolved=None):
        self.samples_in = list(samples)
        self.S = search
        self.A = amplitude or AmplitudeSettings()
        self.Mv = moves or MoveSettings()
        self.K = constraints or ConstraintSettings()
        self.seed, self.save_every = int(seed), int(save_every)
        self.output = None if output is None else Path(output)
        self.resolved = resolved
        self._log = log
        if self.S.states < 2 or self.S.states % 2:
            raise ValueError("search.states must be even and at least 2 (EvAX breeds the states in pairs)")
        rule = self.K.box_rule
        self.box_rule = ("halve" if self.S.gradient_steps == 0 else "project") if rule == "auto" else rule
        if self.box_rule == "halve" and self.S.gradient_steps > 0:
            raise ValueError("constraints.box_rule 'halve' (EvAX's rule) applies to random moves only; with gradient steps use 'project'")
        labels = [q.label for q in self.samples_in]
        if len(set(labels)) != len(labels):
            raise ValueError("the sample labels must differ")
        self.fit_mode = self.A.mode == "fit"
        for q in self.samples_in:
            for e in q.edges:
                e.fit_s02 = e.fit_e0 = self.fit_mode
        self.t0 = time.time()

    def log(self, text):
        if self._log is not None:
            self._log(text)

    # ------------------------------------------------------------------------------------------------------------
    # the start
    # ------------------------------------------------------------------------------------------------------------
    def _start_sample(self, inp):
        S, A, Mv, K = self.S, self.A, self.Mv, self.K
        P, C = S.states, S.cells_per_state
        q = _State()
        q.label, q.model, q.edges = inp.label, inp.model, list(inp.edges)
        e0 = q.edges[0]
        q.dev = e0.device
        dev = q.dev
        q.gen = torch.Generator(device=dev)
        q.gen.manual_seed(self.seed + 11)
        model = q.model
        q.n, q.C = model.n, C
        n = q.n
        q.base = torch.as_tensor(np.repeat(model.positions[None], C, 0), device=dev, dtype=torch.float32)          # (C, n, 3) the start lattice
        q.species_names = model.elements
        q.sp = torch.tensor(model.species_code, device=dev)
        ignored_by_all = set.intersection(*[set(e.ignore) for e in q.edges])
        q.movable = torch.tensor([v not in ignored_by_all for v in model.symbols], device=dev, dtype=torch.float32)
        q.pop = start_population(q.base, P, S.start_kick, q.movable, q.gen, S.start_distribution)
        q.site = torch.as_tensor(model.sites.start_site_of, device=dev, dtype=torch.long)[None, None].repeat(P, C, 1)
        ne = len(q.edges)
        q.weights = EdgeWeights(ne, dev)
        q.anneal = Annealing(S.acceptance, S.froze_in)
        q.from_population, q.from_scale = None, None
        if S.continue_from:
            src = Path(S.continue_from) / q.label
            loaded = load_continuation(src, model.symbols, q.base.cpu().numpy(), P, C)
            rep = loaded["report"]
            names = [e.name for e in q.edges]
            if list(rep["edges"]) != names:
                raise ValueError(f"{src}: the edges differ ({rep['edges']} there, {names} here)")
            q.pop = torch.as_tensor(loaded["population"], device=dev, dtype=torch.float32)
            q.site = torch.as_tensor(loaded["site_of"], device=dev, dtype=torch.long)
            q.from_scale = float(rep["scale"]) if rep.get("scale") else None
            fit, sr = rep["final_refit"], rep["search"]
            for e in q.edges:
                e.set_s02(float(fit["s02"][e.name]))
                e.set_de0(float(fit["de0"][e.name]) - e.e0)
            q.weights = EdgeWeights(ne, dev, sr["weights"]["weights"], sr["weights"]["average_delta"], sr["weights"]["num"])
            q.anneal = Annealing(S.acceptance, S.froze_in, sr["acceptance"]["average_delta"], sr["acceptance"]["num"])
            if S.generation_offset is None:
                self.offset = int(sr["generations_done"])
            q.from_population = {"path": str(src), "states": [int(v) for v in loaded["order"]], "misfits": [float(v) for v in loaded["misfits"]],
                                 "generations_done": int(sr["generations_done"])}
            self.log(f"[{q.label}]   population of {P} loaded from {src}: saved misfits {loaded['misfits'].min():.5f} - {loaded['misfits'].max():.5f}; "
                     f"its S0^2, E0 shifts, spectrum weights " + "/".join(f"{v:.3f}" for v in q.weights.w.tolist()) +
                     f", running means and {sr['generations_done']} generations done")
        else:
            for e in q.edges:
                e.set_s02(e.s02_init)
                e.set_de0(0.0)
        if A.from_run:
            s02s, de0s = shared_values_of_run(A.from_run)
            for e in q.edges:
                if e.name not in s02s or e.name not in de0s:
                    raise ValueError(f"amplitude.from_run: {A.from_run} has no shared values for the spectrum {e.name!r}")
                e.set_s02(s02s[e.name])
                e.set_de0(de0s[e.name] - e.e0)
            self.log(f"[{q.label}]   S0^2 and E0 shifts taken from {A.from_run}" + ("; held" if not self.fit_mode else "; start values of the fit"))
        elif A.mode == "fixed":                                                                    # held at the configured values, also in a continued run
            for e in q.edges:
                e.set_s02(e.s02_init)
                e.set_de0(0.0)
        q.box = Box(model.sites.positions, K.max_displacement, model.cell, dev) if K.max_displacement > 0 else None
        if q.box is not None:
            frac0, max0 = q.box.at_limit(q.pop.view(-1, n, 3), q.site.view(-1, n))
            q.pop = q.box.project(q.pop.view(-1, n, 3), q.site.view(-1, n)).view_as(q.pop)
            self.log(f"[{q.label}]   displacement limit {K.max_displacement:.2f} A from the sites of the start lattice: the start had {100 * frac0:.1f}% "
                     f"of the atoms at or beyond it (largest {max0:.3f} A), projected")
        q.site_pos = torch.as_tensor(model.sites.positions, device=dev, dtype=torch.float32)
        q.site_ids = torch.as_tensor(model.sites.ids, device=dev, dtype=torch.long)
        q.n_ids = int(model.sites.n_ids)
        q.cell = e0.cell
        for e in q.edges:                                            # EvAX's rule: the paths are chosen once, on the start lattice
            if e.ms is not None and e.criteria_rule == "evax":
                counts = e.ms.set_lattice(torch.as_tensor(model.positions, dtype=torch.float32)[e.keep.cpu()], e.ipot[e.keep], e.abs_keep, kweight=e.kweight)
                kept = [t for t in e.ms.types if t["kept"]]
                self.log(f"[{q.label}]   {e.name}: paths chosen on the start lattice by EvAX's rule ({e.critpw} % of the size of the lattice's "
                         f"spectrum): {len(kept)} types, {counts} paths")
        q.paths = [StatePaths(e) for e in q.edges]
        q.history, q.refits = [], []
        q.counts_total = {"accepted_worse": 0, "children_rejected": 0, "refused": 0}
        # every state's own path lists and multiple scattering
        t = time.time()
        flat = q.pop.view(-1, n, 3)
        q.lists = [[sp_.find(flat[j]) for j in range(P * C)] for sp_ in q.paths]
        t_find = time.time() - t
        q.ms = [sp_.signal(L, flat).view(P, C, -1).mean(1) for sp_, L in zip(q.paths, q.lists)]
        t_sig = time.time() - t - t_find
        q.w_np = q.weights.w.cpu().numpy().copy()
        q.best = _Best(q.edges, q.pop[0].clone(), [m[0][None] for m in q.ms], q.w_np)
        q.start_pos = q.best.pos.clone()
        q.start = q.best.record(q.edges)
        q.chi_start = {e.name: c.mean(0).cpu().numpy() for e, c in zip(q.edges, q.best.chi)}
        q.closest = ClosestDistances(model.symbols, K.closest_distance, model.cell, K.spring, dev)
        q.objective = GradientObjective(q.edges, C, q.closest, q.movable)
        q.objective.weights = q.w_np
        M0, grad = q.objective.state_misfit_and_gradient(q.best.pos, q.best.ms_means)
        gmax = float((grad * q.movable[None, :, None]).norm(dim=-1).max())
        q.scale = q.from_scale or S.gradient_force / max(gmax, 1e-30)
        q.scale_source = "the first run (kept)" if q.from_scale else "this run's start structure"
        q.objective.scale = q.scale
        q.objective.ms = q.ms
        self._rescore(q)
        # rearrangements
        ex = list(Mv.exchange)
        q.can_exchange = bool(ex) and all((model.symbols == v).any() for v in ex) and Mv.exchange_probability > 0
        if q.can_exchange:
            q.ex_a = torch.as_tensor(np.nonzero(model.symbols == ex[0])[0], device=dev, dtype=torch.long)
            q.ex_b = torch.as_tensor(np.nonzero(model.symbols == ex[1])[0], device=dev, dtype=torch.long)
        sites = model.sites
        q.hops = HopTables(sites, dev) if sites.hop_element is not None else None
        q.can_hop = (Mv.hops is not None and q.hops is not None and Mv.hops.probability > 0 and len(sites.vacant_hop_sites(sites.start_site_of)) > 0)
        q.stats = ArrangementStatistics(model, ex if q.can_exchange else None)
        q.arr_start = q.stats(q.site[0].cpu().numpy())
        paths0 = [StatePaths.count(L[0]) for L in q.lists]
        self.log(f"[{q.label}] {model.describe()}")
        for e in q.edges:
            self.log(f"[{q.label}]   {e.describe()}")
        self.log(f"[{q.label}] start: state 0 misfit {q.best.M:.5f} (R " + " / ".join(f"{r:.4f}" for r in q.best.ratios) + "); paths of state 0 per edge "
                 f"{paths0}; {P * C} cells: {t_find:.1f} s to find the paths, {t_sig:.1f} s to evaluate them")
        self.log(f"[{q.label}]   population of {P}: misfits {float(q.M.min()):.5f} - {float(q.M.max()):.5f}; gradient scale {q.scale:.3e} eV "
                 f"(from {q.scale_source}); closest distances (A): " + (", ".join(f"{k} {v:.2f}" for k, v in q.closest.as_dict().items()) or "none")
                 + ("; exchanges " + "/".join(ex) if q.can_exchange else "") + (f"; hops of {sites.hop_element}" if q.can_hop else ""))
        return q

    # ------------------------------------------------------------------------------------------------------------
    # helpers (the order of operations is that of the original launcher, so that a seed reproduces a run)
    # ------------------------------------------------------------------------------------------------------------
    def _rescore(self, q):
        """Every state's misfits again with the current S0^2, E0 shifts and weights (the multiple scattering unchanged)."""
        q.R = q.objective.edge_ratios(q.pop.view(-1, q.n, 3), q.ms)
        q.M = q.R @ q.weights.w

    def _evaluate(self, q, flat, lists):
        """States ``flat`` (S x C, n, 3) with their path lists -> their multiple scattering per edge (S, NK) and their
        misfit per edge (S, n_edges), each from the state's own atoms."""
        S = flat.shape[0] // q.C
        ms = [sp_.signal(L, flat).view(S, q.C, -1).mean(1) for sp_, L in zip(q.paths, lists)]
        return ms, q.objective.edge_ratios(flat, ms)

    def _find_all(self, q):
        """All path lists found again; every state evaluated again."""
        t = time.time()
        flat = q.pop.view(-1, q.n, 3)
        q.lists = [[sp_.find(flat[j]) for j in range(flat.shape[0])] for sp_ in q.paths]
        t_find = time.time() - t
        q.ms, q.R = self._evaluate(q, flat, q.lists)
        q.M = q.R @ q.weights.w
        q.objective.ms = q.ms
        return t_find, time.time() - t - t_find

    def _set_weights(self, q):
        q.w_np = q.weights.w.cpu().numpy().copy()
        q.objective.weights = q.w_np

    def _best_state(self, q):
        """The best state, with its own multiple scattering, evaluated alone (for the joint fit and the records)."""
        best = int(q.M.argmin())
        q.best = _Best(q.edges, q.pop[best].clone(), [m[best][None] for m in q.ms], q.w_np)
        if abs(q.best.M - float(q.M[best])) > 1e-3 * max(q.best.M, 1e-12):
            self.log(f"[{q.label}] WARNING: the best state's misfit evaluated alone ({q.best.M:.6f}) differs from the population's record "
                     f"({float(q.M[best]):.6f})")
        return best

    def _joint_refit(self, states):
        groups = [q.best.group(q.edges, q.w_np) for q in states]
        A = self.A
        return fit_shared(groups, s02_bounds=tuple(A.s02_bounds), de0_bounds=tuple(A.de0_bounds), iters=A.fit_iterations)

    def _bad(self, q, kids, parents):
        """Bred children that are refused: a pair below a closest distance by more than the tolerance, or an atom outside
        the displacement limit."""
        n, C = q.n, q.C
        flat = kids.view(-1, n, 3)
        b = q.closest.worst(flat).view(-1, C).amax(1) > self.K.closest_tolerance
        if q.box is not None:
            site = q.site[torch.as_tensor(parents, device=q.dev)].view(-1, n)
            b = b | ((q.box.project(flat, site) - flat).norm(dim=-1).view(-1, C * n).amax(1) > 1e-4)
        return b

    def search_state(self, q, g):
        return {"weights": q.weights.state(), "acceptance": q.anneal.state(), "generation_offset": self.offset, "generations_done": self.offset + g,
                "accepted_worse": q.counts_total["accepted_worse"], "children_rejected": q.counts_total["children_rejected"],
                "refused": q.counts_total["refused"]}

    def _checkpoint(self, q, g):
        if self.output is None:
            return
        o = self.output / q.label
        o.mkdir(parents=True, exist_ok=True)
        cell = q.model.cell if q.model.cell is not None else np.full((3, 3), np.nan)
        np.savez_compressed(o / "checkpoint.npz", population=q.pop.cpu().numpy(), site_of=q.site.cpu().numpy(), misfits=q.M.cpu().numpy(),
                            generation=g, cell=cell, symbols=q.model.symbols, ideal=q.base.cpu().numpy(), edges=np.array([e.name for e in q.edges]),
                            s02=np.array([e.s02 for e in q.edges]), de0=np.array([e.de0 for e in q.edges]), search=json.dumps(self.search_state(q, g)))
        write_json(o / "history.json", {"generations": q.history, "refits": q.refits, "search": self.search_state(q, g)})

    # ------------------------------------------------------------------------------------------------------------
    # one generation
    # ------------------------------------------------------------------------------------------------------------
    def _generation(self, g, states):
        S, A, Mv = self.S, self.A, self.Mv
        G = S.generations
        t_evax = self.offset + g                                                                 # EvAX's count: 1 at the first generation of the first run
        rearrange = g >= Mv.from_generation
        refresh = g > 1 and (g - 1) % S.paths_every == 0
        refit = self.fit_mode and A.refit_from > 0 and g >= A.refit_from and (g - A.refit_from) % A.refit_every == 0
        t_refresh = {}
        if refresh:
            for q in states:
                t_refresh[q.label] = self._find_all(q)
        if refit:
            before = {}
            for q in states:
                self._best_state(q)
                before[q.label] = q.best.M
            fit = self._joint_refit(states)
            for q in states:
                q.best.finish(q.edges, q.w_np)
                self._rescore(q)
                rec = {"generation": g, "best_before": before[q.label], "best_after": q.best.M, "ratios": q.best.record(q.edges)["ratios"],
                       "s02": {e.name: e.s02 for e in q.edges}, "de0": {e.name: e.de0 for e in q.edges}, "weights": q.weights.w.tolist(),
                       "at_bound": fit["at_bound"]}
                q.refits.append(rec)
                self.log(f"[{q.label}] refit at generation {g}: best {rec['best_before']:.5f} -> {rec['best_after']:.5f}; S0^2 "
                         + ", ".join(f"{k} {v:.3f}" for k, v in rec["s02"].items()) + "; E0 shift " + ", ".join(f"{k} {v:+.2f}" for k, v in rec["de0"].items())
                         + (f"; on a bound: {[n for n in fit['at_bound'] if n in rec['s02']]}" if any(n in rec["s02"] for n in fit["at_bound"]) else ""))
        for q in states:
            self._generation_sample(q, g, t_evax, rearrange, t_refresh.get(q.label))
            if g % self.save_every == 0 or g == G:
                self._checkpoint(q, g)

    def _generation_sample(self, q, g, t_evax, rearrange, t_refresh):
        S, Mv, K = self.S, self.Mv, self.K
        P, G = S.states, S.generations
        dev, n, C, gen, movable = q.dev, q.n, q.C, q.gen, q.movable
        t1 = time.time()
        tp = {"paths_all": 0.0, "evaluate_all": 0.0}
        if t_refresh is not None:
            tp["paths_all"], tp["evaluate_all"] = t_refresh
        pop, site, R, M, ms, lists = q.pop, q.site, q.R, q.M, q.ms, q.lists
        # 1. selection, crossover, replacement of bad children
        taken = []

        def cross(i, j):
            cA, cB, nt = site_crossover(pop[i], pop[j], site[i], site[j], q.site_pos, q.site_ids, q.n_ids, q.sp, cell=q.cell, gen=gen)
            taken.append(nt)
            return cA, cB

        parents, pop, kept, sel = evax_selection(pop, M, gen, cross, lambda kids, par: self._bad(q, kids, par))
        site = site[parents].clone()
        R = R[parents].clone()
        ms = [m[parents].clone() for m in ms]
        lists = [[L[p * C + c] for p in parents.tolist() for c in range(C)] for L in lists]       # a child keeps the path lists of its own parent
        if kept:                                                                                  # the bred children, from their own atoms
            idx = torch.tensor(kept, device=dev)
            sub = pop[idx].view(-1, n, 3)
            ms_c, R_c = self._evaluate(q, sub, [[L[sl * C + c] for sl in kept for c in range(C)] for L in lists])
            R[idx] = R_c
            for i in range(len(ms)):
                ms[i][idx] = ms_c[i]
        M = R @ q.weights.w
        M_sel = M.clone()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tp["selection"] = time.time() - t1
        t2 = time.time()
        # 2. every state's own move: exchanges and hops, the uniform step, the gradient steps
        trial = pop.clone()
        trial_site = site.clone()
        n_swap = n_hop = n_hop_skipped = 0
        swapped = torch.zeros(P, dtype=torch.bool, device=dev)
        hopped = torch.zeros(P, dtype=torch.bool, device=dev)
        if rearrange and q.can_exchange:
            swapped = torch.rand(P, device=dev, generator=gen) < Mv.exchange_probability
            n_swap = exchange_atoms(trial, trial_site, swapped, q.ex_a, q.ex_b, Mv.exchange_count, gen)
        if rearrange and q.can_hop:
            hopped = torch.rand(P, device=dev, generator=gen) < Mv.hops.probability
            n_hop, n_hop_skipped = hop_atoms(trial, trial_site, hopped, q.hops, Mv.hops.count, Mv.hops.jitter, gen, any_site=Mv.hops.source == "any")
        changed = (trial != pop).any(-1).view(-1, n)                                              # the atoms an exchange or a hop moved to another site
        old_flat = pop.view(-1, n, 3)
        site_flat = trial_site.view(-1, n)
        valid = torch.ones(P, dtype=torch.bool, device=dev)
        if q.box is not None and self.box_rule == "halve":
            base = q.box.project(trial.view(-1, n, 3), site_flat).view_as(trial)                   # a hopped atom lands inside the limit
            trial = uniform_step(base, S.step, movable, gen)
            for _ in range(100):                                                                  # EvAX: halve the move of a state that leaves the limit
                out = q.box.outside(trial.view(-1, n, 3), site_flat).view(P, C).any(1)
                if not bool(out.any()):
                    break
                trial[out] = 0.5 * (trial[out] + base[out])
            still = q.box.outside(trial.view(-1, n, 3), site_flat).view(P, C).any(1)
            valid &= ~still
            trial = torch.where(still[:, None, None, None], base, trial)
            flat, took = trial.view(-1, n, 3), 0
        else:
            trial = uniform_step(trial, S.step, movable, gen)
            q.objective.ms = ms                                                                   # held during the gradient steps: each state's multiple scattering before the move
            project = (lambda p_: q.box.project(p_, site_flat)) if q.box is not None else None
            if S.gradient_steps > 0:
                flat, _E, _F, took = fire_relax(q.objective, trial.view(-1, n, 3), S.gradient_steps, project=project)
            else:
                flat = project(trial.view(-1, n, 3)) if project is not None else trial.view(-1, n, 3)
                took = 0
            trial = flat.view(P, C, n, 3)
        if K.reject_trials and q.closest.active:
            valid &= ~(q.closest.worst(flat).view(P, C).amax(1) > K.closest_tolerance)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tp["gradient_steps"] = time.time() - t2
        t3 = time.time()
        # 3. the trial's own paths (found again around the moved atoms) and multiple scattering; weights and acceptance
        with torch.no_grad():
            lists_t = [list(L) for L in lists]
            n_refound = 0
            for j in torch.nonzero(changed.any(1)).flatten().tolist():
                pts = torch.cat([old_flat[j][changed[j]], flat[j][changed[j]]])
                for i, sp_ in enumerate(q.paths):
                    sel_abs = sp_.near(pts, flat[j])
                    lists_t[i][j] = sp_.refind(lists[i][j], flat[j], sel_abs)
                    n_refound += int(sel_abs.numel())
            if dev.type == "cuda":
                torch.cuda.synchronize()
            tp["paths_moved"] = time.time() - t3
            t4 = time.time()
            ms_t, R_t = self._evaluate(q, flat, lists_t)
            if S.means_from > 1 and t_evax == S.means_from:
                self.log(f"[{q.label}]   generation {g} (count {t_evax}): the running mean of the acceptance starts again (before: <|dM|> "
                         + ("none" if q.anneal.average_delta is None else f"{q.anneal.average_delta:.3e}") + f" over {q.anneal.num} moves)")
                q.anneal.restart()
            if S.weights_means_from > 1 and t_evax == S.weights_means_from:
                self.log(f"[{q.label}]   generation {g} (count {t_evax}): the running mean of the spectrum weights starts again (weights before "
                         + "/".join(f"{v:.3f}" for v in q.weights.w.tolist()) + f", {q.weights.num} moves)")
                q.weights.restart()
            all_valid = bool(valid.all())
            if S.adapt_weights and q.weights.update(R_t, R, None if all_valid else valid):
                self._set_weights(q)
            w = q.weights.w
            M, M_t = R @ w, R_t @ w
            delta = M_t - M
            rho = torch.rand(P, device=dev, generator=gen).to(delta.dtype)
            acc = q.anneal.accept(delta, t_evax, rho, None if all_valid else valid)
            rearr = swapped | hopped
            worse = delta > 0
            pop[acc] = trial[acc]
            site[acc] = trial_site[acc]
            R[acc] = R_t[acc]
            for i in range(len(ms)):
                ms[i][acc] = ms_t[i][acc]
                for p in torch.nonzero(acc).flatten().tolist():
                    lists[i][p * C:(p + 1) * C] = lists_t[i][p * C:(p + 1) * C]
            M = R @ w
            q.objective.ms = ms
            if dev.type == "cuda":
                torch.cuda.synchronize()
            tp["evaluate"] = time.time() - t4
            # 4. the records
            best = int(M.argmin())
            diversity = float(_state_distance(pop, pop, q.cell, movable).sum() / (P * (P - 1)))
            disp = (pop[best] - nearest_sites(q.site_pos, q.cell, pop[best], site[best])).norm(dim=-1)
            per = {el: float(disp[:, torch.as_tensor(q.model.symbols == el, device=dev)].pow(2).mean().sqrt()) for el in q.species_names}
            n_below, depth = q.closest.below(pop[best])
            at_box = q.box.at_limit(pop[best], site[best])[0] if q.box is not None else None
            arr = q.stats(site[best].cpu().numpy()) if (rearrange and (q.can_exchange or q.can_hop)) or g == G else None
        q.pop, q.site, q.R, q.M, q.ms, q.lists = pop, site, R, M, ms, lists
        accepted, acc_worse = int(acc.sum()), int((acc & worse).sum())
        q.counts_total["accepted_worse"] += acc_worse
        q.counts_total["children_rejected"] += sel["children_rejected"]
        q.counts_total["refused"] += int((~valid).sum())
        an = q.anneal
        rec = {"generation": g, "generation_total": t_evax, "target_acceptance": an.target(t_evax), "average_delta": an.average_delta, "rearrange": rearrange,
               "best": float(M.min()), "mean": float(M.mean()), "worst": float(M.max()), "best_state": best,
               "ratios_best": {e.name: float(v) for e, v in zip(q.edges, R[best])}, "misfits": [float(v) for v in M], "weights": w.tolist(),
               "selection": {**sel, "atoms_exchanged_per_child": (float(np.mean(taken)) if taken else 0.0), "mean_after": float(M_sel.mean()),
                             "best_after": float(M_sel.min())},
               "accepted": accepted, "accepted_worse": acc_worse, "proposed_worse": int(worse.sum()), "refused": int((~valid).sum()),
               "accepted_worse_plain": int((acc & worse & ~rearr).sum()), "accepted_worse_rearranged": int((acc & worse & rearr).sum()),
               "gradient_steps": took, "exchanges": n_swap, "exchange_states": int(swapped.sum()), "accepted_exchange_states": int((acc & swapped).sum()),
               "hops": n_hop, "hops_skipped": n_hop_skipped, "hop_states": int(hopped.sum()), "accepted_hop_states": int((acc & hopped).sum()),
               "absorbers_refound": n_refound, "diversity": diversity, "rms_from_sites_best": float(disp.pow(2).mean().sqrt()),
               "max_from_sites_best": float(disp.max()), "rms_from_sites_best_per_element": per, "pairs_below_closest_best": n_below,
               "deepest_below_closest_best": depth, "at_limit_best": at_box, "arrangement_best": arr, "seconds_parts": tp,
               "seconds": time.time() - t1 + tp["paths_all"] + tp["evaluate_all"]}
        q.history.append(rec)
        self.log(f"[{q.label}] generation {g:4d}: best {rec['best']:.5f} (R " + " / ".join(f"{v:.4f}" for v in rec["ratios_best"].values())
                 + f"), mean {rec['mean']:.5f}; {sel['bred_pairs']} pairs bred ({sel['children_rejected']} children refused); "
                 f"{accepted} of {P} moves accepted ({acc_worse} worse, target {an.target(t_evax):.3f}"
                 + (f", {rec['refused']} refused" if rec["refused"] else "") + "); weights " + "/".join(f"{v:.3f}" for v in w.tolist())
                 + f"; spread {diversity:.3f} A; best {rec['rms_from_sites_best']:.3f} A rms from its sites"
                 + (f"; {n_below} pairs below a closest distance" if n_below else "") + (f"; {100 * at_box:.1f}% at the limit" if at_box is not None else "")
                 + (f"; {n_swap} exchanges, {n_hop} hops" if (n_swap or n_hop) else "") + f"; {rec['seconds']:.1f} s")

    # ------------------------------------------------------------------------------------------------------------
    # the run
    # ------------------------------------------------------------------------------------------------------------
    def run(self):
        """Run the search; write the outputs if an output directory was given. Returns the summary (``run.json``'s content)."""
        S = self.S
        self.offset = S.generation_offset or 0
        states = [self._start_sample(inp) for inp in self.samples_in]
        an0 = states[0].anneal
        self.log(f"search: {S.states} states of {S.cells_per_state} cell(s); generations {self.offset + 1} to {self.offset + S.generations} of the "
                 f"cooling (worse moves of mean size accepted with {an0.target(self.offset + 1):.3f} at the first, {an0.target(self.offset + S.generations):.3f} "
                 f"at the last; zero from {S.froze_in}); step {S.step} A; {S.gradient_steps} gradient steps; displacement limit "
                 + (f"{self.K.max_displacement} A ({self.box_rule})" if self.K.max_displacement > 0 else "none")
                 + f"; paths found again every {S.paths_every} generations; S0^2 and E0 shifts: {self.A.mode}")
        for g in range(1, S.generations + 1):
            self._generation(g, states)
        return self._finish(states)

    def _finish(self, states):
        S, A = self.S, self.A
        P, G = S.states, S.generations
        for q in states:
            self._find_all(q)
            q.best_index = self._best_state(q)
            q.held = q.best.record(q.edges)
            q.chi_held = {e.name: c.mean(0).cpu().numpy() for e, c in zip(q.edges, q.best.chi)}
            with torch.no_grad():
                q.M_final = q.M.cpu().numpy()
                q.R_final = q.R.cpu().numpy()
                q.chi_pop = {e.name: c.cpu().numpy() for e, c in zip(q.edges, q.objective.spectra(q.pop.view(-1, q.n, 3), q.ms))}
            q.arr_final = q.stats(q.site[q.best_index].cpu().numpy())
            q.arr_pop = [q.stats(q.site[p].cpu().numpy()) for p in range(P)]
            self.log(f"[{q.label}] best state {q.best_index}, all paths found again: misfit {q.best.M:.5f} (R " + " / ".join(f"{r:.4f}" for r in q.best.ratios)
                     + f"); population misfits {q.M_final.min():.5f} - {q.M_final.max():.5f}")
        fit, intervals = None, None
        if self.fit_mode:
            fit = self._joint_refit(states)
            for q in states:
                q.best.finish(q.edges, q.w_np)
            groups = [q.best.group(q.edges, q.w_np) for q in states]
            floors = []
            for grp in groups:
                with torch.no_grad():
                    m = sum(w * float(cmp.residual(chi)[0]) for cmp, chi, w in grp)
                floors.append(max(float(A.noise_floor), m))
            try:
                intervals = shared_intervals(groups, W=likelihood_weights(groups, floors))
            except Exception as err:                                                             # pragma: no cover
                intervals = {"error": f"{type(err).__name__}: {err}"}
        for q in states:
            q.refit = q.best.record(q.edges)
            q.chi_refit = {e.name: c.mean(0).cpu().numpy() for e, c in zip(q.edges, q.best.chi)}
            self.log(f"[{q.label}] " + ("S0^2 and E0 shifts refitted on the best state" if self.fit_mode else "final (S0^2 and E0 shifts held)")
                     + f": misfit {q.best.M:.5f}; S0^2 " + ", ".join(f"{k} {v:.3f}" for k, v in q.refit["s02"].items())
                     + "; E0 shift " + ", ".join(f"{k} {v:+.2f}" for k, v in q.refit["de0"].items()))
        shared = {"s02": {}, "de0": {}}
        for q in states:
            for e in q.edges:
                shared["s02"].setdefault(e.name, e.s02)
                shared["de0"].setdefault(e.name, e.de0)
        summary = {"method": METHOD, "format": FORMAT, "samples": [q.label for q in states], "amplitude_mode": A.mode, "shared": shared,
                   "at_bound": fit["at_bound"] if fit else [], "intervals": intervals,
                   "intervals_method": (intervals or {}).get("method") if isinstance(intervals, dict) else None,
                   "misfits": {q.label: {"best": q.refit["misfit"], "ratios": q.refit["ratios"]} for q in states},
                   "generations_done": self.offset + G, "seconds": time.time() - self.t0}
        if self.output is not None:
            self._write(states, summary)
        return summary

    def _write(self, states, summary):
        S, A, Mv, K = self.S, self.A, self.Mv, self.K
        G = S.generations
        out = self.output
        out.mkdir(parents=True, exist_ok=True)
        for q in states:
            o = out / q.label
            o.mkdir(parents=True, exist_ok=True)
            start_pos, final_pos = q.start_pos.cpu().numpy(), q.best.pos.cpu().numpy()
            keys = sorted({k for a in q.arr_pop for k in a})
            spread = {}
            for k in keys:
                v = np.array([a[k] for a in q.arr_pop if k in a], float)
                v = v[np.isfinite(v)]
                if v.size:
                    spread[k] = {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max())}
            report = {"method": METHOD, "format": FORMAT, "label": q.label, "structure": q.model.describe(), "species": q.species_names,
                      "edges": [e.name for e in q.edges], "absorbers": {e.name: e.absorber for e in q.edges},
                      "edge_descriptions": [e.describe() for e in q.edges],
                      "settings": {"search": S.model_dump(), "amplitude": A.model_dump(), "moves": Mv.model_dump(), "constraints": K.model_dump(),
                                   "box_rule": self.box_rule, "seed": self.seed},
                      "closest_distance": q.closest.as_dict(), "max_displacement": K.max_displacement,
                      "at_limit_final": q.box.at_limit(q.best.pos, q.site[q.best_index])[0] if q.box is not None else None,
                      "scale": q.scale, "scale_source": q.scale_source, "start": q.start, "start_kind": "continued" if S.continue_from else "start structure",
                      "from_population": q.from_population, "best_state": q.best_index, "final_held": q.held, "final_refit": q.refit,
                      "population_misfits": q.M_final.tolist(), "population_ratios": q.R_final.tolist(),
                      "arrangement": {"start": q.arr_start, "final_best": q.arr_final, "population": q.arr_pop, "population_spread": spread},
                      "intervals": summary["intervals"], "search": self.search_state(q, G), "seconds": time.time() - self.t0}
            write_json(o / "report.json", report)
            cell = q.model.cell if q.model.cell is not None else np.full((3, 3), np.nan)
            sites = q.model.sites
            np.savez_compressed(o / "state.npz", positions=final_pos, start=start_pos, population=q.pop.cpu().numpy(), site_of=q.site.cpu().numpy(),
                                misfits=q.M_final, ratios=q.R_final, edges=np.array([e.name for e in q.edges]), cell=cell, symbols=q.model.symbols,
                                ideal=q.base.cpu().numpy(), sites=sites.positions, site_ids=sites.ids, start_site_of=sites.start_site_of,
                                best_state=q.best_index)
            write_json(o / "history.json", {"generations": q.history, "refits": q.refits, "search": self.search_state(q, G)})

            def models(chis, rec):
                outm = {}
                for e in q.edges:
                    c = torch.as_tensor(chis[e.name], device=e.device, dtype=torch.float32).reshape(-1, e.NK)
                    with torch.no_grad():
                        outm[e.name] = e.model(c, s02=float(rec["s02"][e.name]), de0=float(rec["de0"][e.name]) - e.e0).cpu().numpy().squeeze()
                return outm
            arrays = {"k": q.edges[0].k}
            for e in q.edges:
                arrays[f"data_{e.name}"] = e.data
                arrays[f"window_{e.name}"] = np.array([e.window[0], e.window[1], e.dk, e.kweight, e.r_range[0], e.r_range[1]], float)
            for tag, chis, rec in (("start", q.chi_start, q.start), ("final", q.chi_held, q.held), ("population", q.chi_pop, q.held),
                                   ("refit", q.chi_refit, q.refit)):
                for name, v in models(chis, rec).items():
                    arrays[f"model_{tag}_{name}"] = v
            for e in q.edges:
                arrays[f"calc_refit_{e.name}"] = q.chi_refit[e.name]
            np.savez_compressed(o / "spectra.npz", **arrays)
        if self.resolved is not None:
            from ..config_tools import to_toml
            (out / "config.resolved.toml").write_text(to_toml(self.resolved))
        write_json(out / "run.json", summary)
        self.log(f"written {out}/ ({time.time() - self.t0:.0f} s)")
