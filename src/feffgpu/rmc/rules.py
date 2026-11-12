"""EvAX's rules of the population search, as small functions and classes.

The rules were read from the source of EvAX 6.16 (Timoshenko and Kuzmin): ``selectionT``, ``crossover`` and
``end_ea`` (selection, crossover, replacement of a bad child), ``mutation`` with ``adjust_acceptance`` (the
acceptance of a worse move and its cooling), and ``adjust_spectra_weights`` (the weights of the spectra). The search
(``feffgpu.rmc.search``) applies them once per generation. Where this package differs from EvAX, the difference is
listed in ``docs/rmc/evax_differences.md``.
"""
from __future__ import annotations
import math
import torch
from .constraints import nearest_sites

__all__ = ["EdgeWeights", "Annealing", "evax_selection", "site_crossover"]


class EdgeWeights:
    """EvAX's weights of the spectra of one sample (``adjust_spectra_weights``).

    ``w_e = (1 / a_e) / sum_f (1 / a_f)``, where ``a_e`` is the running mean over all proposed moves of
    ``|R_e(trial) - R_e(state)|``: a spectrum whose misfit changes less weighs more, so that every spectrum changes the
    total misfit by a similar amount. Start: equal weights. One spectrum keeps the weight 1.

    Parameters
    ----------
    n : int
        Number of spectra.
    device : str
    weights, average_delta, num : optional
        A saved state (``state()``) to continue from.
    """

    def __init__(self, n, device="cpu", weights=None, average_delta=None, num=0):
        self.w = (torch.full((n,), 1.0 / n, dtype=torch.float64, device=device) if weights is None
                  else torch.as_tensor(weights, dtype=torch.float64, device=device))
        self.avg = None if average_delta is None else torch.as_tensor(average_delta, dtype=torch.float64, device=device)
        self.num = int(num)

    def update(self, R_trial, R_current, valid=None):
        """Take the moves of one generation into the running mean and recompute the weights.

        Parameters
        ----------
        R_trial, R_current : torch.Tensor, shape (S, n)
            The misfit of every spectrum for the S trial states and for the states they came from.
        valid : torch.Tensor of bool, shape (S,), optional
            The moves that count (a move refused by a constraint before its spectrum was compared does not).

        Returns
        -------
        bool
            True if the weights changed.
        """
        d = (R_trial - R_current).abs().to(torch.float64)
        if valid is not None:
            d = d[valid]
        S = d.shape[0]
        if S == 0:
            return False
        self.avg = d.mean(0) + 1e-30 if self.avg is None else (self.avg * self.num + d.sum(0)) / (self.num + S)
        self.num += S
        inv = 1.0 / self.avg
        new = inv / inv.sum()
        changed = bool((new - self.w).abs().max() > 1e-6)
        self.w = new
        return changed

    def restart(self):
        """The running mean starts again from zero moves; the weights stay until the next update."""
        self.avg, self.num = None, 0

    def state(self):
        """A JSON-serializable dict that ``EdgeWeights(n, weights=..., average_delta=..., num=...)`` continues from."""
        return {"weights": self.w.tolist(), "average_delta": None if self.avg is None else self.avg.tolist(), "num": self.num}


class Annealing:
    """EvAX's acceptance of a proposed move, cooled linearly to zero (``mutation`` with ``adjust_acceptance``).

    A move that does not raise the misfit is accepted. A worse move is accepted when
    ``exp(logp x dM / <|dM|>) >= rho``, with ``logp = log(acceptance x (1 - t / froze_in))``, ``t`` EvAX's count of
    generations (1 at the first), ``<|dM|>`` the running mean of ``|dM|`` over the proposed moves and ``rho`` uniform in
    [0, 1). A worse move of mean size is therefore accepted with the probability ``acceptance x (1 - t / froze_in)``.
    From ``t = froze_in`` on (or with ``acceptance <= 0``), only moves that do not raise the misfit are accepted.

    Parameters
    ----------
    acceptance : float
        EvAX's ``Acceptance_rate``.
    froze : int
        EvAX's ``Froze_in``: the generation count at which worse moves stop being accepted.
    average_delta, num : optional
        A saved state to continue from.
    """

    def __init__(self, acceptance=0.8, froze=1500, average_delta=None, num=0):
        self.acceptance, self.froze = float(acceptance), int(froze)
        self.average_delta, self.num = (None if average_delta is None else float(average_delta)), int(num)

    def target(self, t):
        """The probability of accepting a worse move of mean size at generation count ``t``."""
        return 0.0 if (self.acceptance <= 0 or t >= self.froze) else self.acceptance * (1.0 - t / self.froze)

    def logp(self, t):
        tg = self.target(t)
        return -math.inf if tg <= 0 else math.log(tg)

    def accept(self, delta, t, rho, valid=None):
        """Which proposed moves are accepted.

        Parameters
        ----------
        delta : torch.Tensor, shape (S,)
            The change of the misfit of every proposed move.
        t : int
            The generation count.
        rho : torch.Tensor, shape (S,)
            Uniform random numbers.
        valid : torch.Tensor of bool, shape (S,), optional
            The moves that count; the others are not accepted and stay out of the running mean.

        Returns
        -------
        torch.Tensor of bool, shape (S,)
            The running mean takes this generation's moves before they are judged.
        """
        d = delta if valid is None else delta[valid]
        S = d.shape[0]
        if S:
            a = float(d.abs().sum())
            self.average_delta = a / S + 1e-30 if self.average_delta is None else (self.average_delta * self.num + a) / (self.num + S)
            self.num += S
        lp = self.logp(t)
        if lp == -math.inf or self.average_delta is None:
            acc = delta <= 0
        else:
            acc = (delta <= 0) | (torch.exp(lp * delta / self.average_delta) >= rho)
        return acc if valid is None else acc & valid

    def restart(self):
        """The running mean of ``|dM|`` starts again from zero moves (the generation count is not changed)."""
        self.average_delta, self.num = None, 0

    def state(self):
        """A JSON-serializable dict that ``Annealing(acceptance, froze, average_delta, num)`` continues from."""
        return {"acceptance": self.acceptance, "froze_in": self.froze, "average_delta": self.average_delta, "num": self.num}


def evax_selection(pop, M, gen, crossover, bad, picks=None, rho=None):
    """EvAX's selection, crossover and replacement (``selectionT``, ``crossover``, ``end_ea``).

    For every pair ``iv`` of slots ``(2 iv, 2 iv + 1)``: the first parent is the better of two random states (the first
    pick on a tie), the second parent the better of two more; the two slots receive copies of the two parents. With
    ``rho > 0.5`` and two different parents the pair breeds: ``crossover(parent 1, parent 2)`` gives the two children.
    A child that ``bad`` flags (a constraint violated) is replaced by its unbred selected parent, the copy already in
    the slot (not the slot's previous occupant). All picks read the population before the selection. No state is
    protected.

    Parameters
    ----------
    pop : torch.Tensor, shape (P, ...)
        The states (P even).
    M : torch.Tensor, shape (P,)
        Their misfits.
    gen : torch.Generator or None
    crossover : callable
        ``crossover(i, j) -> (child_i, child_j)``: the children of states ``i`` and ``j``, each shaped like ``pop[0]``.
    bad : callable
        ``bad(children (m, ...), parents (list of m state indices)) -> (m,) bool``; ``parents[k]`` is the state whose
        copy the child replaces (the child keeps that state's arrangement of atoms on sites).
    picks : array (4, P/2), optional
        The tournament picks (tests).
    rho : array (P/2,), optional
        The breeding draws (tests).

    Returns
    -------
    parents : torch.Tensor (P,)
        The state copied into every slot.
    new : torch.Tensor (P, ...)
        The new population.
    kept : list of int
        The slots that hold a bred child.
    stats : dict
        ``bred_pairs``, ``children``, ``children_rejected``, ``distinct_parents``.
    """
    P = pop.shape[0]
    dev = pop.device
    if P % 2:
        raise ValueError("an even number of states is needed (EvAX breeds in pairs)")
    r = torch.randint(P, (4, P // 2), device=dev, generator=gen) if picks is None else torch.as_tensor(picks, device=dev)
    first = torch.where(M[r[1]] < M[r[0]], r[1], r[0])
    second = torch.where(M[r[3]] < M[r[2]], r[3], r[2])
    rho = torch.rand(P // 2, device=dev, generator=gen) if rho is None else torch.as_tensor(rho, device=dev)
    breed = (rho > 0.5) & (first != second)
    parents = torch.stack([first, second], 1).reshape(-1)
    new = pop[parents].clone()
    slots, kids = [], []
    for iv in torch.nonzero(breed).flatten().tolist():
        a, b = crossover(int(first[iv]), int(second[iv]))
        slots += [2 * iv, 2 * iv + 1]
        kids += [a, b]
    kept = []
    if kids:
        rejected = bad(torch.stack(kids), [int(parents[sl]) for sl in slots])
        for k, sl in enumerate(slots):
            if not bool(rejected[k]):
                new[sl] = kids[k]
                kept.append(sl)
    return parents, new, kept, {"bred_pairs": int(breed.sum()), "children": len(slots), "children_rejected": len(slots) - len(kept),
                                "distinct_parents": int(torch.unique(parents).numel())}


def site_crossover(A, B, site_A, site_B, site_positions, site_ids, n_ids, species, cell=None, gen=None, mask=None):
    """EvAX's crossover of two states, by lattice site.

    EvAX: for every atom, with probability 0.5, child 1 takes the displacement that atom has in parent 2 and child 2
    the one it has in parent 1 (one draw for both children). With exchanges and hops the same atom can sit on
    different sites in the two parents, so the draw is made per site: on a drawn site, child 1 (the arrangement of A)
    gives A's atom on that site the displacement of B's atom on that site, and child 2 the reverse. A site is
    exchanged only if it holds one atom in both parents and the two atoms are of the same element; otherwise both
    children keep their parent's atom as it is. With equal arrangements this is EvAX's rule atom by atom. The
    displacements of a child come from states inside the displacement limit, so the child is inside it too.

    Parameters
    ----------
    A, B : torch.Tensor, shape (C, n, 3)
        The two parents.
    site_A, site_B : torch.Tensor of int, shape (C, n)
        Their sites (indices into the site table).
    site_positions : torch.Tensor, shape (m, 3)
    site_ids : torch.Tensor of int, shape (m,)
        The identifier of every site; ``n_ids`` their number.
    species : torch.Tensor of int, shape (n,)
        The element code of every atom.
    cell : None, float or torch.Tensor (3, 3)
    gen : torch.Generator, optional
    mask : torch.Tensor of bool, shape (C, n_ids), optional
        The drawn sites (default: random with probability 0.5).

    Returns
    -------
    child_A, child_B : torch.Tensor, shape (C, n, 3)
    taken : int
        The number of atoms of child A that took B's displacement.
    """
    C, n, _ = A.shape
    dev = A.device
    nsite = int(n_ids)
    idA, idB = site_ids[site_A], site_ids[site_B]
    sA = nearest_sites(site_positions, cell, A, site_A)
    sB = nearest_sites(site_positions, cell, B, site_B)
    if mask is None:
        mask = torch.rand(C, nsite, device=dev, generator=gen) > 0.5
    atoms = torch.arange(n, device=dev)[None].expand(C, n)

    def table(ids):
        """per cell: the atom on every site identifier (-1: none or several), and how many atoms share each atom's site"""
        slot = torch.where(ids >= 0, ids, torch.full_like(ids, nsite))
        cnt = torch.zeros(C, nsite + 1, dtype=torch.long, device=dev).scatter_add_(1, slot, torch.ones_like(slot))
        occ = torch.full((C, nsite + 1), -1, dtype=torch.long, device=dev).scatter_(1, slot, atoms)
        occ[cnt != 1] = -1
        occ[:, nsite] = -1
        return occ, torch.gather(cnt, 1, slot)

    occA, ownA = table(idA)
    occB, ownB = table(idB)

    def take(X, idX, sX, ownX, Y, occY, sY):
        ids = idX.clamp(min=0)
        j = torch.gather(occY, 1, ids)                                                       # Y's atom on the atom's site
        jj = j.clamp(min=0)
        ok = (idX >= 0) & (ownX == 1) & (j >= 0) & torch.gather(mask, 1, ids) & (species[jj] == species[None].expand(C, n))
        dY = torch.gather(Y - sY, 1, jj[..., None].expand(C, n, 3))
        return torch.where(ok[..., None], sX + dY, X), int(ok.sum())

    child_A, n_taken = take(A, idA, sA, ownA, B, occB, sB)
    child_B, _ = take(B, idB, sB, ownB, A, occA, sA)
    return child_A, child_B, n_taken
