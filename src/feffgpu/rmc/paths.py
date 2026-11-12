"""The multiple-scattering path lists of every state of a population, each state's paths from its own atoms.

A path list holds atom indices, not coordinates: ``{number of legs: {atoms (B, legs - 1), absorber (B,), ipots
(B, legs), deg (B,)}}`` with indices in the edge's atom order (ignored elements left out). The signal of a list is
always computed from the current atoms, so between two searches of the path finder the multiple scattering still
follows the atoms; only the choice of paths (the importance filter) is held. A list is never changed in place
(states share lists after a selection); ``refind`` returns a new one.
"""
from __future__ import annotations
import torch
from ..cell import wrap

__all__ = ["StatePaths"]


class StatePaths:
    """The multiple-scattering path lists of one edge, one list per configuration.

    Parameters
    ----------
    edge : feffgpu.exafs.edge.EdgeModel
    margin : float
        Added to the path cutoff in ``near`` (A), for the small moves made since a list was found.
    """

    def __init__(self, edge, margin=0.5):
        self.e, self.ms = edge, edge.ms
        self.keep, self.abs_keep, self.A, self.NK = edge.keep, edge.abs_keep, edge.nA, edge.NK
        self.nk = int(edge.keep.numel())
        self.ipot = edge.ipot[edge.keep]
        self.device = edge.device
        self.margin = float(margin)
        if self.ms is not None:
            self.finder, self.box, self.rmax = self.ms.full.finder, self.ms.box, float(self.ms.full.finder.rmax)

    @torch.no_grad()
    def find(self, pos, absorbers=None):
        """The list of one configuration ``pos`` (n, 3): FEFF's path finder for every absorber (or for the given ones,
        indices in the edge's atom order), the paths of three and four legs after the importance filter."""
        if self.ms is None:
            return {}
        p = torch.as_tensor(pos, device=self.device, dtype=torch.float32)[self.keep]
        absorbers = self.abs_keep.tolist() if absorbers is None else [int(a) for a in torch.as_tensor(absorbers).tolist()]
        parts = {}
        for a in absorbers:
            kept, _ = self.finder.enumerate(p, self.box, a, self.ipot)
            for nleg in (3, 4):
                if nleg not in kept or kept[nleg][0].shape[0] == 0:
                    continue
                coords, ipots, deg, xp, atoms = kept[nleg]
                g = parts.setdefault(nleg, ([], [], [], []))
                g[0].append(atoms)
                g[1].append(torch.full((atoms.shape[0],), a, device=self.device, dtype=torch.long))
                g[2].append(ipots)
                g[3].append(deg)
        return {nleg: dict(atoms=torch.cat(g[0]), absorber=torch.cat(g[1]), ipots=torch.cat(g[2]), deg=torch.cat(g[3])) for nleg, g in parts.items()}

    @torch.no_grad()
    def refind(self, L, pos, absorbers):
        """The list ``L`` with the paths of these absorbers found again on ``pos`` (n, 3): a new list."""
        absorbers = torch.as_tensor(absorbers, device=self.device, dtype=torch.long)
        if self.ms is None or absorbers.numel() == 0:
            return L
        new = self.find(pos, absorbers)
        out = {}
        for nleg in sorted(set(L) | set(new)):
            parts = []
            if nleg in L:
                stay = ~torch.isin(L[nleg]["absorber"], absorbers)
                parts.append({k: v[stay] for k, v in L[nleg].items()})
            if nleg in new:
                parts.append(new[nleg])
            out[nleg] = {k: torch.cat([q[k] for q in parts]) for k in parts[0]}
        return out

    def near(self, points, pos):
        """The absorbers (indices in the edge's atom order) within the path cutoff plus the margin of any of the points
        (m, 3) in the configuration ``pos`` (n, 3). The paths of an absorber depend only on the atoms within the cutoff,
        so these are all the absorbers whose list can change when atoms leave or reach the points."""
        if self.ms is None or len(points) == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        pa = torch.as_tensor(pos, device=self.device, dtype=torch.float32)[self.keep][self.abs_keep]
        d = wrap(pa[:, None, :] - torch.as_tensor(points, device=self.device, dtype=torch.float32)[None], self.box).norm(dim=-1)
        return self.abs_keep[(d <= self.rmax + self.margin).any(1)]

    @staticmethod
    def count(L):
        """``{number of legs: number of paths}`` of a list."""
        return {nleg: int(v["atoms"].shape[0]) for nleg, v in L.items()}

    @torch.no_grad()
    def signal(self, lists, pos):
        """The multiple-scattering chi (m, NK) of configurations ``pos`` (m, n, 3), each from its own list and its own
        atoms, averaged over the edge's absorbers (what ``EdgeModel.ms_pass`` gives for the same paths)."""
        pos = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        m = pos.shape[0]
        if self.ms is None:
            return torch.zeros(m, self.NK, device=self.device)
        p, nk, ms = pos[:, self.keep], self.nk, self.ms
        cat = {}
        for nleg in (3, 4):
            parts = [(j, L[nleg]) for j, L in enumerate(lists) if nleg in L and L[nleg]["atoms"].shape[0]]
            if parts:
                cat[nleg] = dict(atoms=torch.cat([v["atoms"] + j * nk for j, v in parts]), absorber=torch.cat([v["absorber"] + j * nk for j, v in parts]),
                                 ipots=torch.cat([v["ipots"] for _, v in parts]), deg=torch.cat([v["deg"] for _, v in parts]))
        ms.lists, ms.C, ms.n, ms.A, ms.n_abs = cat, m, nk, self.A, m * self.A
        ms.abs_atoms = (self.abs_keep[None, :] + nk * torch.arange(m, device=self.device)[:, None]).reshape(-1)
        ms.row_of = torch.full((m * nk,), -1, device=self.device, dtype=torch.long)
        ms.row_of[ms.abs_atoms] = torch.arange(m * self.A, device=self.device)
        ms.counts = {nleg: int(v["atoms"].shape[0]) for nleg, v in cat.items()}
        rows = ms.full_pass(p.reshape(-1, 3))
        rows = rows.real if rows.is_complex() else rows
        return rows.view(m, self.A, -1).mean(1).float()
