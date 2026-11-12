"""FEFF's path finder for one absorber of an arbitrary configuration, on the GPU (feff85exafs PATH/*.f).

What FEFF does (paths.f, pathsd.f, outcrt.f, mcritk.f, prcrit.f, timrep.f) with the heap and keep limits at zero,
the EXAFS default: enumerate every sequence of up to nleg-1 scatterers whose total length (out and back) is at
most 2 rmax, the first and last atoms not the absorber, no atom repeated consecutively, the absorber allowed in the
middle; merge each sequence with its time reverse (same signal by reciprocity) into one path of degeneracy 2
(1 if palindromic); score every path by its plane-wave importance, the integral over the real-axis energies of
|f(beta_1)|/rho_1 ... |f(beta_m)|/rho_m x max(|cos beta_0|, 0.3)/rho_0 x exp(-R_total/lambda), with |f(beta)|
tabulated on a cos(beta) grid of 0.025 from the j = l + 1/2 phase shifts; keep paths whose degeneracy-weighted
importance is at least critpw percent of the shortest single-scattering path's. FEFF also merges paths whose
standard-frame coordinates round to the same 0.001 A within a greedy 0.001 A window of total length (accidental
degeneracy); that is done here for single scattering, where it occurs in disordered configurations, and not for
multiple scattering, where it does not. Output: sequences grouped by number of legs, ready for GPUPathStage.
"""
from __future__ import annotations
import numpy as np
import torch
from .cell import as_cell, wrap
from scipy.special import eval_legendre
from .pad import PhaseData, read_phase_pad, BOHR

NBETA = 40


class GPUPathFinder:
    def __init__(self, phase: PhaseData | str, device="cuda", rmax=9.4, nlegmax=4, critpw=0.1, merge_degenerate=True):
        P = read_phase_pad(phase) if isinstance(phase, str) else phase
        self.P, self.device = P, torch.device(device)
        self.rmax, self.nlegmax, self.critpw = float(rmax), int(nlegmax), float(critpw)
        self.merge_degenerate = merge_degenerate
        E = P.ne1
        lmax = P.lmax[:E]
        cosb = 0.025 * np.arange(-NBETA, NBETA + 1)
        cosb[0], cosb[-1] = -1.0, 1.0
        fb = np.zeros((E, P.nph + 1, 2 * NBETA + 1))
        for iph in range(P.nph + 1):
            for ie in range(E):
                f = np.zeros(len(cosb), complex)
                for l in range(lmax[ie, iph] + 1):
                    tl = (np.exp(2j * P.phase(ie, -l, iph)) - 1) / 2j            # prcrit: ph4(ie, -il, ...)
                    f += tl * (2 * l + 1) * eval_legendre(l, cosb)
                fb[ie, iph] = np.abs(f)
        self.fbeta = torch.tensor(fb, device=self.device, dtype=torch.float32)       # (E, nph+1, 81)
        ck = P.ck[:E]
        self.cksp = torch.tensor(ck.real / BOHR, device=self.device, dtype=torch.float32)          # 1/A
        xlam = np.where(np.abs(ck.imag) > 1e-16, 1 / ck.imag, 1e10) * BOHR                        # A
        self.xlam = torch.tensor(xlam, device=self.device, dtype=torch.float32)
        self.ik0 = P.ik0 - 1
        # trapezoid weights of strap over cksp[ik0:]
        x = ck.real[self.ik0:] / BOHR
        w = np.zeros(len(x))
        w[0] = x[1] - x[0]
        w[1:-1] = x[2:] - x[:-2]
        w[-1] = x[-1] - x[-2]
        self.trap = torch.tensor(w / 2, device=self.device, dtype=torch.float32)

    # -- importance -------------------------------------------------------------------------------------------
    def _indbet(self, cosb):
        tmp = cosb.abs()
        n = torch.floor(tmp / 0.025)
        n = torch.where(tmp - n * 0.025 > 0.0125, n + 1, n)
        n = torch.where(cosb < 0, -n, n)
        return (n.long() + NBETA).clamp(0, 2 * NBETA)

    def importance(self, pts, ipots):
        """pts (B, m+1, 3): the path atoms after the absorber (scatterers 1..m) with the absorber at index 0 and
        implicitly closing back to it; ipots (B, m+1) with the absorber 0. Returns xport (B,)."""
        B, m1, _ = pts.shape
        m = m1 - 1
        closed = torch.cat([pts, pts[:, :1]], 1)                                    # 0, 1..m, 0
        legs = closed[:, 1:] - closed[:, :-1]                                       # (B, m+1, 3): leg j = atom j-1 -> j
        ri = torch.linalg.norm(legs, dim=-1)                                        # (B, m+1): leg 1..m+1
        rtot = ri.sum(1)
        # cos(beta) at scatterers 1..m: between incoming leg j and outgoing leg j+1; at the absorber: last leg -> first leg
        u = legs / ri.clamp_min(1e-12)[..., None]
        cos_sc = (u[:, :-1] * u[:, 1:]).sum(-1)                                     # (B, m): atoms 1..m
        cos_abs = (u[:, -1] * u[:, 0]).sum(-1)                                      # (B,)
        E = self.fbeta.shape[0]
        ies = torch.arange(self.ik0, E, device=self.device)
        rho0 = ri[:, -1][:, None] * self.cksp[None, ies]                            # (B, Eu)
        crit = torch.maximum(cos_abs.abs(), torch.full_like(cos_abs, 0.3))[:, None] / rho0
        ib = self._indbet(cos_sc)                                                   # (B, m)
        for j in range(m):
            f = self.fbeta[ies][:, ipots[:, j + 1], ib[:, j]].T                      # (B, Eu)
            crit = crit * f / (ri[:, j][:, None] * self.cksp[None, ies])
        crit = crit * torch.exp(-rtot[:, None] / self.xlam[None, ies])
        return (crit.abs() * self.trap[None]).sum(1)

    def _merge_single_scattering(self, out, dev):
        """FEFF's accidental-degeneracy merge as it applies to single scattering (pathsd.f, phash.f): all candidate
        paths are sorted by total length and cut into greedy windows of width 1e-3 A; within a window, paths whose
        standard-frame coordinates round to the same 0.001 A are one path with summed degeneracy. For a single
        scatterer that means the same 0.001 A bin of its distance, and the window test. The representative FEFF
        keeps depends on its unstable index sort; the shorter path is kept here."""
        eps3 = 1.0e-3
        rtot = []
        for nleg, (coords, ipots, deg, xp, atoms) in out.items():
            closed = torch.cat([coords, coords[:, :1]], 1)
            rtot.append(torch.linalg.norm(closed[:, 1:] - closed[:, :-1], dim=-1).sum(1))
        allr = torch.cat(rtot).cpu().numpy().astype(np.float32)
        order = np.argsort(allr, kind="stable")
        sr = allr[order]
        window = np.zeros(len(sr), dtype=np.int64)
        i, w = 0, 0
        while i < len(sr):
            j = int(np.searchsorted(sr, np.float32(sr[i] + eps3), side="left"))
            j = max(j, i + 1)
            window[i:j] = w
            w += 1
            i = j
        win_of = np.empty(len(sr), dtype=np.int64)
        win_of[order] = window
        n2 = out[2][0].shape[0]
        win2 = win_of[:n2]
        coords, ipots, deg, xp, atoms = out[2]
        d = coords[:, 1].norm(dim=-1).cpu().numpy()
        key = np.round(d * 1000).astype(np.int64)
        ip = ipots[:, 1].cpu().numpy()
        groups = {}
        for k in np.argsort(d, kind="stable"):
            groups.setdefault((int(key[k]), int(ip[k]), int(win2[k])), []).append(int(k))
        rep = np.array([g[0] for g in groups.values()])
        cnt = np.array([len(g) for g in groups.values()], dtype=np.float32)
        rep_t = torch.tensor(rep, device=dev)
        out[2] = (coords[rep_t], ipots[rep_t], torch.tensor(cnt, device=dev), xp[rep_t], atoms[rep_t])
        return out

    @staticmethod
    def degenerate_total(coords, deg):
        """FEFF's degeneracy merge for multiple-scattering paths (pathsd.f, phash.f): paths with the same internal
        geometry, leg lengths and scattering angles to 1e-3 in either time direction (mirror images alike), are one
        path whose degeneracy is the sum; FEFF then applies the importance filter to that sum and evaluates one
        representative. Here every member stays in the list with its own degeneracy (the same sum for exactly
        degenerate paths) and the filter uses the group total. Returns the group total per path, (B,)."""
        closed = torch.cat([coords, coords[:, :1]], 1)                                   # (B, nleg+1, 3)
        legs = closed[:, 1:] - closed[:, :-1]                                             # (B, nleg, 3): leg j = atom j-1 -> j
        ri = torch.linalg.norm(legs, dim=-1)
        u = legs / ri.clamp_min(1e-12)[..., None]
        cosb = (u * torch.roll(u, -1, 1)).sum(-1)                                        # angle at atom j (last: absorber)
        nleg = ri.shape[1]
        fwd = torch.cat([ri, cosb], 1)
        rev = torch.cat([ri.flip(1), cosb[:, :nleg - 1].flip(1), cosb[:, nleg - 1:]], 1)
        kf, kr = torch.round(fwd * 1000).long(), torch.round(rev * 1000).long()
        # canonical orientation: the lexicographically smaller key
        less = torch.zeros(kf.shape[0], dtype=torch.bool, device=kf.device)
        undecided = torch.ones_like(less)
        for j in range(kf.shape[1]):
            less |= undecided & (kr[:, j] < kf[:, j])
            undecided &= kr[:, j] == kf[:, j]
        key = torch.where(less[:, None], kr, kf)
        _, inverse = torch.unique(key, dim=0, return_inverse=True)
        total = torch.zeros(int(inverse.max()) + 1 if len(inverse) else 0, device=deg.device, dtype=deg.dtype).index_add_(0, inverse, deg)
        return total[inverse]

    # -- enumeration ------------------------------------------------------------------------------------------
    def enumerate(self, pos, box, absorber, ipot, chunk=256):
        """pos (n, 3) A, the periodic cell (a cubic length, a 3 x 3 cell or None), absorber index, ipot (n,) potential index per atom
        (the absorber's own potential is 0 when it is the absorber). Returns {nleg: (coords (B, nleg, 3) relative
        to the absorber, ipots (B, nleg), deg (B,), xport (B,), atoms (B, nleg-1) indices)} after the importance
        filter, and the reference importance."""
        dev = self.device
        pos = torch.as_tensor(pos, device=dev, dtype=torch.float32)
        ipot = torch.as_tensor(ipot, device=dev)
        rel = wrap(pos - pos[absorber], as_cell(box, dev))
        d0 = torch.linalg.norm(rel, dim=-1)
        two_r = 2 * self.rmax
        cl = torch.nonzero((d0 <= self.rmax) & (torch.arange(len(pos), device=dev) != absorber)).squeeze(1)   # scatterer candidates
        inter = torch.cat([cl, torch.tensor([absorber], device=dev)])                                      # intermediates may be the absorber
        rc = rel[cl]                                                                                        # (nc, 3)
        ri_ = rel[inter]                                                                                    # (ni, 3)
        dc0 = d0[cl]
        out = {}
        ipot_of = lambda idx: torch.where(idx == absorber, torch.zeros_like(idx), ipot[idx])

        def finish(nleg, atoms, deg):
            coords = torch.cat([torch.zeros(len(atoms), 1, 3, device=dev), rel[atoms]], 1)
            ipots = torch.cat([torch.zeros(len(atoms), 1, dtype=torch.long, device=dev), ipot_of(atoms)], 1)
            xp = torch.cat([self.importance(coords[s:s + 65536], ipots[s:s + 65536]) for s in range(0, len(atoms), 65536)]) if len(atoms) else torch.zeros(0, device=dev)
            return coords, ipots, deg, xp, atoms

        # 2 legs
        atoms2 = cl[:, None]
        deg2 = torch.ones(len(cl), device=dev)
        out[2] = finish(2, atoms2, deg2)
        ref = out[2][3][torch.argmin(dc0)] if len(cl) else torch.tensor(1.0, device=dev)      # shortest SS path, deg 1
        # 3 legs: (i, j), i < j (time-reversed pair merged, deg 2)
        if self.nlegmax >= 3 and len(cl) > 1:
            dij = torch.cdist(rc, rc)
            tot = dc0[:, None] + dij + dc0[None, :]
            ii, jj = torch.nonzero((tot <= two_r) & (torch.arange(len(cl), device=dev)[:, None] < torch.arange(len(cl), device=dev)[None, :]), as_tuple=True)
            atoms3 = torch.stack([cl[ii], cl[jj]], 1)
            out[3] = finish(3, atoms3, torch.full((len(ii),), 2.0, device=dev))
        # 4 legs: (i, j, l) with j any intermediate (incl. the absorber), i != j, j != l; canonical i <= l (lexicographic
        # on (i, l) for the pair with its reverse (l, j, i)); deg 2 unless i == l
        if self.nlegmax >= 4 and len(cl) > 1:
            d_ci = torch.cdist(rc, ri_)                                                         # (nc, ni): scatterer -> intermediate
            parts = []
            for s in range(0, len(cl), chunk):
                i_idx = torch.arange(s, min(s + chunk, len(cl)), device=dev)
                tot = dc0[i_idx][:, None, None] + d_ci[i_idx][:, :, None] + d_ci.T[None, :, :] + dc0[None, None, :]   # (ci, ni, cl)
                ok = tot <= two_r
                ok &= (i_idx[:, None, None] <= torch.arange(len(cl), device=dev)[None, None, :])         # canonical orientation
                ok &= (cl[i_idx][:, None, None] != inter[None, :, None]) & (inter[None, :, None] != cl[None, None, :])
                a, b, c = torch.nonzero(ok, as_tuple=True)
                parts.append(torch.stack([cl[i_idx[a]], inter[b], cl[c]], 1))
            atoms4 = torch.cat(parts) if parts else torch.zeros(0, 3, dtype=torch.long, device=dev)
            deg4 = torch.where(atoms4[:, 0] == atoms4[:, 2], 1.0, 2.0)
            out[4] = finish(4, atoms4, deg4)
        if self.merge_degenerate and len(cl) > 1:
            out = self._merge_single_scattering(out, dev)
            ref = out[2][3][torch.argmin(out[2][0][:, 1].norm(dim=-1))] * out[2][2][torch.argmin(out[2][0][:, 1].norm(dim=-1))]
        # importance filter: degeneracy-weighted (symmetry-equivalent multiple-scattering paths counted together, as
        # FEFF does), relative to the shortest single-scattering path
        kept = {}
        for nleg, (coords, ipots, deg, xp, atoms) in out.items():
            dtot = self.degenerate_total(coords, deg) if (nleg > 2 and self.merge_degenerate and len(deg)) else deg
            frac = 100 * dtot * xp / ref
            keep = frac >= self.critpw
            kept[nleg] = (coords[keep], ipots[keep], deg[keep], xp[keep], atoms[keep])
        return kept, float(ref)
