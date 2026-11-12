"""FEFF's path stage on the GPU: the Rehr-Albers algorithm batched over atom sequences and energies in torch.

Same algorithm as feffgpu.genfmt (the literal Python port, the reference), reorganised for tensors:
  * every sequence of a batch has the same number of legs; coordinates (B, nleg, 3) in A, absorber first at the
    origin, scatterers in path order; potential indices (B, nleg);
  * rotation matrix elements d^l_{mn}(beta) for |m|, |n| <= 2 by a three-term recursion in l (validated against
    FEFF's rot3i), spherical-wave factors by FEFF's recursion vectorised over (sequence, energy), with FEFF's own
    partial-wave truncation per energy applied inside the recursion;
  * scattering matrices as contractions over the partial waves for each pair of lambda indices; the termination
    matrix, the product over legs and the trace as batched small matrix products;
  * output: FEFF's complex path signal cchi(k) on the phase file's real-axis grid (chi = Im[cchi exp(2ikR)]), and
    a fine-grid evaluation by interpolation of log-amplitude and unwrapped phase (what FEFF and Larch do).
Everything is differentiable in the coordinates. Single precision (complex64) is enough because FEFF's truncation
keeps the recursion factors below 1e2 (checked for W); the reference module stays in double for validation.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from .pad import read_phase_pad, BOHR
from .genfmt import setlam, snlm, setkap, bcoef_unpolarized, MTOT

MMAX = 3            # largest |m| of the lambda sets used here (iorder 2; exact single scattering needs |m| <= l_init + 1, 3 for M edges)


def wigner_d_explicit(beta, l, m, n):
    """d^l_{mn}(beta) by Wigner's explicit sum (same convention as the closed forms below); for the small-l seeds."""
    from math import factorial, sqrt
    c2, s2 = torch.cos(beta / 2), torch.sin(beta / 2)
    out = torch.zeros_like(beta)
    pref = sqrt(factorial(l + m) * factorial(l - m) * factorial(l + n) * factorial(l - n))
    for k in range(0, 2 * l + 1):
        a, b, c, d = l + n - k, k, m - n + k, l - m - k
        if min(a, b, c, d) < 0:
            continue
        coef = (-1) ** (m - n + k) * pref / (factorial(a) * factorial(b) * factorial(c) * factorial(d))
        out = out + coef * c2 ** (2 * l + n - m - 2 * k) * s2 ** (m - n + 2 * k)
    return out


def wigner_d_small(beta, lmax, mmax=MMAX):
    """d^l_{mn}(beta) for l = 0..lmax and |m|, |n| <= mmax, by the three-term recursion in l (Edmonds' convention
    as in FEFF's rot3i, checked against it), seeded by closed forms at l = 1, 2 and by Wigner's explicit sum for the
    entries with |m| or |n| = 3 at l = 3. beta: (B,) tensor. Returns (B, lmax+1, 2 mmax+1, 2 mmax+1)."""
    B = beta.shape[0]
    dev, dt = beta.device, beta.dtype
    c, s = torch.cos(beta), torch.sin(beta)
    c2, s2 = torch.cos(beta / 2), torch.sin(beta / 2)
    d = torch.zeros(B, lmax + 1, 2 * mmax + 1, 2 * mmax + 1, device=dev, dtype=dt)
    o = mmax                                                   # index offset: m -> m + o
    d[:, 0, o, o] = 1.0
    if lmax >= 1:
        # l = 1 (Edmonds' convention, as FEFF's rot3i): d_{10} = -sin/sqrt2, d_{01} = +sin/sqrt2
        d[:, 1, o + 1, o + 1] = c2 ** 2
        d[:, 1, o + 1, o] = -s / math.sqrt(2.0)
        d[:, 1, o + 1, o - 1] = s2 ** 2
        d[:, 1, o, o + 1] = s / math.sqrt(2.0)
        d[:, 1, o, o] = c
        d[:, 1, o, o - 1] = -s / math.sqrt(2.0)
        d[:, 1, o - 1, o + 1] = s2 ** 2
        d[:, 1, o - 1, o] = s / math.sqrt(2.0)
        d[:, 1, o - 1, o - 1] = c2 ** 2
    if lmax >= 2:
        # l = 2 closed forms (same convention), then the symmetries d_{nm} = (-1)^{m-n} d_{mn} = d_{-m,-n}
        d2 = {(2, 2): c2 ** 4, (2, 1): -(1 + c) / 2 * s, (2, 0): math.sqrt(3 / 8) * s ** 2,
              (2, -1): -(1 - c) / 2 * s, (2, -2): s2 ** 4, (1, 1): (1 + c) / 2 * (2 * c - 1),
              (1, 0): -math.sqrt(3 / 2) * s * c, (1, -1): (1 - c) / 2 * (2 * c + 1), (0, 0): (3 * c ** 2 - 1) / 2}
        full = {}
        for (m, n), v in d2.items():
            full[(m, n)] = v
            full[(n, m)] = (-1) ** (m - n) * v
            full[(-m, -n)] = (-1) ** (m - n) * v
            full[(-n, -m)] = v
        for (m, n), v in full.items():
            if abs(m) <= mmax and abs(n) <= mmax:
                d[:, 2, o + m, o + n] = v
    ms = torch.arange(-mmax, mmax + 1, device=dev, dtype=dt)
    mm = ms[:, None].expand(2 * mmax + 1, 2 * mmax + 1)
    nn = ms[None, :].expand(2 * mmax + 1, 2 * mmax + 1)
    for l in range(2, lmax):
        # d^{l+1} from d^l and d^{l-1} (recurrence in l for fixed m, n; Varshalovich 4.8.2 form)
        a = (l + 1) * torch.sqrt(((l) ** 2 - mm ** 2).clamp_min(0) * ((l) ** 2 - nn ** 2).clamp_min(0))
        b = (2 * l + 1) * (l * (l + 1) * c[:, None, None] - mm * nn)
        den = l * torch.sqrt((((l + 1) ** 2 - mm ** 2) * ((l + 1) ** 2 - nn ** 2)).clamp_min(0))
        den = torch.where(den == 0, torch.ones_like(den), den)
        d[:, l + 1] = (b * d[:, l] - a * d[:, l - 1]) / den
        if l + 1 == 3 and mmax >= 3:
            # the recursion cannot start entries with |m| or |n| = 3 (d^2 = 0 there): seed them explicitly
            for m in range(-3, 4):
                for n in range(-3, 4):
                    if max(abs(m), abs(n)) == 3:
                        d[:, 3, o + m, o + n] = wigner_d_explicit(beta, 3, m, n)
    return d


class GPUPathStage:
    def __init__(self, phase, device="cuda", dtype=torch.complex64, iorder=2):
        P = read_phase_pad(phase) if isinstance(phase, str) else phase
        if P.nsp != 1:
            raise NotImplementedError("spin-dependent phase files are not ported")
        self.P, self.iorder, self.device, self.cdtype = P, iorder, torch.device(device), dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        E = self.E = P.ne1
        lmax = P.lmax[:E]                                                          # (E, nph+1)
        self.L1 = int(lmax.max()) + 1                                              # largest il
        self.lmax = torch.tensor(lmax, device=self.device)
        self.xk = torch.tensor(P.xk[:E], device=self.device, dtype=self.rdtype)
        self.ck = torch.tensor(P.ck[:E], device=self.device, dtype=dtype)
        kinit, linit = setkap(P.ihole)
        self.kinit, self.ilinit = kinit, linit + 1
        # j-averaged t-matrix per (energy, il, potential): tl(-l)(l+1) + tl(l) l, il = l + 1 (index 0 unused)
        tt = np.zeros((E, self.L1 + 1, P.nph + 1), complex)
        for iph in range(P.nph + 1):
            for il in range(1, self.L1 + 1):
                ll = il - 1
                tm = (np.exp(2j * P.ph[:E, -ll + P.ltot, 0, iph]) - 1) / 2j
                tp = (np.exp(2j * P.ph[:E, ll + P.ltot, 0, iph]) - 1) / 2j
                tt[:, il, iph] = tm * (ll + 1) + tp * ll
        self.tltl = torch.tensor(tt, device=self.device, dtype=dtype)
        xn = snlm(self.L1 + 1, MTOT + 1)
        self.xnlm = torch.tensor(xn, device=self.device, dtype=self.rdtype)      # (L1+2, MTOT+2), 1-based
        self.lind, bdiag = bcoef_unpolarized(kinit, le2=0)
        self.bdiag = torch.tensor(bdiag, device=self.device, dtype=dtype)
        self.rkk = torch.tensor(P.rkk[:E, :, 0], device=self.device, dtype=dtype)    # (E, 8)
        self.lam = {}                                                               # per nsc: lambda data

    # -- lambda bookkeeping ----------------------------------------------------------------------------------
    def _lambda(self, nsc):
        if nsc in self.lam:
            return self.lam[nsc]
        mlam, nlam, lamx, laml0x, mmaxp1, nmax = setlam(self.iorder, None, nsc, nsc + 1, self.ilinit)
        m = torch.tensor(mlam[1:], device=self.device)
        n = torch.tensor(nlam[1:], device=self.device)
        am = m.abs()
        il = torch.arange(self.L1 + 2, device=self.device)                        # 0..L1+1 (0 unused), as the c arrays
        # static mask over (lam1, lam2, il): il >= max(|m1|+1, |m2|+1, |m1|+n1+1, n2+1), |m1| < il, |m2| < il
        ilmin = torch.maximum(torch.maximum(am[:, None] + 1, am[None, :] + 1),
                              torch.maximum((am + n)[:, None] + 1, (n + 1)[None, :]))
        mask = (il[None, None, :] >= ilmin[:, :, None]) & (il[None, None, :] > am[:, None, None]) \
               & (il[None, None, :] > am[None, :, None])
        data = {"m": m, "n": n, "am": am, "lamx": lamx, "laml0x": laml0x, "mmaxp1": mmaxp1, "nmax": nmax,
                "mask": mask.to(self.cdtype), "mnp1": min(self.L1, mmaxp1 + nmax)}
        self.lam[nsc] = data
        return data

    # -- geometry (pathgeom.f, vectorised) ----------------------------------------------------------------
    @staticmethod
    def _trig(v):
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        r = torch.sqrt(x * x + y * y + z * z)
        rxy = torch.sqrt(x * x + y * y)
        small_r, small_xy = r < 1e-6, rxy < 1e-6
        ct = torch.where(small_r, torch.ones_like(r), z / r.clamp_min(1e-30))
        st = torch.where(small_r, torch.zeros_like(r), rxy / r.clamp_min(1e-30))
        cp = torch.where(small_xy, torch.where(ct < 0, -torch.ones_like(r), torch.ones_like(r)), x / rxy.clamp_min(1e-30))
        sp = torch.where(small_xy, torch.zeros_like(r), y / rxy.clamp_min(1e-30))
        return ct, st, cp, sp

    @staticmethod
    def _arg(c, fi):
        x, y = c.real, c.imag
        x = torch.where(x.abs() < 1e-6, torch.zeros_like(x), x)
        y = torch.where(y.abs() < 1e-6, torch.zeros_like(y), y)
        both = (x == 0) & (y == 0)
        # atan2's backward at (0, 0) is 0/0; feed it (0, 1) there so the gradient is finite (the value is fi anyway)
        xs = torch.where(both, torch.ones_like(x), x)
        ys = torch.where(both, torch.zeros_like(y), y)
        return torch.where(both, fi, torch.atan2(ys, xs))

    def geometry(self, coords_bohr):
        """ri (B, nleg), beta (B, nleg), eta (B, nleg) for legs 1..nleg (0-based leg index = leg-1)."""
        B, nleg, _ = coords_bohr.shape
        nsc = nleg - 1
        # rat[:, j] for j = 0..nleg: j=0 absorber copy, 1..nsc scatterers, nleg absorber
        rat = torch.cat([coords_bohr[:, :1], coords_bohr[:, 1:], coords_bohr[:, :1]], 1)   # (B, nleg+1, 3)
        ri, beta, alpha, gamma = [], [], [], []
        for j in range(1, nleg + 1):
            if j == nsc + 1:
                i, ip1, im1 = 0, 1, nsc
            else:
                i, ip1, im1 = j, j + 1, j - 1
            ctp, stp, cpp, spp = self._trig(rat[:, ip1] - rat[:, i])
            ct, st, cp, sp = self._trig(rat[:, i] - rat[:, im1])
            cppp = cp * cpp + sp * spp
            sppp = spp * cp - cpp * sp
            phi, phip = torch.atan2(sp, cp), torch.atan2(spp, cpp)
            alph = torch.complex(-(st * ctp - ct * stp * cppp), stp * sppp)
            b = (ct * ctp + st * stp * cppp).clamp(-1.0, 1.0)
            # sin(beta) from the cross product of the unit leg vectors, regularised: beta = atan2(sin, cos) equals
            # arccos(b) in exact arithmetic and keeps the gradient finite on straight legs (beta = 0 or pi), where
            # arccos' derivative diverges
            u1 = torch.stack([st * cp, st * sp, ct], -1)
            u2 = torch.stack([stp * cpp, stp * spp, ctp], -1)
            sb2 = (torch.linalg.cross(u1, u2) ** 2).sum(-1)
            gamm = torch.complex(-(st * ctp * cppp - ct * stp), -st * sppp)
            a_j = self._arg(alph, phip - phi)
            g_j = self._arg(gamm, torch.zeros_like(phi))
            alpha.append(math.pi - g_j)
            gamma.append(math.pi - a_j)
            beta.append(torch.atan2(torch.sqrt(sb2 + 1e-30), b))
            ri.append(torch.linalg.norm(rat[:, i] - rat[:, im1], dim=-1))
        ri, beta = torch.stack(ri, 1), torch.stack(beta, 1)
        alpha, gamma = torch.stack(alpha, 1), torch.stack(gamma, 1)
        alpha_prev = torch.cat([alpha[:, -1:], alpha[:, :-1]], 1)              # alpha(j-1) with alpha(0) = alpha(nleg)
        eta = alpha_prev + gamma
        return ri, beta, eta

    # -- spherical-wave factors (sclmz.f, vectorised over sequences and energies) --------------------------
    def _clm(self, rho, lxp1, mnp1):
        """rho (B, E) complex; lxp1 (B, E) int (FEFF's per-energy partial-wave limit for this leg).
        Returns c (B, E, L1+2, mnp1+1) with FEFF's 1-based [il, im] indexing, zero beyond lxp1."""
        B, E = rho.shape
        z = -1j / rho
        L1 = self.L1
        c = [torch.zeros(B, E, mnp1 + 1, device=self.device, dtype=self.cdtype) for _ in range(L1 + 2)]
        c[1][:, :, 1] = 1.0
        c[2][:, :, 1] = 1.0 - z
        lmaxv = L1 - 1
        alive = lambda il: (il <= lxp1).to(self.cdtype)                           # il within this leg's limit
        for il in range(2, lmaxv + 1):
            c[il + 1][:, :, 1] = (c[il - 1][:, :, 1] - z * (2 * il - 1) * c[il][:, :, 1]) * alive(il + 1)
        cmm = torch.ones_like(z)
        for im in range(2, mnp1 + 1):
            m = im - 1
            cmm = -cmm * (2 * m - 1) * z
            c[im][:, :, im] = cmm * alive(im)
            if im + 1 <= L1 + 1:
                c[im + 1][:, :, im] = cmm * (2 * m + 1) * (1 - im * z) * alive(im + 1)
            for il in range(im + 1, lmaxv + 1):
                l = il - 1
                c[il + 1][:, :, im] = (c[l][:, :, im] - (2 * l + 1) * z * (c[il][:, :, im] + c[il][:, :, m])) * alive(il + 1)
        return torch.stack(c, 2)                                                     # (B, E, L1+2, mnp1+1)

    # -- the path signal ------------------------------------------------------------------------------------
    def cchi(self, coords, ipots):
        """coords (B, nleg, 3) A, ipots (B, nleg) long. Returns cchi (B, E) complex and reff (B,) in A."""
        coords_bohr = coords.to(self.rdtype) / BOHR
        B, nleg, _ = coords.shape
        nsc = nleg - 1
        lam = self._lambda(nsc)
        lamx, laml0x = lam["lamx"], lam["laml0x"]
        ri, beta, eta = self.geometry(coords_bohr)                                  # (B, nleg) each, leg j at index j-1
        reff = ri.sum(1) / 2
        ipots = ipots.to(self.device)
        # ipot per path atom in FEFF's numbering: atom j (1..nsc) scatterers, atom nleg = absorber (= ipots[:, 0])
        ipot_atom = torch.cat([ipots[:, 1:], ipots[:, :1]], 1)                    # (B, nleg): index j-1 -> atom j
        # rotation matrices per leg: d^l(beta_leg), (B, nleg, L1, 5, 5) with l = il - 1
        dl = wigner_d_small(beta.reshape(-1), self.L1 - 1, MMAX).reshape(B, nleg, self.L1, 2 * MMAX + 1, 2 * MMAX + 1)
        m, n, am = lam["m"], lam["n"], lam["am"]
        E = self.E
        ck = self.ck[None, :]                                                        # (1, E)
        rho = ck * ri.to(self.cdtype)[:, :, None]                                     # (B, nleg, E)
        # per leg: the leg's two endpoint potentials set FEFF's partial-wave limit
        lmax_atom = self.lmax[:, ipot_atom]                                           # (E, B, nleg)
        lmax_atom = lmax_atom.permute(1, 2, 0)                                        # (B, nleg, E)
        lmax_prev = torch.cat([lmax_atom[:, -1:], lmax_atom[:, :-1]], 1)              # atom j-1 (atom 0 = absorber = atom nleg)
        lxp1 = torch.maximum(lmax_atom, lmax_prev) + 1                                # (B, nleg, E)
        mnp1 = lam["mnp1"]
        clm = [self._clm(rho[:, j], lxp1[:, j], mnp1) for j in range(nleg)]           # each (B, E, L1+2, mnp1+1)
        il = torch.arange(self.L1 + 2, device=self.device)
        # gam_lambda[b, e, lam, il] and gamtl_lambda[b, e, lam, il] for each leg
        idx_imn = (am + n + 1)                                                        # c index for gam
        idx_in = (n + 1)                                                              # c index for gamtl
        sign = ((-1.0) ** am.to(self.rdtype)).to(self.cdtype)
        xn_m = self.xnlm[:, :]                                                        # (L1+2, MTOT+2)

        def gam_of(leg):
            c = clm[leg]                                                              # (B, E, L1+2, mnp1+1)
            g = c[:, :, :, idx_imn.clamp(max=mnp1)]                                   # (B, E, L1+2, lamx)
            ok = (idx_imn[None, None, None, :] <= il[None, None, :, None]) & (idx_imn[None, None, None, :] <= mnp1)
            xn = xn_m[:, am + 1]                                                      # (L1+2, lamx)
            return g * ok.to(self.cdtype) * (xn.to(self.cdtype) * sign[None, :])[None, None]

        def gamtl_of(leg, ipot_leg, use_t=True):
            c = clm[leg]
            g = c[:, :, :, idx_in]                                                    # (B, E, L1+2, lamx)
            xn = xn_m[:, am + 1]                                                      # (L1+2, lamx); zero outside im <= il
            xn = torch.where(xn == 0, torch.ones_like(xn), xn).to(self.cdtype)       # safe divide, masked later
            if use_t:
                t = self.tltl[:, :, :]                                                # (E, L1+1, nph+1)
                t_leg = t[:, :, ipot_leg].permute(2, 0, 1)                           # (B, E, L1+1)
                t_leg = torch.cat([t_leg, torch.zeros_like(t_leg[:, :, :1])], 2)     # pad to L1+2
                return g / xn[None, None] * t_leg[:, :, :, None]
            tl = (2 * il - 1).to(self.cdtype)                                         # termination: 2l+1
            return g / xn[None, None] * tl[None, None, :, None]

        def fmat(leg_gam, leg_gamtl, ipot_leg, eta_leg, dl_leg, lam1x, lam2x):
            """F[b, e, lam1 <= lam1x, lam2 <= lam2x] = sum_il gam(leg_gam) gamtl(leg_gamtl) d(leg_gamtl) mask."""
            G1 = gam_of(leg_gam)[:, :, :, :lam1x]                                     # (B, E, L, lam1)
            G2 = gamtl_of(leg_gamtl, ipot_leg)[:, :, :, :lam2x]                       # (B, E, L, lam2)
            # per-energy partial-wave limit of the scattering atom (fmtrxi: il <= lmax(ie, ipot(ilegp)) + 1)
            ilx = (self.lmax[:, ipot_leg] + 1).T                                       # (B, E)
            alive = (il[None, None, :] <= ilx[:, :, None]).to(self.cdtype)             # (B, E, L)
            D = dl_leg[:, :, m[:lam1x, None] + MMAX, m[None, :lam2x] + MMAX]           # (B, L1, lam1, lam2), l = il-1
            D = torch.cat([torch.zeros_like(D[:, :1]), D, torch.zeros_like(D[:, :1])], 1)   # il index 0..L1+1
            mask = lam["mask"][:lam1x, :lam2x, :]                                      # (lam1, lam2, L)
            Dm = D.to(self.cdtype) * mask.permute(2, 0, 1)[None]                       # (B, L, lam1, lam2), masked
            G1a = G1 * alive[:, :, :, None]                                            # (B, E, L, lam1)
            cols = []
            for b in range(lam2x):                                                     # small loop keeps memory at (B, E, L, lam1)
                Hb = G1a * Dm[:, None, :, :, b]                                        # (B, E, L, lam1)
                cols.append(torch.einsum("zeia,zei->zea", Hb, G2[:, :, :, b]))
            F = torch.stack(cols, -1)                                                  # (B, E, lam1, lam2)
            ph = torch.exp(-1j * eta_leg.to(self.cdtype)[:, None, None] * m[:lam1x].to(self.cdtype)[None, None, :])
            return F * ph[:, :, :, None]

        # scattering matrices: fmtrxi(ileg = ilegp + 1, ilegp): gam from leg ileg, gamtl/d/ipot/eta from ilegp;
        # leg indices here are 0-based: FEFF leg j -> index j-1; atom at the end of leg j is atom j (ipot_atom[j-1])
        P = fmat(1, 0, ipot_atom[:, 0], eta[:, 1], dl[:, 0], lamx, laml0x)              # f(2,1)
        for ilegp in range(2, nsc):                                                    # f(ilegp+1, ilegp), ilegp = 2..nsc-1
            Fm = fmat(ilegp, ilegp - 1, ipot_atom[:, ilegp - 1], eta[:, ilegp], dl[:, ilegp - 1], lamx, lamx)
            P = Fm @ P
        if nleg > 2:
            Fl = fmat(nleg - 1, nleg - 2, ipot_atom[:, nleg - 2], eta[:, nleg - 1], dl[:, nleg - 2], laml0x, lamx)
            P = Fl @ P                                                                 # (B, E, laml0x, laml0x)
        else:
            P = P[:, :, :laml0x, :laml0x]
        # termination (mmtrxi + mmtr, ipol = 0): gam from leg 1 (index 0), gamtl from leg nleg (index nleg-1),
        # only il = l+1 of the final-state channels, d^l(beta_nleg) with the spin-traced diagonal coefficients
        G1 = gam_of(0)[:, :, :, :laml0x]
        G2t = gamtl_of(nleg - 1, None, use_t=False)[:, :, :, :laml0x]
        T = torch.zeros(B, E, laml0x, laml0x, device=self.device, dtype=self.cdtype)
        for k in range(1, 9):
            l = self.lind[k]
            if l < 0:
                continue
            il_k = l + 1
            okm = (am[:laml0x] + 1 <= il_k)
            w = self.bdiag[k] * self.rkk[:, k - 1] ** 2                                # (E,)
            Dk = dl[:, nleg - 1, l][:, m[:laml0x, None] + MMAX, m[None, :laml0x] + MMAX]   # (B, lam, lam)
            g1 = G1[:, :, il_k, :] * okm.to(self.cdtype)[None, None, :]
            g2 = G2t[:, :, il_k, :] * okm.to(self.cdtype)[None, None, :]
            T = T + w[None, :, None, None] * g1[:, :, :, None] * g2[:, :, None, :] * Dk.to(self.cdtype)[:, None]
        T = T * torch.exp(-1j * eta[:, 0].to(self.cdtype)[:, None, None, None] * m[:laml0x].to(self.cdtype)[None, None, :, None])
        ptrac = torch.einsum("zeab,zeba->ze", T, P)
        srho, prho = rho.sum(1), rho.prod(1)                                           # (B, E)
        cfac = torch.exp(1j * (srho - 2 * self.xk[None, :].to(self.cdtype) * reff.to(self.cdtype)[:, None])) / prho
        return ptrac * cfac, reff * BOHR

    # -- fine-grid signal ---------------------------------------------------------------------------------------
    def spline_matrix(self, q):
        """Interpolation matrix S (len(q), E) from FEFF's real-axis k grid (1/A) to the points q, cubic spline
        (scipy UnivariateSpline s=0, as Larch), as a dense linear map; q beyond the grid is clamped."""
        from scipy.interpolate import UnivariateSpline
        kf = self.P.xk[: self.E] / BOHR
        qn = np.clip(np.asarray(q, float), kf[0], kf[-1])
        S = np.zeros((len(qn), self.E))
        for j in range(self.E):
            y = np.zeros(self.E); y[j] = 1.0
            S[:, j] = UnivariateSpline(kf, y, s=0, k=3)(qn)
        return torch.tensor(S, device=self.device, dtype=self.rdtype)

    def linear_matrix(self, q):
        """Interpolation matrix for FEFF's own final-stage convention (terp1 in ff2gen.f): linear interpolation of
        amplitude and phase from the real-axis grid, as a dense linear map; q beyond the grid is clamped."""
        kf = self.P.xk[: self.E] / BOHR
        qn = np.clip(np.asarray(q, float), kf[0], kf[-1])
        S = np.zeros((len(qn), self.E))
        j = np.clip(np.searchsorted(kf, qn, side="right") - 1, 0, self.E - 2)
        f = (qn - kf[j]) / (kf[j + 1] - kf[j])
        S[np.arange(len(qn)), j] = 1 - f
        S[np.arange(len(qn)), j + 1] = f
        return torch.tensor(S, device=self.device, dtype=self.rdtype)

    @staticmethod
    def unwrap(phase):
        """Unwrap along the last axis without branches (differences brought into (-pi, pi])."""
        d = torch.diff(phase, dim=-1)
        d = d - 2 * math.pi * torch.round(d / (2 * math.pi))
        return torch.cat([phase[..., :1], phase[..., :1] + torch.cumsum(d, -1)], -1)

    def chi_fine(self, cchi, reff, q, S=None, convention="feff"):
        """chi(q) = Im[cchi(q) exp(2 i q R)] per sequence on a fine grid q (1/A), with amplitude and unwrapped
        phase of cchi interpolated from FEFF's real-axis grid. convention "feff": linear interpolation of
        amplitude and phase (FEFF's ff2x, hence chi.dat); "larch": cubic spline of amplitude and phase (Larch's
        feffpath tables). S is the matching interpolation matrix (linear_matrix or spline_matrix).
        Returns (B, len(q)) real."""
        if S is None:
            S = self.linear_matrix(q) if convention == "feff" else self.spline_matrix(q)
        amp = torch.sqrt(cchi.real ** 2 + cchi.imag ** 2 + 1e-30)            # |cchi| with a finite gradient at 0
        # genfmt.f: the phase is 0 where the signal vanishes (|cchi| < 1e-16); otherwise the angle of a numerically
        # zero signal (e.g. a scatterer whose phase shifts are zero at some energies) would wander by pi and spoil the
        # interpolation of the unwrapped phase to the next non-zero point
        pha = self.unwrap(torch.where(amp >= 1e-16, torch.angle(cchi), torch.zeros_like(amp)))
        a = amp.to(S.dtype) @ S.T
        ph = pha.to(S.dtype) @ S.T
        qt = torch.as_tensor(q, device=cchi.device, dtype=S.dtype)
        return a * torch.sin(ph + 2 * qt[None, :] * reff.to(S.dtype)[:, None])
