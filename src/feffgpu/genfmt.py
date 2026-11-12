"""FEFF's path stage (GENFMT, the Rehr-Albers separable multiple-scattering algorithm) in Python/numpy.

A literal port of feff85exafs: GENFMT/onepath.f, pathgeom.f, setlam.f, rot3i.f, sclmz.f, snlm.f, fmtrxi.f,
mmtrxi.f, mmtr.f, MATH/bcoef.f (unpolarised branch), FF2X/fdtarr.f, COMMON/setkap.f, pijump.f. Polarisation
and spin are not ported (ipol = 0, nsp = 1): the EXAFS case. Input is the phase file (feffgpu.pad) and a
path geometry; output the seven feffNNNN.dat columns, exactly what the one-path library returns.

Index conventions follow the Fortran (1-based legs, il = l + 1, im = m + 1) so the two can be read side by side;
arrays are allocated one larger and index 0 is unused where the Fortran starts at 1. This reference version loops
in Python over energies and lambda indices; it exists to be exact and readable, and is the oracle for the batched
torch version that follows. "Based on or developed using Distribution: FEFF8.5L, Copyright (c) [2013] University of
Washington" (BSD-style licence of feff85exafs; see the proposal, section 9).
"""
from __future__ import annotations
import numpy as np
from .pad import PhaseData, read_phase_pad, BOHR

MTOT, NTOT, LAMTOT = 4, 2, 15
EPS = 1.0e-16


def setkap(ihole):
    """Initial-state angular momentum and kappa from the hole index (COMMON/setkap.f)."""
    if ihole in (1, 2, 5, 10, 17, 24, 27):
        return -1, 0
    if ihole in (3, 6, 11, 18, 25, 30):
        return 1, 1
    if ihole in (4, 7, 12, 19, 26):
        return -2, 1
    if ihole in (8, 13, 20, 27):
        return 2, 2
    if ihole in (9, 14, 21, 28):
        return -3, 2
    if ihole in (15, 22):
        return 3, 3
    if ihole in (16, 23):
        return -4, 3
    raise ValueError(f"invalid hole number {ihole}")


def snlm(lmaxp1, mmaxp1):
    """Legendre normalisation factors xnlm[il, im] (snlm.f), 1-based."""
    afac = 1.0 / 64
    flg = np.zeros(211)
    flg[0], flg[1] = 1.0, afac
    for i in range(2, 211):
        flg[i] = flg[i - 1] * i * afac
    xnlm = np.zeros((lmaxp1 + 1, mmaxp1 + 1))
    for il in range(1, lmaxp1 + 1):
        for im in range(1, min(mmaxp1, il) + 1):
            l, m = il - 1, im - 1
            xnlm[il, im] = np.sqrt((2 * l + 1) * flg[l - m] / flg[l + m]) * afac ** m
    return xnlm


def trig(v):
    """cos/sin of theta and phi of a vector (trig.f)."""
    x, y, z = v
    r = np.sqrt(x * x + y * y + z * z)
    rxy = np.sqrt(x * x + y * y)
    if r < 1e-6:
        ct, st = 1.0, 0.0
    else:
        ct, st = z / r, rxy / r
    if rxy < 1e-6:
        cp, sp = (1.0 if ct >= 0 else -1.0), 0.0
    else:
        cp, sp = x / rxy, y / rxy
    return ct, st, cp, sp


def arg(c, fi):
    x, y = c.real, c.imag
    if abs(x) < 1e-6:
        x = 0.0
    if abs(y) < 1e-6:
        y = 0.0
    return fi if (x == 0.0 and y == 0.0) else np.arctan2(y, x)


def pathgeom(rat, nleg):
    """Leg lengths ri[1..nleg], scattering angles beta[1..nleg] and Euler combinations eta[0..nleg+1] from
    rat[:, 0..nleg+1] (bohr; rat[:, nleg] is the absorber, rat[:, 0] its copy). pathgeom.f, ipol = 0."""
    nsc = nleg - 1
    ri = np.zeros(nleg + 2)
    beta = np.zeros(nleg + 2)
    alpha = np.zeros(nleg + 2)
    gamma = np.zeros(nleg + 2)
    eta = np.zeros(nleg + 2)
    for j in range(1, nleg + 1):
        if j == nsc + 1:
            i, ip1, im1 = 0, 1, nsc
        else:
            i, ip1, im1 = j, j + 1, j - 1
        ctp, stp, cpp, spp = trig(rat[:, ip1] - rat[:, i])
        ct, st, cp, sp = trig(rat[:, i] - rat[:, im1])
        cppp = cp * cpp + sp * spp
        sppp = spp * cp - cpp * sp
        phi, phip = np.arctan2(sp, cp), np.arctan2(spp, cpp)
        alph = -(st * ctp - ct * stp * cppp - 1j * stp * sppp)
        b = min(1.0, max(-1.0, ct * ctp + st * stp * cppp))
        gamm = -(st * ctp * cppp - ct * stp + 1j * st * sppp)
        a_j = arg(alph, phip - phi)
        beta[j] = np.arccos(b)
        g_j = arg(gamm, phi - phi)
        alpha[j] = np.pi - g_j                       # rotation of frame -> rotation of vectors
        gamma[j] = np.pi - a_j
        ri[j] = np.linalg.norm(rat[:, i] - rat[:, im1])
    alpha[0] = alpha[nleg]
    for j in range(1, nleg + 1):
        eta[j] = alpha[j - 1] + gamma[j]
    return ri, beta, eta


def setlam(iorder, beta, nsc, nleg, ilinit):
    """Lambda index set (setlam.f): mlam, nlam (1-based lists), lamx, laml0x, mmaxp1, nmax."""
    if nsc == 1:
        mmax = nmax = ilinit
        iord = 2 * nmax + mmax
    elif iorder < 10:
        iord, mmax, nmax = iorder, iorder, iorder // 2
    else:
        raise NotImplementedError("icalc >= 10 not ported")
    mlam0, nlam0 = [], []
    filled = False
    for n in range(0, nmax + 1):
        for m in range(0, mmax + 1):
            if 2 * n + m > iord:
                continue
            if len(mlam0) >= LAMTOT:                 # FEFF: 'Lambda array filled, some order lost' (M edges, nsc = 1)
                filled = True
                break
            mlam0.append(-m); nlam0.append(n)
            if m == 0:
                continue
            if len(mlam0) >= LAMTOT:
                filled = True
                break
            mlam0.append(m); nlam0.append(n)
        if filled:
            break
    lamx = len(mlam0)
    first = [i for i in range(lamx) if nlam0[i] <= ilinit and abs(mlam0[i]) <= ilinit]
    rest = [i for i in range(lamx) if i not in first]
    order = first + rest
    mlam = [0] + [mlam0[i] for i in order]
    nlam = [0] + [nlam0[i] for i in order]
    laml0x = len(first)
    mmaxp1 = max(m + 1 for m in mlam[1:])
    nmax_out = max(nlam[1:])
    if nmax_out > NTOT or mmaxp1 > MTOT + 1:
        raise ValueError("lambda set exceeds mtot/ntot")
    return mlam, nlam, lamx, laml0x, mmaxp1, nmax_out


def rot3i(lxp1, mxp1, beta_leg, ltot):
    """Rotation matrix elements d^l(beta) by Edmonds' recursion (rot3i.f). Returns dri[il, m1+MTOT+1, m2+MTOT+1]
    for il = 1..lxp1 and |m| <= min(il-1, mxp1-1), 1-based in il."""
    nm = mxp1
    ndm = lxp1 + nm - 1
    xc, xs, s = np.cos(beta_leg / 2), np.sin(beta_leg / 2), np.sin(beta_leg)
    size = max(2 * lxp1 + 1, ndm + 1) + 1
    d0 = np.zeros((lxp1 + 1, size, size))
    d0[1, 1, 1] = 1.0
    if lxp1 >= 2:
        d0[2, 1, 1] = xc ** 2
        d0[2, 1, 2] = s / np.sqrt(2.0)
        d0[2, 1, 3] = xs ** 2
        d0[2, 2, 1] = -d0[2, 1, 2]
        d0[2, 2, 2] = np.cos(beta_leg)
        d0[2, 2, 3] = d0[2, 1, 2]
        d0[2, 3, 1] = d0[2, 1, 3]
        d0[2, 3, 2] = -d0[2, 2, 3]
        d0[2, 3, 3] = d0[2, 1, 1]
    for l in range(3, lxp1 + 1):
        ln = min(2 * l - 1, ndm)
        lm = min(2 * l - 3, ndm)
        for n in range(1, ln + 1):
            for m in range(1, lm + 1):
                t1 = (2 * l - 1 - n) * (2 * l - 2 - n)
                t = (2 * l - 1 - m) * (2 * l - 2 - m)
                f1 = np.sqrt(t1 / t)
                f2 = np.sqrt((2 * l - 1 - n) * (n - 1) / t)
                f3 = np.sqrt((n - 2) * (n - 1) / t)
                dlnm = f1 * xc ** 2 * d0[l - 1, n, m]
                if n - 1 > 0:
                    dlnm -= f2 * s * d0[l - 1, n - 1, m]
                if n - 2 > 0:
                    dlnm += f3 * xs ** 2 * d0[l - 1, n - 2, m]
                d0[l, n, m] = dlnm
                if n > 2 * l - 3:
                    d0[l, m, n] = (-1) ** (n - m) * d0[l, n, m]
            if n > 2 * l - 3:
                d0[l, 2 * l - 2, 2 * l - 2] = d0[l, 2, 2]
                d0[l, 2 * l - 1, 2 * l - 2] = -d0[l, 1, 2]
                d0[l, 2 * l - 2, 2 * l - 1] = -d0[l, 2, 1]
                d0[l, 2 * l - 1, 2 * l - 1] = d0[l, 1, 1]
    dri = np.zeros((ltot + 2, 2 * MTOT + 2, 2 * MTOT + 2))
    for il in range(1, lxp1 + 1):
        mx = min(il - 1, mxp1 - 1)
        for m1 in range(-mx, mx + 1):
            for m2 in range(-mx, mx + 1):
                dri[il, m1 + MTOT + 1, m2 + MTOT + 1] = d0[il, m1 + il, m2 + il]
    return dri


def sclmz(rho_leg, lmaxp1, mmaxp1, ltot):
    """Spherical-wave factors c_l^(m) z^m / m! for one leg (sclmz.f); clmi[il, im], 1-based."""
    clmi = np.zeros((ltot + 2, MTOT + NTOT + 2), complex)
    z = -1j / rho_leg
    cmm = 1.0 + 0j
    clmi[1, 1] = 1.0
    clmi[2, 1] = clmi[1, 1] - z
    lmax = lmaxp1 - 1
    for il in range(2, lmax + 1):
        clmi[il + 1, 1] = clmi[il - 1, 1] - z * (2 * il - 1) * clmi[il, 1]
    mmxp1 = min(mmaxp1, lmaxp1)
    for im in range(2, mmxp1 + 1):
        m = im - 1
        imp1 = im + 1
        cmm = -cmm * (2 * m - 1) * z
        clmi[im, im] = cmm
        clmi[imp1, im] = cmm * (2 * m + 1) * (1 - im * z)
        for il in range(imp1, lmax + 1):
            l = il - 1
            clmi[il + 1, im] = clmi[l, im] - (2 * l + 1) * z * (clmi[il, im] + clmi[il, m])
    return clmi


def bcoef_unpolarized(kinit, le2=0, lx=4):
    """Final-state channels (lind, 8 entries, -1 = unphysical) and the diagonal of bmat for ipol = 0 (bcoef.f)."""
    lind = [0] * 9
    for k in (-1, 0, 1):
        kap = kinit + k
        if k == 0:
            kap = -kap
        lkap = kap if kap > 0 else abs(kap) - 1
        if lkap > lx:
            lkap = -1
        lind[k + 2] = lkap
    for k in (-2, -1, 0, 1, 2):
        jkap = abs(kinit) + k
        if jkap <= 0:
            jkap = 0
        kap = jkap
        if kinit < 0 and abs(k) != 1:
            kap = -jkap
        if kinit > 0 and abs(k) == 1:
            kap = -jkap
        lkap = kap if kap > 0 else -kap - 1
        if lkap > lx or le2 == 0 or (le2 == 1 and abs(k) == 2):
            lkap = -1
        lind[k + 6] = lkap
    bdiag = [0.0] * 9
    for k in range(1, 9):
        if lind[k] < 0:
            continue
        i2 = 5 if (le2 == 2 and k > 3) else 3
        v = 0.5 / (2 * lind[k] + 1.0) / i2
        bdiag[k] = -v if k <= 3 else v
    # ispin = 0: the Green's function is spin diagonal, so bcoef traces bmat over the two spin projections
    # (bmat(..,0,..) += bmat(..,1,..) at the end of bcoef.f): the diagonal doubles
    bdiag = [2 * b for b in bdiag]
    return lind, bdiag


def pijump_unwrap(ph):
    """Remove 2 pi jumps along an array (pijump.f applied sequentially)."""
    out = np.array(ph, float)
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        jump = int((abs(d) + np.pi) / (2 * np.pi))
        cands = [d, d - jump * 2 * np.pi, d + jump * 2 * np.pi]
        best = min(cands, key=abs)
        out[i] = out[i - 1] + best
    return out


class PathStage:
    """FEFF's path stage for one phase file; compute(coords, ipots) returns the feffNNNN.dat columns."""

    def __init__(self, phase: PhaseData | str, iorder=2):
        P = read_phase_pad(phase) if isinstance(phase, str) else phase
        if P.nsp != 1:
            raise NotImplementedError("spin-dependent phase files are not ported")
        self.P, self.iorder = P, iorder
        self.ne1 = P.ne1
        self.ltot = P.ltot
        self.lmax = P.lmax[: self.ne1]                                    # (ne1, nph+1)
        self.lmaxp1 = int(self.lmax.max()) + 1
        self.kinit, self.linit = setkap(P.ihole)
        self.ilinit = self.linit + 1
        ll = self.linit + 1
        self.ll = -ll if self.kinit < 0 else ll
        self.caps = P.ph[: self.ne1, self.ll + P.ltot, 0, 0]
        self.xk = P.xk[: self.ne1]
        self.ck = P.ck[: self.ne1]
        self.rkk = P.rkk[: self.ne1, :, 0]
        self.xnlm = snlm(self.ltot + 1, MTOT + 1)
        self.lind, self.bdiag = bcoef_unpolarized(self.kinit, le2=0)

    # -- per-path pieces ------------------------------------------------------------------------------------------
    def _ph(self, ie, l, iph):
        return self.P.ph[ie, l + self.ltot, 0, iph]

    def _fmtrxi(self, lam1x, lam2x, ie, ileg, ilegp, clmi, mlam, nlam, dri, eta, ipot, fmati):
        ilx = self.lmax[ie, ipot[ilegp]] + 1
        gam = np.zeros((self.ltot + 2, MTOT + 2, NTOT + 2), complex)
        gamtl = np.zeros_like(gam)
        xnlm = self.xnlm
        for il in range(1, ilx + 1):
            ll = il - 1
            tl = (np.exp(2j * self._ph(ie, -ll, ipot[ilegp])) - 1) / 2j
            tltl = (np.exp(2j * self._ph(ie, ll, ipot[ilegp])) - 1) / 2j
            tltl = tl * (ll + 1) + tltl * ll
            for lam in range(1, max(lam1x, lam2x) + 1):
                m = mlam[lam]
                if m < 0:
                    continue
                im = m + 1
                if im > il:
                    continue
                in_ = nlam[lam] + 1
                imn = in_ + m
                if lam <= lam1x:
                    gam[il, im, in_] = xnlm[il, im] * (-1) ** m * clmi[ileg][il, imn] if imn <= il else 0.0
                if lam <= lam2x:
                    gamtl[il, im, in_] = tltl / xnlm[il, im] * clmi[ilegp][il, in_]
        for lam1 in range(1, lam1x + 1):
            m1, in1 = mlam[lam1], nlam[lam1] + 1
            iam1 = abs(m1) + 1
            for lam2 in range(1, lam2x + 1):
                m2, in2 = mlam[lam2], nlam[lam2] + 1
                iam2 = abs(m2) + 1
                imn1 = iam1 + in1 - 1
                ilmin = max(iam1, iam2, imn1, in2, 1)
                cterm = 0.0 + 0j
                for il in range(ilmin, ilx + 1):
                    if abs(m1) >= il or abs(m2) >= il:
                        continue
                    cterm += gam[il, iam1, in1] * gamtl[il, iam2, in2] * dri[ilegp][il, m1 + MTOT + 1, m2 + MTOT + 1]
                if eta[ileg] != 0.0:
                    cterm *= np.exp(-1j * eta[ileg] * m1)
                fmati[lam1, lam2, ilegp] = cterm

    def _mmtrxi(self, lam1x, bmati, ie, ileg, ilegp, clmi, mlam, nlam, eta, fmati):
        lind = self.lind
        valid = [lind[k] for k in range(1, 9) if lind[k] >= 0]
        iln, ilx = min(valid) + 1, max(valid) + 1
        gam = np.zeros((self.ltot + 2, MTOT + 2, NTOT + 2), complex)
        gamtl = np.zeros_like(gam)
        xnlm = self.xnlm
        for il in range(iln, ilx + 1):
            tltl = 2 * il - 1
            for lam in range(1, lam1x + 1):
                m = mlam[lam]
                if m < 0:
                    continue
                im = m + 1
                if im > il:
                    continue
                in_ = nlam[lam] + 1
                imn = in_ + m
                gam[il, im, in_] = xnlm[il, im] * (-1) ** m * clmi[ileg][il, imn] if imn <= il else 0.0
                gamtl[il, im, in_] = tltl / xnlm[il, im] * clmi[ilegp][il, in_]
        rkk = self.rkk[ie]
        for lam1 in range(1, lam1x + 1):
            m1, in1 = mlam[lam1], nlam[lam1] + 1
            iam1 = abs(m1) + 1
            for lam2 in range(1, lam1x + 1):
                m2, in2 = mlam[lam2], nlam[lam2] + 1
                iam2 = abs(m2) + 1
                f = 0.0 + 0j
                for k1 in range(1, 9):
                    for k2 in range(1, 9):
                        l1, l2 = lind[k1] + 1, lind[k2] + 1
                        if l1 > 0 and l2 > 0 and iam1 <= l1 and iam2 <= l2:
                            f += bmati[m1 + MTOT, k1, m2 + MTOT, k2] * rkk[k1 - 1] * rkk[k2 - 1] \
                                 * gam[l1, iam1, in1] * gamtl[l2, iam2, in2]
                fmati[lam1, lam2, ilegp] = f * np.exp(-1j * eta[ileg] * m1)

    def _mmtr(self, dri, nleg):
        """Energy-independent termination coefficients, ipol = 0 (mmtr.f): bmati[mu1+MTOT, k1, mu2+MTOT, k2]."""
        bmati = np.zeros((2 * MTOT + 1, 9, 2 * MTOT + 1, 9), complex)
        lxx = min(MTOT, self.ilinit)
        for mu1 in range(-lxx, lxx + 1):
            for mu2 in range(-lxx, lxx + 1):
                for k1 in range(1, 9):
                    l1 = self.lind[k1] + 1
                    if l1 > 0:
                        bmati[mu1 + MTOT, k1, mu2 + MTOT, k1] += self.bdiag[k1] * dri[nleg][l1, mu1 + MTOT + 1, mu2 + MTOT + 1]
        return bmati

    # -- the path -----------------------------------------------------------------------------------------------
    def compute(self, coords, ipots, energies=None):
        """coords (nleg, 3) in A with the absorber first at the origin; ipots (nleg,) with the absorber 0.
        Returns the seven feffNNNN.dat columns (k in 1/A, central-atom phase, |F_eff|, phase, reduction factor,
        mean free path, Re p) plus reff (A), the complex path signal and the leg data."""
        coords = np.asarray(coords, float) / BOHR
        nleg = len(coords)
        nsc = nleg - 1
        rat = np.zeros((3, nleg + 2))
        rat[:, 1:nleg] = coords[1:].T                                     # scatterers 1..nsc
        rat[:, nleg] = coords[0]                                          # absorber last ...
        rat[:, 0] = coords[0]                                             # ... and its copy at 0
        ipot = [0] * (nleg + 1)
        ipot[1:nleg] = [int(i) for i in ipots[1:]]
        ipot[nleg] = int(ipots[0])
        ipot[0] = ipot[nleg]
        ri, beta, eta = pathgeom(rat, nleg)
        reff = ri[1:nleg + 1].sum() / 2
        mlam, nlam, lamx, laml0x, mmaxp1, nmax = setlam(self.iorder, beta, nsc, nleg, self.ilinit)
        dri = {isc: rot3i(self.lmaxp1, mmaxp1, beta[isc], self.ltot) for isc in range(1, nleg + 1)}
        bmati = self._mmtr(dri, nleg)
        ne1 = self.ne1 if energies is None else len(energies)
        ies = range(self.ne1) if energies is None else energies
        cchi = np.zeros(ne1, complex)
        mnmxp1 = mmaxp1 + nmax
        for out_i, ie in enumerate(ies):
            ck = self.ck[ie]
            rho = np.zeros(nleg + 1, complex)
            rho[1:] = ck * ri[1:nleg + 1]
            if abs(ck) <= EPS:
                continue
            clmi = {}
            for ileg in range(1, nleg + 1):
                isc0 = ileg - 1 if ileg > 1 else nleg
                lxp1 = max(self.lmax[ie, ipot[isc0]] + 1, self.lmax[ie, ipot[ileg]] + 1)
                mnp1 = min(lxp1, mnmxp1)
                clmi[ileg] = sclmz(rho[ileg], lxp1, mnp1, self.ltot)
            fmati = np.zeros((LAMTOT + 1, LAMTOT + 1, nleg + 1), complex)
            self._fmtrxi(lamx, laml0x, ie, 2, 1, clmi, mlam, nlam, dri, eta, ipot, fmati)
            if nleg > 2:
                self._fmtrxi(laml0x, lamx, ie, nleg, nleg - 1, clmi, mlam, nlam, dri, eta, ipot, fmati)
            for ilegp in range(2, nsc):
                self._fmtrxi(lamx, lamx, ie, ilegp + 1, ilegp, clmi, mlam, nlam, dri, eta, ipot, fmati)
            pmat = fmati[1:lamx + 1, 1:laml0x + 1, 1].copy()                  # f(2,1)
            for isc in range(2, nleg):
                pmat = fmati[1:lamx + 1, 1:lamx + 1, isc] @ pmat
            self._mmtrxi(laml0x, bmati, ie, 1, nleg, clmi, mlam, nlam, eta, fmati)
            term = fmati[1:laml0x + 1, 1:laml0x + 1, nleg]
            ptrac = np.sum(term * pmat[:laml0x, :laml0x].T)
            srho, prho = rho[1:].sum(), np.prod(rho[1:])
            cfac = np.exp(1j * (srho - 2 * self.xk[ie] * reff)) / prho
            cchi[out_i] = ptrac * cfac
        return self._columns(cchi, reff, ies) | {"reff": reff * BOHR, "cchi": cchi, "ri": ri[1:nleg + 1] * BOHR,
                                                   "beta": beta[1:nleg + 1], "eta": eta[:nleg + 2], "nleg": nleg}

    def _columns(self, cchi, reff, ies):
        """fdtarr.f: the seven columns from the complex path signal (reff in bohr)."""
        ies = np.array(list(ies))
        xk, ck, caps = self.xk[ies], self.ck[ies], self.caps[ies]
        amff, phff = np.abs(cchi), np.where(np.abs(cchi) >= EPS, np.angle(cchi), 0.0)
        phff = pijump_unwrap(phff)
        xlam = np.where(np.abs(ck.imag) > EPS, 1 / ck.imag, 1e10)
        redfac = np.exp(-2 * caps.imag)
        cdelt = pijump_unwrap(2 * caps.real)
        cfms = amff * np.exp(1j * phff) * xk * reff ** 2 * np.exp(2 * reff / xlam) / redfac
        lzero = self.ilinit
        return {"k": xk / BOHR, "real_phc": cdelt + lzero * np.pi, "mag": np.abs(cfms) * BOHR,
                "phase": phff - cdelt - lzero * np.pi, "red_fact": redfac, "lam": xlam * BOHR, "rep": ck.real / BOHR}
