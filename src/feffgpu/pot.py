"""FEFF's potentials stage (feff85exafs POT) without self-consistency: free-atom data (from FEFF's own
Dirac-Fock atom code, structure independent, cached), overlapped Coulomb potentials and densities (Mattheiss
prescription, POT/ovrlp.f, sumax.f), Norman radii (frnrm.f), muffin-tin radii, ground-state exchange and the
interstitial potential and density (istprm.f, movrlp.f, ovp2mt.f, afolp.f, fermi.f). Produces the content of
pot.pad (pad.PotData) for the phase-shift stage. Units: Hartree and bohr. Grid: r_i = exp(-8.8 + 0.05 (i-1)),
251 points, as in FEFF.
"""
from __future__ import annotations
import math
import numpy as np
from .pad import PotData, HART
from .xsph import terp, f32, FA, PI, THIRD

DX, X0 = 0.05, 8.8
NOVP = 40
LX = 4


def xx(j):
    return -X0 + (j - 1) * DX


def rr(j):
    return math.exp(xx(j))


def ii(r):
    return int((math.log(r) + X0) / DX) + 1


RI = np.array([rr(j) for j in range(1, 252)])            # 0-based array of the 251 grid points


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# ------------------------------------------------------------------ free atoms
_ATOMS = {}
TABLE_HELP = ("the free-atom table was not found. It ships with the package (feffgpu/data/atoms); if the environment variable "
              "FEFFGPU_ATOMS is set, it must name a directory that holds Z001.npz .. Z099.npz (or the packed .npz.xz files). "
              "`feffgpu doctor` shows where the table is looked for")
_TABLE_KEYS = ("vcoul", "rho", "dmag", "rhoval", "dgc", "dpc", "adgc", "adpc", "xnvmu", "xnval", "iorb", "eorb", "kappa")


def _from_table(iz, ihole, ispinr):
    """The stored free-atom results (tools/build_atoms_table.py): neutral atom, f electrons frozen."""
    from .atoms_table import load_element, table_status
    T = load_element(iz)
    if T is None:
        if table_status()["files"] == 0:
            raise FileNotFoundError(TABLE_HELP)
        return None
    holes = set(int(h) for h in T["holes"])
    if ihole not in holes or (ispinr and ispinr not in holes):
        return None
    out = {k: T[f"h{ihole}_{k}"] for k in _TABLE_KEYS}
    s02, efrozn, eatom, norb = T[f"h{ihole}_scalars"]
    out.update(s02=float(s02), efrozn=float(efrozn), eatom=float(eatom), norb=int(norb))
    if ispinr:
        out["dgc0"], out["dpc0"] = T[f"h{ispinr}_dgc0"], T[f"h{ispinr}_dpc0"]
        s02i, efrozn_i, eatom_i = T[f"h{ispinr}_initial"]
        out.update(s02=float(s02i), efrozn=float(efrozn_i), eatom=float(eatom_i))
    else:
        out["dgc0"], out["dpc0"] = np.zeros(251), np.zeros(251)
    return out


def atomic_data(iz, ihole, xion, iunf, ispinr=0):
    """FEFF's free-atom Dirac-Fock results for one species: from the stored table (all elements, every occupied
    core hole, neutral, f frozen) or, for ions and unfrozen f electrons, from FEFF's atom code through the library."""
    key = (int(iz), int(ihole), float(xion), int(iunf), int(ispinr))
    if key not in _ATOMS:
        out = _from_table(int(iz), int(ihole), int(ispinr)) if (float(xion) == 0.0 and int(iunf) == 0) else None
        if out is None:
            from .potph_lib import scfdat
            out = scfdat(*key)
        _ATOMS[key] = out
    return _ATOMS[key]


# ------------------------------------------------------------------ overlap (POT/sumax.f, ovrlp.f, frnrm.f)
def sumax(rn, ann, aa2, aasum):
    """Spherical average of a neighbour's function aa2 at distance rn, ann neighbours, added into aasum (both
    1-based in FEFF; here 0-based arrays of 250+ points)."""
    rws = 15.0
    jjchi = ii(rws)
    jtop = ii(rn)
    xbl = 0.0
    topx = xx(jjchi)
    stor = np.zeros(jtop)
    a = lambda j: aa2[j - 1]
    for i in range(1, jtop + 1):
        x = xx(i)
        xint = 0.0
        et = math.exp(x)
        blx = math.log(rn - et)
        if blx >= topx:
            stor[i - 1] = 0.5 * xint * ann / (rn * et)
            continue
        jbl = int(2.0 + 20.0 * (blx + 8.8))
        if jbl < 1:
            jbl = 1
        if jbl >= 2:
            xjbl = float(jbl)
            xbl = 0.05 * (xjbl - 1.0) - 8.8
            g = xbl - blx
            xint = xint + 0.5 * g * (a(jbl) * (2.0 - 20.0 * g) * math.exp(2.0 * xbl) + 20.0 * g * a(jbl - 1) * math.exp(2.0 * (xbl - 0.05)))
        tlx = math.log(rn + et)
        if tlx >= topx:
            jtl = jjchi
        else:
            jtl = int(1.0 + 20.0 * (tlx + 8.8))
            if jtl < jbl:
                fzn = a(jtl) * math.exp(2.0 * (xbl - 0.05))
                fz3 = a(jbl) * math.exp(2.0 * xbl)
                fz2 = fzn + 20.0 * (fz3 - fzn) * (tlx - xbl + 0.05)
                fz1 = fzn + 20.0 * (fz3 - fzn) * (blx - xbl + 0.05)
                xint = 0.5 * (fz1 + fz2) * (tlx - blx)
                stor[i - 1] = 0.5 * xint * ann / (rn * et)
                continue
            xjtl = float(jtl)
            xtl = 0.05 * (xjtl - 1.0) - 8.8
            c = tlx - xtl
            xint = xint + 0.5 * c * (a(jtl) * (2.0 - 20.0 * c) * math.exp(2.0 * xtl) + a(jtl + 1) * 20.0 * c * math.exp(2.0 * (xtl + 0.05)))
        if jtl > jbl:
            while True:
                xint = xint + 0.5 * (a(jbl) * math.exp(2.0 * xbl) + a(jbl + 1) * math.exp(2.0 * (xbl + 0.05))) * 0.05
                jbl = jbl + 1
                if jbl < jtl:
                    xbl = xbl + 0.05
                else:
                    break
        stor[i - 1] = 0.5 * xint * ann / (rn * et)
    aasum[:jtop] += stor
    return aasum


def somm2(dr, dp, dpas, da, rnrm, m, npts):
    """Simpson integration of dp dr^m from 0 to rnrm with FEFF's end corrections (MATH/somm2.f); dr, dp 0-based."""
    mm = m + 1
    d1 = da + mm
    acc = 0.0
    a1 = math.log(rnrm / dr[npts - 3]) / dpas
    a2 = a1 ** 2 / 8.0
    a3 = a1 ** 3 / 12.0
    for i in range(1, npts + 1):
        v = dp[i - 1] * dr[i - 1] ** mm
        if i == 1:
            dc = v * 9.0 / 24.0
        elif i == 2:
            dc = v * 28.0 / 24.0
        elif i == 3:
            dc = v * 23.0 / 24.0
        elif i == npts - 3:
            dc = v * (25.0 / 24.0 - a2 + a3)
        elif i == npts - 2:
            dc = v * (0.5 + a1 - 3 * a2 - a3)
        elif i == npts - 1:
            dc = v * (-1.0 / 24.0 + 5 * a2 - a3)
        elif i == npts:
            dc = v * (-a2 + a3)
        else:
            dc = v
        acc = acc + dc
    acc = dpas * acc
    dd = math.exp(dpas) - 1.0
    db = d1 * (d1 + 1.0) * dd * math.exp((d1 - 1.0) * dpas)
    db = dr[0] * (dr[1] ** m) / db
    dd = (dr[0] ** mm) * (1.0 + 1.0 / (dd * (d1 + 1.0))) / d1
    return acc + dd * dp[0] - db * dp[1]


def frnrm(rho, iz):
    """Norman radius: the sphere holding iz electrons of the overlapped density rho (times 4 pi)."""
    r = lambda i: rr(i)
    s = (9 * rho[0] * r(1) ** 3 + 28 * rho[1] * r(2) ** 3 + 23 * rho[2] * r(3) ** 3) / 480
    dpas, d1 = 0.05, 3.0
    dd = math.exp(dpas) - 1.0
    db = d1 * (d1 + 1.0) * dd * math.exp((d1 - 1.0) * dpas)
    db = r(1) / db
    dd = r(1) * (1.0 + 1.0 / (dd * (d1 + 1.0))) / d1
    s = s + dd * rho[0] * r(1) ** 2 - db * rho[1] * r(2) ** 2
    fl = rho[3] * r(4) ** 3
    fr = rho[4] * r(5) ** 3
    frr = rho[5] * r(6) ** 3
    s = s + (25 * fl + 12 * fr - frr) / 480
    inrm, x = None, 0.0
    for i in range(7, len(rho) + 1):
        fll, fl, fr = fl, fr, frr
        frr = rho[i - 1] * r(i) ** 3
        sumsav = s
        s = s + (13 * (fr + fl) - fll - frr) / 480
        if s >= iz:
            inrm = i - 2
            x = (iz - sumsav) / (s - sumsav)
            break
    if inrm is None:
        raise RuntimeError("FRNRM: could not integrate enough charge")
    rnrm = r(inrm) * (1 + x * 0.05)
    jnrm = int((math.log(rnrm) + X0) / DX) + 2
    i0 = jnrm + 1
    ri = np.array([r(k) for k in range(1, jnrm + 3)])
    xpc = rho[:jnrm + 2] * ri ** 2
    xirf = somm2(ri, xpc, DX, 2.0, rnrm, 0, i0)
    dn1 = xirf - iz
    x2 = x - dn1 / ((1 - x) * xpc[inrm - 1] + x * xpc[inrm])
    if abs(x2 - x) > 0.0001:
        rnrm = r(inrm) * (1 + x2 * 0.05)
        xirf = somm2(ri, xpc, DX, 2.0, rnrm, 0, i0)
        dn2 = xirf - iz
        x = x2 - dn2 * (x2 - x) / (dn2 - dn1)
    return r(inrm) * (1 + x * 0.05)


def ovrlp(iph, iphat, rat, iatph, iz, nat, rho, dmag, rhoval, vcoul):
    """Overlapped Coulomb potential, total and valence densities of unique potential iph from the free atoms of
    all neighbours within 12 bohr of its representative atom; returns vclap, edens, edenvl, rnrm, dmag ratio."""
    vclap = vcoul[:, iph].copy()
    edens = rho[:, iph].copy()
    edenvl = rhoval[:, iph].copy()
    iat = iatph[iph]
    rlapx = 12.0
    for inat in range(nat):
        if inat == iat:
            continue
        rnn = dist(rat[inat], rat[iat])
        if rnn > rlapx:
            continue
        infr = iphat[inat]
        sumax(rnn, 1.0, vcoul[:, infr], vclap)
        sumax(rnn, 1.0, rho[:, infr], edens)
        sumax(rnn, 1.0, rho[:, infr], edenvl)
    rnrm = frnrm(edens, iz[iph])
    dm = np.where(edens > 0, dmag[:, iph] / np.where(edens > 0, edens, 1.0), 0.0)
    return vclap, edens, edenvl, rnrm, dm


# ------------------------------------------------------------------ muffin tins and the interstitial (istprm.f, movrlp.f, ovp2mt.f, sidx.f, fermi.f)
def calcvl(r1, r2, r):
    xl = (r1 ** 2 - r2 ** 2 + r ** 2) / (2 * r)
    h = r1 - xl
    return (PI / 3) * h ** 2 * (3 * r1 - h)


def vbh(rs, xmag):
    """von Barth-Hedin exchange-correlation potential (EXCH/vbh.f), Hartree."""
    if rs > 1000:
        return 0.0
    gamma = 5.129762802484098
    flarge = lambda x: (1 + x ** 3) * math.log(1 + 1 / x) + x / 2 - x * x - 1.0 / 3.0
    epc = -0.0504 * flarge(rs / 30)
    efc = -0.0254 * flarge(rs / 75)
    xmup = -0.0504 * math.log(1.0 + 30.0 / rs)
    vu = gamma * (efc - epc)
    alg = -1.22177412 / rs + vu
    blg = xmup - vu
    return (alg * xmag ** (1.0 / 3.0) + blg) / 2.0


def sidx(rholap, npts, rmt, rnrm):
    imt, inrm = ii(rmt), ii(rnrm)
    imax = imt
    for i in range(imt, npts + 1):
        if rholap[i - 1] <= 1.0e-5:
            break
        imax = i
    if inrm > imax:
        inrm = imax
        rnrm = rr(inrm)
    if imt > imax:
        imt = imax
        rmt = rr(imt)
    return imax, imt, inrm, rmt, rnrm


def fermi(rhoint, vint):
    den = rhoint / (4 * PI)
    rs = (3 / (4 * PI * den)) ** THIRD
    xf = FA / rs
    return vint + xf ** 2 / 2, rs, xf


class Interstitial:
    """The overlap matrix of the muffin-tin spheres (movrlp.f) and its solves (ovp2mt.f)."""

    def __init__(self, nph, nat, iphat, rat, iatph, xnatph, imt, rmt, rnrm, lnear, volint, inters=0):
        ipot = inters % 2
        irav = (inters - ipot) // 2
        ri = RI
        exphx = math.exp(0.025)
        ncp = NOVP * (nph + 1) + 1
        cm = np.eye(ncp)
        cm[:, ncp - 1] = 0.01
        cm[ncp - 1, ncp - 1] = 0.01
        bmat = np.zeros((nph + 1, ncp - 1))
        xn = 0.0
        for ip1 in range(nph + 1):
            iat0 = iatph[ip1]
            ntmp = 1
            if irav == 1:
                rav = (rmt[ip1] + rnrm[ip1]) / 2
            elif irav == 0:
                rav = rnrm[ip1]
            else:
                rav = ri[imt[ip1]]
            if lnear[ip1]:
                rav = ri[imt[ip1]]
            for iat in range(nat):
                if iat == iat0:
                    continue
                ip2 = iphat[iat]
                rnn = dist(rat[iat0], rat[iat])
                if rnn < rmt[ip1] + rmt[ip2]:
                    volint = volint + xnatph[ip1] * ntmp * (calcvl(rmt[ip1], rmt[ip2], rnn) + calcvl(rmt[ip1], rmt[ip2], rnn)) / 2.0
                ix1 = ip1
                if rav + rmt[ip2] > rnn:
                    imin2 = ii(rnn - rav)
                    if imt[ip2] - imin2 >= NOVP - 1:
                        raise RuntimeError("MOVRLP-1: FOLP too big")
                    imin2 = imt[ip2] - NOVP + 1
                    for i2 in range(imin2, imt[ip2] + 1):
                        r1 = ri[i2 - 1] / exphx
                        r2 = ri[i2 - 1] * exphx
                        if i2 == imt[ip2]:
                            r2 = rmt[ip2]
                            r1 = (r1 + 2 * ri[imt[ip2] - 1] - rmt[ip2]) / 2.0
                        if i2 == imt[ip2] - 1:
                            r2 = (r2 + 2 * ri[imt[ip2] - 1] - rmt[ip2]) / 2.0
                        if r2 + rav < rnn:
                            continue
                        if r1 + rav < rnn:
                            xr = (rnn - rav - r1) / (r2 - r1)
                            r1 = rnn - rav
                            temp = (r2 ** 2 - r1 ** 2) / (4 * rnn * rav) * ntmp
                            ind2 = i2 + 1
                            if i2 == imt[ip2]:
                                ind2 = i2 - 1
                            xr = xr * (r2 - ri[i2 - 1]) / (ri[ind2 - 1] - ri[i2 - 1])
                            ix2 = ip2 * NOVP + i2 - imin2 + 1
                            bmat[ix1, ix2 - 1] += f32(temp * (1 - xr))
                            ix2 = ip2 * NOVP + ind2 - imin2 + 1
                            bmat[ix1, ix2 - 1] += f32(temp * xr)
                        else:
                            temp = (r2 ** 2 - r1 ** 2) / (4 * rnn * rav) * ntmp
                            ix2 = ip2 * NOVP + i2 - imin2 + 1
                            bmat[ix1, ix2 - 1] += f32(temp)
                if rmt[ip1] + rmt[ip2] <= rnn:
                    continue
                imin1 = ii(rnn - rmt[ip2])
                imin2 = ii(rnn - rmt[ip1])
                if imt[ip1] - imin1 >= NOVP - 1 or imt[ip2] - imin2 >= NOVP - 1:
                    raise RuntimeError("increase NOVP")
                imin1 = imt[ip1] - NOVP + 1
                imin2 = imt[ip2] - NOVP + 1
                for i1 in range(imin1, imt[ip1] + 1):
                    ri1 = ri[i1 - 1] / exphx
                    ri2 = ri[i1 - 1] * exphx
                    if i1 == imt[ip1]:
                        ri2 = rmt[ip1]
                        ri1 = (ri1 + 2 * ri[imt[ip1] - 1] - rmt[ip1]) / 2.0
                    if i1 == imt[ip1] - 1:
                        ri2 = (ri2 + 2 * ri[imt[ip1] - 1] - rmt[ip1]) / 2.0
                    ix1 = i1 - imin1 + 1 + ip1 * NOVP
                    for i2 in range(imin2, imt[ip2] + 1):
                        r1 = ri[i2 - 1] / exphx
                        r2 = ri[i2 - 1] * exphx
                        if i2 == imt[ip2]:
                            r2 = rmt[ip2]
                            r1 = (r1 + 2 * ri[imt[ip2] - 1] - rmt[ip2]) / 2.0
                        if i2 == imt[ip2] - 1:
                            r2 = (r2 + 2 * ri[imt[ip2] - 1] - rmt[ip2]) / 2.0
                        if r2 + ri2 < rnn:
                            continue
                        temp = calcvl(ri2, r2, rnn) + calcvl(r2, ri2, rnn)
                        if ri1 + r2 > rnn:
                            temp = temp - calcvl(ri1, r2, rnn) - calcvl(r2, ri1, rnn)
                        if ri2 + r1 > rnn:
                            temp = temp - calcvl(ri2, r1, rnn) - calcvl(r1, ri2, rnn)
                        if ri1 + r1 > rnn:
                            temp = temp + calcvl(ri1, r1, rnn) + calcvl(r1, ri1, rnn)
                        temp = temp / (4.0 / 3.0 * PI * (ri2 ** 3 - ri1 ** 3)) * ntmp
                        if r1 + ri2 < rnn:
                            xr = (rnn - ri[i1 - 1] - r1) / (r2 - r1)
                            ind2 = i2 + 1
                            if i2 == imt[ip2]:
                                ind2 = i2 - 1
                            xr = xr * (r2 - ri[i2 - 1]) / (ri[ind2 - 1] - ri[i2 - 1])
                            ix2 = i2 - imin2 + 1 + ip2 * NOVP
                            cm[ix1 - 1, ix2 - 1] += f32(temp * (1 - xr))
                            ix2 = ind2 - imin2 + 1 + ip2 * NOVP
                            cm[ix1 - 1, ix2 - 1] += f32(temp * xr)
                        else:
                            ix2 = i2 - imin2 + 1 + ip2 * NOVP
                            cm[ix1 - 1, ix2 - 1] += f32(temp)
            xn = xn + xnatph[ip1]
        if ipot == 0:
            for iph in range(nph + 1):
                aa = xnatph[iph] / xn
                cm[ncp - 1, :ncp - 1] += f32(aa) * bmat[iph]
        else:
            cm[ncp - 1, :ncp - 1] += bmat[0]
        self.cm, self.ncp, self.nph, self.ipot, self.irav = cm, ncp, nph, ipot, irav
        self.imt, self.inrm_, self.rmt, self.rnrm, self.lnear, self.xnatph = imt, None, rmt, rnrm, lnear, xnatph
        self.volint = volint

    def _rhs(self, vtot, inrm, vint, lrewr):
        ri = RI
        cv = []
        for ip1 in range(self.nph + 1):
            for i in range(1, NOVP + 1):
                ix1 = self.imt[ip1] - NOVP + i
                v = vtot[ix1 - 1, ip1]
                if lrewr == 2:
                    v = v - vint
                cv.append(v)
        vtotav = []
        for ip1 in range(self.nph + 1):
            if self.irav == 1:
                rav = (self.rmt[ip1] + self.rnrm[ip1]) / 2
            elif self.irav == 0:
                rav = self.rnrm[ip1]
            else:
                rav = ri[self.imt[ip1]]
            if self.lnear[ip1]:
                rav = ri[self.imt[ip1]]
            vtotav.append(terp(ri, vtot[:, ip1], inrm[ip1] + 2, 3, rav))
        return np.array(cv), vtotav

    def potential(self, vtot, inrm, vint, lrewr):
        """lrewr 1: estimate vint and rewrite vtot inside the spheres; 2: vint fixed. Returns vint, new vtot."""
        cv, vtotav = self._rhs(vtot, inrm, vint, lrewr)
        ncp = self.ncp
        if lrewr == 1:
            s, bsum = 0.0, 0.0
            nphlst = self.nph if self.ipot == 0 else 0
            for iph in range(nphlst + 1):
                s = s + vtotav[iph] * self.xnatph[iph]
                bsum = bsum + self.xnatph[iph]
            cv = np.concatenate([cv, [s / bsum]])
            x = np.linalg.solve(self.cm, cv)
            vint = x[ncp - 1] / 100.0
        else:
            x = np.linalg.solve(self.cm[:ncp - 1, :ncp - 1], cv)
        out = vtot.copy()
        for iph in range(self.nph + 1):
            for i in range(1, NOVP + 1):
                index1 = self.imt[iph] - NOVP + i
                out[index1 - 1, iph] = x[i + NOVP * iph - 1] + vint
            j = self.imt[iph] + 1
            out[j - 1, iph] = terp(RI, out[:, iph], self.imt[iph], 2, RI[j - 1])
            out[j:, iph] = vint
        return vint, out

    def density(self, edens, qtot, inrm):
        """Charge outside the muffin-tin spheres (ovp2mt with lrewr = 0): returns qtot - charge inside."""
        cv, _ = self._rhs(edens, inrm, 0.0, 0)
        ncp = self.ncp
        x = np.linalg.solve(self.cm[:ncp - 1, :ncp - 1], cv)
        inside = 0.0
        for iph in range(self.nph + 1):
            npts = self.imt[iph] + 2
            crho = np.zeros(npts)
            for i in range(1, npts + 1):
                if i < self.imt[iph] - NOVP + 1:
                    crho[i - 1] = edens[i - 1, iph] * RI[i - 1] ** 2
                elif i <= self.imt[iph]:
                    ix1 = NOVP * iph + i - self.imt[iph] + NOVP
                    crho[i - 1] = x[ix1 - 1] * RI[i - 1] ** 2
                else:
                    crho[i - 1] = terp(RI, crho, self.imt[iph], 2, RI[i - 1])
            inside = inside + self.xnatph[iph] * somm2(RI, crho, 0.05, 0.0, self.rmt[iph], 0, npts)
        return qtot - inside


class _State:
    """Mutable state of the muffin-tin/interstitial calculation (istprm's in/out arguments)."""
    pass


def istprm(S, nph, nat, iphat, rat, iatph, xnatph, folp, iafolp, ixc, qtotel, inters=0, totvol=0.0, first=False):
    """Muffin-tin radii (first call), ground-state exchange added to the potentials, interstitial parameters."""
    if first:
        S.lnear = [False] * (nph + 1)
        S.rmt = np.zeros(nph + 1)
        S.inrm = np.zeros(nph + 1, int)
        S.inn, S.rnnmin = [0] * (nph + 1), [0.0] * (nph + 1)
        big = 5000.0
        for iph in range(nph + 1):
            voltot, rmtavg = 0.0, 0.0
            S.inrm[iph] = ii(S.rnrm[iph])
            iat = iatph[iph]
            rnear = big
            S.rmt[iph] = big
            for inat in range(nat):
                if inat == iat:
                    continue
                rnn = dist(rat[inat], rat[iat])
                inph = iphat[inat]
                if rnn <= rnear:
                    rnear = rnn
                    S.rnnmin[iph] = rnn
                    S.inn[iph] = inph
                if S.rnrm[iph] + S.rnrm[inph] >= rnn:
                    voltmp = calcvl(S.rnrm[iph], S.rnrm[inph], rnn) + calcvl(S.rnrm[inph], S.rnrm[iph], rnn)
                    rmttmp = rnn * folp[iph] * S.rnrm[iph] / (S.rnrm[iph] + S.rnrm[inph])
                    rmtavg = rmtavg + rmttmp * voltmp
                    voltot = voltot + voltmp
            if S.rnrm[iph] >= rnear:
                S.lnear[iph] = True
            if rmtavg <= 0:
                S.rmt[iph] = S.rnrm[iph]
            else:
                S.rmt[iph] = rmtavg / voltot
                if S.rnrm[iph] >= rnear:
                    imax = ii(rnear) - 1
                    while not (S.vclap[imax - 1, iph] < S.vclap[imax, iph]):
                        imax -= 1
                    S.rmt[iph] = math.exp(xx(imax)) - 0.0001
        for iph in range(nph + 1):
            temp = 0.2 + 0.8 * S.rnrm[iph] / S.rmt[iph] if iafolp > 0 else 0.3 + 0.7 * S.rnrm[iph] / S.rmt[iph]
            if temp < S.folpx[iph]:
                S.folpx[iph] = temp
            temp = S.rnnmin[iph] / S.rmt[iph] / 1.06
            if temp < S.folpx[iph]:
                S.folpx[iph] = temp
            temp = math.exp(-(NOVP - 3) * 0.05)
            if S.lnear[iph]:
                temp = S.rnnmin[iph] / (S.rmt[iph] * 1.05 + temp * S.rmt[S.inn[iph]])
                if temp < S.folpx[iph]:
                    S.folpx[iph] = temp
                if temp < S.folpx[S.inn[iph]]:
                    S.folpx[S.inn[iph]] = temp
            else:
                temp = (S.rnnmin[iph] - S.rnrm[iph]) / (temp * S.rmt[S.inn[iph]])
                if temp < S.folpx[S.inn[iph]]:
                    S.folpx[S.inn[iph]] = temp
    S.imt = np.zeros(nph + 1, int)
    S.vtot = np.zeros((251, nph + 1))
    S.vvalgs = np.zeros((251, nph + 1))
    for iph in range(nph + 1):
        imax, imt, inrm, rmt_, rnrm_ = sidx(S.edens[:, iph], 250, S.rmt[iph], S.rnrm[iph])
        S.imt[iph], S.inrm[iph], S.rmt[iph], S.rnrm[iph] = imt, inrm, rmt_, rnrm_
        for i in range(1, imax + 1):
            e = S.edens[i - 1, iph]
            if e <= 0:
                rs, xmag = 100.0, 1.0
            else:
                rs = (e / 3) ** (-THIRD)
                xmag = 1.0
            S.vtot[i - 1, iph] = S.vclap[i - 1, iph] + vbh(rs, xmag)
    rnrmav, xn, volint = 0.0, 0.0, 0.0
    for iph in range(nph + 1):
        rnrmav = rnrmav + xnatph[iph] * S.rnrm[iph] ** 3
        volint = volint - xnatph[iph] * S.rmt[iph] ** 3
        xn = xn + xnatph[iph]
    if totvol <= 0.0:
        volint = 4 * PI / 3 * (volint + rnrmav)
    else:
        volint = 4 * PI / 3 * volint + totvol
    S.rnrmav = (rnrmav / xn) ** THIRD
    I = Interstitial(nph, nat, iphat, rat, iatph, xnatph, S.imt, S.rmt, S.rnrm, S.lnear, volint, inters)
    if I.volint <= 0:
        raise RuntimeError("ISTPRM: no interstitial density")
    S.rhoint = 4 * PI * I.density(S.edens, qtotel, S.inrm) / I.volint
    S.vint, S.vtot = I.potential(S.vtot, S.inrm, 0.0, 1)
    if S.vint >= S.xmu:
        S.vint = S.xmu - 0.05
        _, S.vtot = I.potential(S.vtot, S.inrm, S.vint, 2)
    S.xmunew, S.rs, S.xf = fermi(S.rhoint, S.vint)
    return S


# ------------------------------------------------------------------ driver (POT/pot.f without the self-consistency loop)
def potentials(iz, ihole, xion, iunf, nph, nat, iphat, rat, iatph, xnatph, folp0, iafolp, ixc, nohole=-1, inters=0,
               totvol=0.0, ecv0=-40.0 / HART, title=None, nscmt=0, rfms1=0.0, lfms1=0, ca1=0.2, lmaxsc=None, icoul=0,
               nmix=1, rgrd=f32(0.05), ispec=0, jumprm=0, verbose=False, device="cpu"):
    """rat (nat, 3) bohr; iphat (nat,) potential index; iatph (nph+1,) representative atom of each potential;
    nscmt > 0 runs FEFF's self-consistency loop (SCF card: rfms1 bohr, lfms1, ca1, nscmt)."""
    if str(device).startswith("cuda"):
        # Parts of this calculation take GPU memory outside PyTorch's memory cache. In a process that used the GPU
        # before, that cache can hold nearly all the memory and the calculation then fails with an out-of-memory
        # error; releasing the cached blocks first puts it in the state of a fresh process.
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    iz, xion, xnatph, folp0 = list(iz)[:nph + 1], list(xion)[:nph + 1], list(xnatph)[:nph + 1], list(folp0)[:nph + 1]
    rho = np.zeros((251, nph + 2)); vcoul = np.zeros((251, nph + 2)); dmag = np.zeros((251, nph + 2)); rhoval = np.zeros((251, nph + 2))
    dgc = np.zeros((251, 30, nph + 2)); dpc = np.zeros((251, 30, nph + 2)); adgc = np.zeros((10, 30, nph + 2)); adpc = np.zeros((10, 30, nph + 2))
    xnval = np.zeros((30, nph + 2)); xnvmu = np.zeros((LX + 1, nph + 2)); eorb = np.zeros((30, nph + 2)); kappa = np.zeros((30, nph + 2), int)
    iorb = np.zeros((8, nph + 2), int); norb = np.zeros(nph + 2, int)
    etfin = 0.0
    # with ions (ION card) FEFF runs the free atoms and the overlap twice: first neutral, which fixes the Norman
    # radii, then as ions for everything else (pot.f: nfree = 2)
    nfree = 2 if any(abs(x) > 1.0e-3 for x in xion) else 1
    vclap = np.zeros((251, nph + 1)); edens = np.zeros((251, nph + 1)); edenvl = np.zeros((251, nph + 1)); dmr = np.zeros((251, nph + 1))
    rnrm = np.zeros(nph + 1)
    for ifree in range(nfree):
        xion_p = [0.0] * (nph + 1) if (nfree == 2 and ifree == 0) else xion
        for iph in range(nph + 1):
            itmp = ihole if iph == 0 else 0
            A = atomic_data(iz[iph], itmp, xion_p[iph], iunf)
            vcoul[:, iph], rho[:, iph], dmag[:, iph], rhoval[:, iph] = A["vcoul"], A["rho"], A["dmag"], A["rhoval"]
            dgc[:, :, iph], dpc[:, :, iph], adgc[:, :, iph], adpc[:, :, iph] = A["dgc"], A["dpc"], A["adgc"], A["adpc"]
            xnval[:, iph], xnvmu[:, iph], eorb[:, iph], kappa[:, iph], iorb[:, iph], norb[iph] = A["xnval"], A["xnvmu"], A["eorb"], A["kappa"], A["iorb"], A["norb"]
            if iph == 0:
                etfin, s02 = A["eatom"], A["s02"]
        B = atomic_data(iz[0], 0, xion_p[0], iunf, ispinr=ihole)              # initial state: no hole, keep its spinor
        dgc0, dpc0, etinit, efrozn = B["dgc0"], B["dpc0"], B["eatom"], B["efrozn"]
        s02 = B["s02"] if math.isfinite(B["s02"]) else s02          # informational only (FEFF's estimate of S0^2)
        erelax = -efrozn - (etfin - etinit)
        emu = etfin - etinit
        qnrm_pass = np.zeros(nph + 1)
        for iph in range(nph + 1):
            vclap[:, iph], edens[:, iph], edenvl[:, iph], qnrm_pass[iph], dmr[:, iph] = ovrlp(iph, iphat, rat, iatph, iz, nat, rho, dmag, rhoval, vcoul)
            if iph == 0:
                emu = emu - vclap[0, 0] + vcoul[0, 0]
        if ifree == 0:
            rnrm = qnrm_pass.copy()
    qtotel = sum((iz[iph] - xion[iph]) * xnatph[iph] for iph in range(nph + 1))
    S = _State()
    S.rnrm, S.vclap, S.edens, S.edenvl = rnrm.copy(), vclap, edens, edenvl
    S.rhoval = np.zeros_like(edenvl)
    S.xmu = 100.0
    folp = np.array(folp0, float).copy()
    S.folpx = folp.copy()
    if iafolp >= 0:
        folp[:] = 1.0
    istprm(S, nph, nat, iphat, rat, iatph, xnatph, folp, iafolp, ixc, qtotel, inters, totvol, first=True)
    S.xmu = S.xmunew
    if iafolp >= 0:                                                       # afolp.f
        rmtx = S.rmt / folp
        for iph in range(nph + 1):
            folp[iph] = S.folpx[iph]
            S.rmt[iph] = folp[iph] * rmtx[iph]
        istprm(S, nph, nat, iphat, rat, iatph, xnatph, folp, iafolp, ixc, qtotel, inters, totvol)
        S.xmu = S.xmunew
    wp = math.sqrt(12.0 * S.rs / FA ** 4) * S.xf ** 2 / 2.0
    xnmues = np.zeros((LX + 1, nph + 1))
    qnrm = np.zeros(nph + 1)
    ecv = ecv0
    if nscmt > 0:
        from .scf import SCFRunner
        from .fms import Cluster
        lmaxsc = list(lmaxsc)[:nph + 1] if lmaxsc is not None else [2] * (nph + 1)
        inp = dict(nph=nph, nat=nat, iz=iz, xion=xion, iunf=iunf, ihole=ihole, nohole=nohole, jumprm=jumprm, xnatph=xnatph,
                   adgc=adgc, adpc=adpc, dgc=dgc, dpc=dpc, xnvmu=xnvmu, xnval=xnval, eorb=eorb, kappa=kappa, norb=norb)
        R = SCFRunner(inp, device=device)
        S.qnrm = qnrm.copy()
        qold = np.zeros(nph + 1)
        clusters = [Cluster(iph0, rat, iphat, rfms1, device="cpu") for iph0 in range(nph + 1)]      # the cluster solves are small; keep them on the CPU
        tolq, tolmu = 1.0e-3, 3.0e-3
        restart = True
        while restart:
            restart = False
            ecv = R.corval(S, ecv, rgrd, lmaxsc, verbose)
            xntot = 0.0
            for iph in range(nph + 1):
                xntot = xntot + xnatph[iph] * sum(xnvmu[:, iph])
            if verbose:
                print(f"    : mu_old = {S.xmu * HART:.3f}")
            S.xmunew = S.xmu
            for iscmt in range(1, nscmt + 1):
                vclapp = S.vclap.copy()
                xmu_used = S.xmu
                xnmues = R.scmt(S, iscmt, ecv, xntot, rgrd, lmaxsc, rfms1, lfms1, ca1, nscmt, clusters, verbose)
                xmunew = S.xmu
                lpass = not (iscmt < nscmt and iscmt <= 3)
                if abs(xmunew - xmu_used) > tolmu:
                    lpass = False
                if verbose:
                    print(f" iteration {iscmt}: mu_new = {xmunew * HART:.4f} eV; charge transfer " + " ".join(f"{-S.qnrm[i] + xion[i]:.4f}" for i in range(nph + 1)))
                for iph in range(nph + 1):
                    if abs(S.qnrm[iph] - qold[iph]) > tolq:
                        lpass = False
                    qold[iph] = S.qnrm[iph]
                    sm = -S.qnrm[iph] + sum(xnmues[il, iph] - xnvmu[il, iph] for il in range(LX + 1))
                    if abs(sm) > 0.05:
                        lpass = False
                if iscmt == nscmt or lpass:
                    for ip in range(nph + 1):
                        S.edens[:, ip] = S.edens[:, ip] - S.rhoval[:, ip] + S.edenvl[:, ip]
                    S.vclap = vclapp
                    qnrm = -S.qnrm + np.array(xion[:nph + 1], float)
                    break
                for ip in range(nph + 1):
                    S.edenvl[:, ip] = S.rhoval[:, ip]
                istprm(S, nph, nat, iphat, rat, iatph, xnatph, folp, iafolp, ixc, qtotel, inters, totvol)
                S.xmunew = S.xmu
    return PotData(1, title or [""], nph, nohole, ihole, inters, iafolp, 0, iunf, S.rnrmav, S.xmu, S.vint, S.rhoint, emu, s02, erelax,
                   wp, ecv, S.rs, S.xf, qtotel, totvol, S.imt, S.rmt, S.inrm, np.array(iz[:nph + 1]), kappa[:, 0], S.rnrm, np.array(folp0, float),
                   S.folpx, np.array(xnatph[:nph + 1], float), np.array(xion[:nph + 1], float), dgc0, dpc0, dgc[:, :, :nph + 1],
                   dpc[:, :, :nph + 1], adgc[:, :, :nph + 1], adpc[:, :, :nph + 1], S.edens, S.vclap, S.vtot, S.edenvl, S.vvalgs, dmr,
                   xnval[:, :nph + 1], eorb[:, 0], iorb[:, :nph + 1], qnrm, xnmues)
