"""Pieces of FEFF's self-consistency loop (POT/scmt.f and what it calls), batched over energies in torch:
the embedded-atom density of states of one potential at complex energies (rholie.f), the complex-plane energy
contour (grids.f), the density accumulation (ff2g.f). The full multiple scattering of the cluster is in
fms.py."""
from __future__ import annotations
import math
import numpy as np
import torch
from .xsph import (_Potential, exjlnl, phamp, ALPHFS, ALPINV, PI, X0, CDTYPE, RDTYPE, NRPTX)
from .pad import HART

LX = 4


def csomm2(dr, dp, dpas, da, rnrm, npts):
    """Simpson integration of dp dr from 0 to rnrm with FEFF's end corrections (csomm2.f), dp (E, >= npts)."""
    d1 = da + 1
    a1 = math.log(rnrm / float(dr[npts - 3])) / dpas
    a2 = a1 ** 2 / 8.0
    a3 = a1 ** 3 / 12.0
    acc = torch.zeros(dp.shape[0], dtype=dp.dtype, device=dp.device)
    for i in range(1, npts + 1):
        v = dp[:, i - 1] * float(dr[i - 1])
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
    db = float(dr[0]) / db
    dd = float(dr[0]) * (1.0 + 1.0 / (dd * (d1 + 1.0))) / d1
    return acc + dd * dp[:, 0] - db * dp[:, 1]


_TERP_CACHE = {}


def terpc_rows(ri, y, n, m, x0s):
    """FEFF's terp (order m polynomial through m+1 points) of rows y (E, >= n) at the points x0s, as a weight
    matrix (cached per grid); returns (E, len(x0s))."""
    key = (int(n), int(m), float(ri[0]), float(ri[1]), len(x0s), float(x0s[0]), float(x0s[-1]))
    if key not in _TERP_CACHE:
        from .xsph import _locat
        xs = [float(v) for v in ri[:n]]
        W = np.zeros((len(x0s), n))
        for k, x0 in enumerate(x0s):
            i = _locat(x0, xs)
            kk = min(max(i - m // 2, 1), n - m)
            xa = xs[kk - 1:kk + m]
            for a in range(m + 1):
                w = 1.0
                for b in range(m + 1):
                    if b != a:
                        w *= (x0 - xa[b]) / (xa[a] - xa[b])
                W[k, kk - 1 + a] = w
        _TERP_CACHE[key] = W
    W = torch.tensor(_TERP_CACHE[key], dtype=torch.float64, device=y.device)
    return y[:, :n] @ W.T.to(y.dtype)


def ldos_batched(pots, em, erefs, v_rels, lmaxs, ri05, bessel_orders=8):
    """rholie.f for several potentials at a batch of complex energies with every radial solve in one batched call:
    pots list of _Potential; em (E,) complex; erefs, v_rels (jri+1,), lmaxs per potential. Returns per potential
    (xrhole (E, lx+1), xrhoce (E, lx+1), yrhole (E, 251, lx+1), yrhoce (E, 251), ph (E, lx+1), nr05)."""
    from .xsph import dirac_outward, dirac_inward, _lane_setup
    dev = pots[0].device
    E = em.shape[0]
    ie = torch.arange(E, device=dev)
    lanes = []          # per potential: (lane offset, nl, bases)
    kap_l, ic3_l, ie_l, dv_l, vmb_l, av2_l, p2_l, fl_l, dr_l, rmt_l, jri_l, av1_l, jlast_l = ([] for _ in range(13))
    meta = []
    off = 0
    for ip, P in enumerate(pots):
        lmax = min(lmaxs[ip], LX)
        if P.iz <= 4:
            lmax = 2
        if P.iz <= 2:
            lmax = 1
        nl = lmax + 1
        v = torch.as_tensor(v_rels[ip], dtype=CDTYPE, device=dev)[None, :].expand(E, -1)
        dv, vmb, av2 = P.bases(v)
        inrm = int((math.log(P.rnrm) + X0) / P.hx) + 1
        jnrm = inrm + 1
        nr05 = min(int((math.log(P.rnrm) + X0) / 0.05) + 5, 251)
        ilast = min(int(round((nr05 - 1) * 0.05 / P.hx)) + 1, NRPTX)
        p2 = em - erefs[ip]
        a = (P.dz / 137.0373) ** 2
        for l in range(nl):
            kap = -(l + 1)
            ic3 = 1 if l > 0 else 0
            kap_l.append(torch.full((E,), kap, dtype=torch.long, device=dev))
            ic3_l.append(torch.full((E,), ic3, dtype=torch.long, device=dev))
            ie_l.append(ie)
            dv_l.append(dv)
            vmb_l.append(vmb * (kap + 1.0) / ALPINV)
            av2_l.append(av2)
            p2_l.append(p2)
            fl_l.append(torch.full((E,), math.sqrt((kap * kap - a) + (kap + 1) * ic3), dtype=RDTYPE, device=dev))
            dr_l.append(P.dr[None, :].expand(E, -1))
            rmt_l.append(torch.full((E,), P.rmt, dtype=RDTYPE, device=dev))
            jri_l.append(torch.full((E,), P.jri, dtype=torch.long, device=dev))
            av1_l.append(torch.full((E,), P.av1, dtype=RDTYPE, device=dev))
            jlast_l.append(torch.full((E,), min(ilast, P.np_), dtype=torch.long, device=dev))
        meta.append((off, nl, ilast, jnrm, nr05, p2))
        off += nl * E
    cat = torch.cat
    kap, ic3, p2, fl, rmt, jri, av1, jlast = cat(kap_l), cat(ic3_l), cat(p2_l), cat(fl_l), cat(rmt_l), cat(jri_l), cat(av1_l), cat(jlast_l)
    dv, vm, av2, dr = cat(dv_l), cat(vmb_l), cat(av2_l), cat(dr_l)
    hx, idim = pots[0].hx, pots[0].idim
    iwkb, iflat = _lane_setup(p2, hx, idim, jri, jlast)
    ri = pots[0].ri
    gg, gp, pu, qu = dirac_outward(p2 / 137.0373, p2, kap, ic3, fl, av1, av2, dv, vm, dr, ri, hx, iflat - 1, iwkb - 1, jri - 1,
                                   jlast - 1, rmt, bessel_orders)
    gn, qn = dirac_inward(p2 / 137.0373, p2, kap, ic3, dv, vm, dr, hx, iflat - 1, iwkb - 1, jri - 1, jlast - 1, bessel_orders)
    out = []
    for ip, P in enumerate(pots):
        off, nl, ilast, jnrm, nr05, p2p = meta[ip]
        L = nl * E
        sl = slice(off, off + L)
        ck = torch.sqrt(2 * p2p + (p2p * ALPHFS) * (p2p * ALPHFS))
        xkmt = P.rmt * ck
        lanes_l = torch.arange(L, device=dev) // E
        ck_l = ck.repeat(nl)
        jl = torch.stack([exjlnl(xkmt, l)[0] for l in range(nl)]).reshape(-1)
        nl_ = torch.stack([exjlnl(xkmt, l)[1] for l in range(nl)]).reshape(-1)
        jlp1 = torch.stack([exjlnl(xkmt, l + 1)[0] for l in range(nl)]).reshape(-1)
        nlp1 = torch.stack([exjlnl(xkmt, l + 1)[1] for l in range(nl)]).reshape(-1)
        phx, temp = phamp(P.rmt, pu[sl], qu[sl], ck_l, jl, nl_, jlp1, nlp1, kap[sl])
        xfnorm = 1 / temp
        pr = gg[sl, :ilast] * xfnorm[:, None]
        qr = gp[sl, :ilast] * xfnorm[:, None]
        puf = ck_l * ALPHFS
        puf = -puf / (1 + torch.sqrt(1 + puf * puf))
        tempc = torch.exp(1j * phx)
        j0 = P.jri - 1
        quw = 2 * ALPINV * tempc * (gn[sl, j0] * qr[:, j0] - pr[:, j0] * qn[sl, j0])
        quw = 1 / quw / ck_l
        pn = 1j * pr - tempc[:, None] * gn[sl, :ilast] * quw[:, None]
        qnn = 1j * qr - tempc[:, None] * qn[sl, :ilast] * quw[:, None]
        lll = lanes_l.to(RDTYPE)
        tnorm = (2 * lll + 1.0) / (1 + puf * puf) / PI * ck_l * 2
        xpc = pr * pr + qr * qr
        yr = (terpc_rows(ri, xpc, ilast, 3, ri05[:nr05]) * tnorm[:, None]).reshape(nl, E, nr05)
        xr = (csomm2_batched(ri, xpc, P.hx, 2 * lanes_l + 2, P.rnrm, jnrm + 1) * tnorm).reshape(nl, E)
        xpc2 = pn * pr - 1j * pr * pr + qnn * qr - 1j * qr * qr
        yc = (-tnorm[:, None] * terpc_rows(ri, xpc2, ilast, 3, ri05[:nr05])).reshape(nl, E, nr05).sum(0)
        xc = (-csomm2(ri, xpc2, P.hx, 1.0, P.rnrm, jnrm + 1) * tnorm).reshape(nl, E)
        xrhole = torch.zeros(E, LX + 1, dtype=CDTYPE, device=dev); xrhoce = torch.zeros_like(xrhole)
        yrhole = torch.zeros(E, 251, LX + 1, dtype=CDTYPE, device=dev); yrhoce = torch.zeros(E, 251, dtype=CDTYPE, device=dev)
        ph = torch.zeros(E, LX + 1, dtype=CDTYPE, device=dev)
        xrhole[:, :nl] = xr.T
        xrhoce[:, :nl] = xc.T
        yrhole[:, :nr05, :nl] = yr.permute(1, 2, 0)
        yrhoce[:, :nr05] = yc
        ph[:, :nl] = phx.reshape(nl, E).T
        out.append((xrhole, xrhoce, yrhole, yrhoce, ph, nr05))
    return out


def csomm2_batched(dr, dp, dpas, da, rnrm, npts):
    """csomm2 with a per-row power da (E,) (only the end correction depends on it)."""
    d1 = da.to(RDTYPE) + 1
    a1 = math.log(rnrm / float(dr[npts - 3])) / dpas
    a2 = a1 ** 2 / 8.0
    a3 = a1 ** 3 / 12.0
    w = torch.ones(npts, dtype=RDTYPE, device=dp.device)
    w[0], w[1], w[2] = 9.0 / 24.0, 28.0 / 24.0, 23.0 / 24.0
    w[npts - 4], w[npts - 3], w[npts - 2], w[npts - 1] = 25.0 / 24.0 - a2 + a3, 0.5 + a1 - 3 * a2 - a3, -1.0 / 24.0 + 5 * a2 - a3, -a2 + a3
    acc = torch.zeros(dp.shape[0], dtype=dp.dtype, device=dp.device)
    for i in range(npts):
        acc = acc + dp[:, i] * (float(dr[i]) * float(w[i]))
    acc = dpas * acc
    dd = math.exp(dpas) - 1.0
    db = d1 * (d1 + 1.0) * dd * torch.exp((d1 - 1.0) * dpas)
    db = float(dr[0]) / db
    ddv = float(dr[0]) * (1.0 + 1.0 / (dd * (d1 + 1.0))) / d1
    return acc + ddv * dp[:, 0] - db * dp[:, 1]


def rholie(P: _Potential, em, eref, v_rel, lmaxsc, ri05):
    """Embedded-atom l-resolved densities of one potential at a batch of complex energies (rholie.f). Returns
    xrhole (E, lx+1), xrhoce (E, lx+1), yrhole (E, 251, lx+1), yrhoce (E, 251), ph (E, lx+1), nr05."""
    return ldos_batched([P], em, [float(eref[0].real) if torch.is_tensor(eref) else float(eref)], [v_rel], [lmaxsc], ri05)[0]


def grids(ecv, xmu, negx=80, nflrx=17):
    """The complex-energy contour of the self-consistency loop (POT/grids.f): emg (neg,), step (nflrx,)."""
    eimmin = 1j * 0.05 / HART
    neg1 = (nflrx + 1) // 2
    neg3 = nflrx - 1
    neg2mx = negx - neg1 - neg3
    emg = []
    step = np.zeros(nflrx)
    for i in range(1, neg1 + 1):
        eim = eimmin * i ** 2
        emg.append(ecv + eim)
    step[nflrx - 1] = eim.imag / 4
    de = emg[neg1 - 1].imag / 4
    neg2 = int(round((xmu - ecv) / de))
    neg2 = min(neg2, neg2mx)
    neg2 = max(neg2, neg1)
    de = (xmu - ecv) / neg2
    for i in range(neg1 + 1, neg1 + neg2 + 1):
        emg.append(emg[-1] + de)
    neg = neg1 + neg2 + neg3
    tail = [0j] * neg3
    for i in range(1, neg3 + 1):
        eim = eimmin * (i + 1) ** 2 / 4.0
        if i <= nflrx:
            step[i - 1] = eim.imag / 4
        tail[neg3 - i] = xmu + eim
    return np.array(emg + tail), step


# ------------------------------------------------------------------ Coulomb update, Broyden mixing (POT/coulom.f, broydn.f, ATOM/potslw.f)
RI05 = np.array([math.exp(-8.8 + 0.05 * (i - 1)) for i in range(1, 252)])


def potslw(d, dr, dpas, npts):
    """Coulomb potential of a density d (times r^2... as FEFF passes it) by its 4-point integration (potslw.f)."""
    dv = np.zeros(251)
    dp = np.zeros(251)
    das = dpas / 24.0
    for i in range(npts):
        dv[i] = d[i] * dr[i]
    dlo = math.exp(dpas)
    dlo2 = dlo * dlo
    dp[1] = dr[0] * (d[1] - d[0] * dlo2) / (12.0 * (dlo - 1.0))
    dp[0] = dv[0] / 3.0 - dp[1] / dlo2
    dp[1] = dv[1] / 3.0 - dp[1] * dlo2
    j = npts - 1
    for i in range(3, j + 1):
        dp[i - 1] = dp[i - 2] + das * (13.0 * (dv[i - 1] + dv[i - 2]) - (dv[i - 3] + dv[i]))
    dp[npts - 1] = dp[j - 1]
    dv[j - 1] = dp[j - 1]
    dv[npts - 1] = dp[j - 1]
    for i in range(3, j + 1):
        k = npts + 1 - i
        dv[k - 1] = dv[k] / dlo + das * (13.0 * (dp[k] / dlo + dp[k - 1]) - (dp[k + 1] / dlo2 + dp[k - 2] * dlo))
    dv[0] = dv[2] / dlo2 + dpas * (dp[0] + 4.0 * dp[1] / dlo + dp[2] / dlo2) / 3.0
    for i in range(npts):
        dv[i] = dv[i] / dr[i]
    return dv


def fab(aa, bb, r0, r1, r2):
    a2 = (r2 ** 2 - r1 ** 2) / 2.0
    a3 = (r2 ** 3 - r1 ** 3) / 3.0
    a4 = (r2 ** 4 - r1 ** 4) / 4.0
    return aa * (a4 / r0 - a3) + bb * (a3 / r0 - a2)


def coulom(nph, ilast, rhoval, edenvl, edens, iz, vclap):
    """New overlapped Coulomb potential from the change of the valence density, normalised by the Norman picture
    (coulom.f with icoul = 0). Arrays (251, nph+1); returns the updated vclap."""
    from .pot import frnrm
    vclap = vclap.copy()
    for ip in range(nph + 1):
        n = ilast[ip]
        drho = np.zeros(251)
        drho[:n] = (rhoval[:n, ip] - edenvl[:n, ip]) * RI05[:n] ** 2
        dvcl = potslw(drho, RI05, 0.05, n)
        dvcl[n:] = 0.0
        rnrm1 = frnrm(edens[:, ip], iz[ip])
        drho2 = edens[:, ip] - edenvl[:, ip] + rhoval[:, ip]
        rnrm2 = frnrm(drho2, iz[ip])
        rmin = min(rnrm1, rnrm2)
        inrm = int((math.log(rmin) + 8.8) / 0.05) + 1
        r0 = RI05[inrm - 1]
        delv = 0.0
        if rnrm2 > rnrm1:
            aa = (drho2[inrm] - drho2[inrm - 1]) / (RI05[inrm] - RI05[inrm - 1])
            bb = drho2[inrm - 1] - aa * RI05[inrm - 1]
            delv = delv - fab(aa, bb, r0, rnrm1, rnrm2)
        else:
            aa = (edens[inrm - 1, ip] - edens[inrm, ip]) / (RI05[inrm] - RI05[inrm - 1])
            bb = -edens[inrm - 1, ip] - aa * RI05[inrm - 1]
            delv = delv - fab(aa, bb, r0, rnrm2, rnrm1)
        aa = (drho2[inrm] - drho2[inrm - 1] + edens[inrm - 1, ip] - edens[inrm, ip]) / (RI05[inrm] - RI05[inrm - 1])
        bb = drho2[inrm - 1] - edens[inrm - 1, ip] - aa * RI05[inrm - 1]
        delv = delv - fab(aa, bb, r0, r0, rmin)
        dvnrm = delv - dvcl[inrm - 1]
        vclap[:n, ip] = vclap[:n, ip] + dvcl[:n] + dvnrm
        vclap[n:, ip] = 0.0
    return vclap


class Broyden:
    """Density mixing between self-consistency iterations (broydn.f), with its saved history."""

    def __init__(self, nbr=30):
        self.nbr = nbr
        self.cmi = np.zeros((nbr + 1, nbr + 1))
        self.frho, self.urho = {}, {}
        self.xnorm = np.zeros(nbr + 1)
        self.wt = RI05 ** 3
        self.rhoold = None

    def mix(self, iscmt, ca, nph, xnvmu, ilast, xnatph, rnrm, qnrm, edenvl, rhoval):
        """Returns the mixed valence density (251, nph+1), the updated qnrm and dq."""
        from .pot import somm2
        n1 = nph + 1
        if self.rhoold is None:
            self.rhoold = np.zeros((251, n1))
        f = np.zeros((251, n1))
        for iph in range(n1):
            n = ilast[iph]
            f[:n, iph] = rhoval[:n, iph] * RI05[:n] - edenvl[:n, iph] * self.wt[:n]
        self.frho[iscmt] = f
        dq = np.zeros(n1)
        xnferm = 0.0
        for ip in range(n1):
            dq[ip] = sum(xnvmu[il, ip] for il in range(5))
            xnferm = xnferm + dq[ip] * xnatph[ip]
        if iscmt > 1:
            xn = 0.0
            for iph in range(n1):
                n = ilast[iph]
                xn = xn + np.sum((self.frho[iscmt][:n, iph] - self.frho[iscmt - 1][:n, iph]) ** 2)
            self.xnorm[iscmt] = xn
            for j in range(2, iscmt + 1):
                c = 0.0
                for iph in range(n1):
                    n = ilast[iph]
                    c = c + np.sum(self.frho[iscmt][:n, iph] * (self.frho[j][:n, iph] - self.frho[j - 1][:n, iph]))
                self.cmi[iscmt, j] = c / self.xnorm[j]
            u = np.zeros((251, n1))
            for iph in range(n1):
                n = ilast[iph]
                u[:n, iph] = ca * (self.frho[iscmt][:n, iph] - self.frho[iscmt - 1][:n, iph]) + (edenvl[:n, iph] - self.rhoold[:n, iph]) * self.wt[:n]
            for j in range(2, iscmt):
                for iph in range(n1):
                    n = ilast[iph]
                    u[:n, iph] = u[:n, iph] - self.urho[j][:n, iph] * (self.cmi[iscmt, j] - self.cmi[iscmt - 1, j])
            self.urho[iscmt] = u
        out = np.zeros((251, n1))
        for iph in range(n1):
            n = ilast[iph]
            self.rhoold[:n, iph] = edenvl[:n, iph]
            out[:n, iph] = edenvl[:n, iph] + ca * self.frho[iscmt][:n, iph] / self.wt[:n]
            for j in range(2, iscmt + 1):
                out[:n, iph] = out[:n, iph] - self.cmi[iscmt, j] * self.urho[j][:n, iph] / self.wt[:n]
        dqav, xnat = 0.0, 0.0
        qnrm = np.array(qnrm, float).copy()
        for iph in range(n1):
            jnrm = int((math.log(rnrm[iph]) + 8.8) / 0.05) + 2
            i0 = jnrm + 1
            xpc = np.zeros(251)
            n = ilast[iph]
            xpc[:n] = out[:n, iph] * RI05[:n] ** 2
            xirf = somm2(RI05, xpc, 0.05, 2.0, rnrm[iph], 0, i0)
            dq[iph] = xirf - qnrm[iph] - dq[iph]
            dqav = dqav + xnatph[iph] * dq[iph]
            xnat = xnat + xnatph[iph]
        aa = dqav / xnferm
        dqav = dqav / xnat
        for iph in range(n1):
            dq[iph] = dq[iph] - dqav
            qnrm[iph] = qnrm[iph] + dq[iph]
            n = ilast[iph]
            out[:n, iph] = out[:n, iph] - aa * edenvl[:n, iph]
        return out, qnrm, dq


# ------------------------------------------------------------------ one self-consistency iteration (POT/scmt.f, ff2g.f)
class SCFRunner:
    """State that FEFF keeps across calls of scmt (ient, the Broyden history) and the pieces of one iteration."""

    def __init__(self, pot_inputs, device="cpu", lookahead=8):
        self.ient = 0
        self.device = device
        self.broyden = Broyden()
        self.inp = pot_inputs
        self.lookahead = lookahead

    def _prepare(self, S, iph, rgrd, x0=8.8):
        """fixvar + referencing as scmt does; returns a _Potential-like solver object and v_rel, eref."""
        from .xsph import fixvar, _Potential
        inp = self.inp
        P = _Potential.__new__(_Potential)
        P.iph, P.rmt, P.rnrm, P.iz, P.device = iph, float(S.rmt[iph]), float(S.rnrm[iph]), int(inp["iz"][iph]), self.device
        ri, vtotph, rhoph, dmagx, P.jrinew = fixvar(P.rmt, S.edens[:, iph], S.vtot[:, iph], np.zeros(251), S.vint, S.rhoint, 0.05, rgrd, inp["jumprm"])
        P.ri = torch.tensor(ri, dtype=RDTYPE, device=self.device)
        P.vtotph = torch.tensor(vtotph, dtype=RDTYPE, device=self.device)
        P.rhoph = torch.tensor(rhoph, dtype=RDTYPE, device=self.device)
        P.imt = int((math.log(P.rmt) + x0) / rgrd) + 1
        P.jri = P.imt + 1
        itmp = inp["ihole"] if (iph == 0 and inp["nohole"] < 0) else 0
        from .xsph import getorb, nucdec_grid, origin_density_term, f32, CL_W, NRPTX
        orb = getorb(P.iz, itmp, float(inp["xion"][iph]), inp["iunf"])
        P.nk, P.xnel = orb["nk"], orb["xnel"]
        P.idim = 1 + int(round(250 * 0.05 / rgrd))
        if P.idim > NRPTX:
            P.idim = NRPTX
        if P.idim % 2 == 0:
            P.idim -= 1
        P.np_ = 1 + int((f32(8.8) + f32(math.log(10.0))) / rgrd)
        if P.idim < P.np_:
            P.np_ = P.idim
        P.dr = torch.tensor(nucdec_grid(P.iz, rgrd, P.idim), dtype=RDTYPE, device=self.device)
        P.dz = float(P.iz)
        P.av1 = (-P.dz) / CL_W
        P.ap1 = origin_density_term(inp["adgc"][:, :, iph], inp["adpc"][:, :, iph], P.nk, P.xnel, P.dz, float(P.dr[0]))
        P.hx = rgrd
        jri = int((math.log(P.rmt) + x0) / rgrd) + 2
        jri1 = jri + 1
        eref = float(vtotph[jri1 - 1])
        v_rel = vtotph[:jri1] - eref
        return P, v_rel, eref, itmp

    def ldos(self, S, ees, rgrd, lmaxsc):
        """rholie for all potentials at a batch of energies: xrhole, xrhoce (E, lx+1, nph+1), yrhole (E, 251, lx+1, nph+1),
        yrhoce (E, 251, nph+1), ph (E, lx+1, nph+1), nr05 (nph+1,), eref (nph+1,)."""
        inp = self.inp
        nph = inp["nph"]
        E = len(ees)
        em = torch.tensor(np.asarray(ees), dtype=CDTYPE, device=self.device)
        xl = torch.zeros(E, LX + 1, nph + 1, dtype=CDTYPE); xc = torch.zeros_like(xl)
        yl = torch.zeros(E, 251, LX + 1, nph + 1, dtype=CDTYPE); yc = torch.zeros(E, 251, nph + 1, dtype=CDTYPE)
        ph = torch.zeros(E, LX + 1, nph + 1, dtype=CDTYPE)
        nr05 = np.zeros(nph + 1, int)
        erefs = np.zeros(nph + 1)
        pots, vrels = [], []
        for iph in range(nph + 1):
            P, v_rel, eref, itmp = self._prepare(S, iph, rgrd)
            P.bessel_orders = 8
            erefs[iph] = eref
            pots.append(P)
            vrels.append(v_rel)
        res = ldos_batched(pots, em, list(erefs), vrels, list(lmaxsc), RI05)
        for iph, (a, b, c, d, e, n) in enumerate(res):
            xl[:, :, iph], xc[:, :, iph], yl[:, :, :, iph], yc[:, :, iph], ph[:, :, iph], nr05[iph] = a.cpu(), b.cpu(), c.cpu(), d.cpu(), e.cpu(), n
        return xl, xc, yl, yc, ph, nr05, erefs


    def scmt(self, S, iscmt, ecv, xnferm, rgrd, lmaxsc, rfms1, lfms1, ca1, nscmt, clusters, verbose=False):
        """One self-consistency iteration (scmt.f): new Fermi level, valence densities (mixed), Coulomb potential,
        electron counts. Updates S in place (edens, edenvl, vclap, xmu, qnrm) and returns xnmues (lx+1, nph+1)."""
        inp = self.inp
        nph, nat = inp["nph"], inp["nat"]
        xnatph, iunf = inp["xnatph"], inp["iunf"]
        n1 = nph + 1
        self.ient += 1
        nflrx = 17
        emg, step = grids(ecv, S.xmu)
        neg = len(emg)
        rhoval = np.zeros((251, n1))
        xrhoce = np.zeros((LX + 1, n1), complex)
        xnmues = np.zeros((LX + 1, n1))
        yrhoce = np.zeros((251, n1), complex)
        iflr = iflrp = nflrx
        ie = 0
        xndifp = xndif = 0.0
        ee = complex(emg[0])
        ep = complex(ee.real)
        upok, idir = False, 1
        # the fixed part of the contour, evaluated in one batch
        nfix = nflrx if self.ient == 1 else neg
        cache = {}

        def evaluate(energies):
            xl, xc, yl, yc, ph, nr05, erefs = self.ldos(S, energies, rgrd, lmaxsc)
            E = len(energies)
            gtr = np.zeros((E, LX + 1, n1), complex)
            if rfms1 > 0:
                ck = torch.sqrt(2 * (torch.tensor(np.asarray(energies), dtype=CDTYPE) - erefs[nph]))
                for iph0 in range(n1):
                    from .fms import fms
                    gtr[:, :, iph0] = fms(clusters[iph0], ck, lmaxsc, ph, iph0, lfms1).cpu().numpy()
            for k, e in enumerate(energies):
                cache[complex(e)] = (xl[k].numpy(), xc[k].numpy(), yl[k].numpy(), yc[k].numpy(), gtr[k], nr05)
            return nr05

        nr05 = evaluate(list(emg[:nfix]))
        fl = fr = 0j
        a_fermi = 0.0
        while True:
            ie += 1
            xrhocp = xrhoce.copy()
            yrhocp = yrhoce.copy()
            if complex(ee) not in cache:
                # the walk moves on the current floor in steps of step[iflr-1]: evaluate a short line of candidate
                # points around ee in one batch (every point FEFF would visit is still computed exactly)
                st = step[iflr - 1]
                line = [ee] + [ee + k * st for k in range(1, self.lookahead + 1)] + [ee - k * st for k in range(1, self.lookahead + 1)]
                evaluate([e for e in line if complex(e) not in cache])
            xrhole_e, xrhoce_e, yrhole_e, yrhoce_e, gtr_e, nr05 = cache[complex(ee)]
            xrhoce[:, :] = xrhoce_e
            yrhoce[:, :] = yrhoce_e
            xntot = 0.0
            fl = fr = 0j
            for iph in range(n1):
                cchi = gtr_e[:, iph]
                xrhoce[:, iph] = xrhoce[:, iph] + cchi * xrhole_e[:, iph]
                if ie == 1:
                    xrhocp[:, iph] = xrhoce[:, iph]
                dl = ee - ep
                dr_ = dl
                if iflr == 1:
                    dr_ = dr_ - 1j * 2 * ee.imag
                if iflrp == 1:
                    dl = dl + 1j * 2 * ep.imag
                for il in range(LX + 1):
                    if il <= 2 or iunf != 0:
                        fl = fl + 2 * xrhocp[il, iph] * xnatph[iph]
                        fr = fr + 2 * xrhoce[il, iph] * xnatph[iph]
                        xnmues[il, iph] = xnmues[il, iph] + (xrhoce[il, iph] * dr_ + xrhocp[il, iph] * dl).imag
                        xntot = xntot + xnmues[il, iph] * xnatph[iph]
                n = nr05[iph]
                for il in range(LX + 1):
                    if il <= 2 or iunf != 0:
                        yrhoce[:n, iph] = yrhoce[:n, iph] + cchi[il] * yrhole_e[:n, il, iph]
                if ie == 1:
                    yrhocp[:, iph] = yrhoce[:, iph]
                rhoval[:n, iph] = rhoval[:n, iph] + (yrhoce[:n, iph] * dr_ + yrhocp[:n, iph] * dl).imag
            if ie != 1:
                xndifp = xndif
            xndif = xntot - xnferm
            if verbose and (ie == 1 or ie % 20 == 0):
                print(f"     point # {ie}  energy = {ee.real * HART:.3f}  count {xntot:.4f}")
            if (ie < neg and self.ient > 1) or (self.ient == 1 and ie < nflrx):
                ep = ee
                ee = complex(emg[ie])
                if ie == neg - 1:
                    iflrp, iflr = 2, 1
                continue
            elif self.ient == 1 and ie == nflrx:
                upok, idir = False, 1
                if xntot > xnferm:
                    idir = -1
                ep = ee
                ee = ee + idir * step[iflr - 1]
                continue
            elif self.ient > 1 and ie == neg:
                upok = True
                iflrp = iflr = 1
                idir = -1
                if xntot < xnferm:
                    idir = 1
                ep = ee
                ee = ee + idir * step[iflr - 1]
                continue
            else:
                if iflrp == 1 and iflr == 1 and xndifp * xndif <= 0.0:
                    if xndif == 0:
                        xmunew = ee.real
                        a_fermi = 0.0
                    else:
                        a = xndif / (xndif - xndifp)
                        for _ in range(4):
                            fxa = a * fl + (1 - a) * fr
                            bb = ((ep - ee) * (fr + fxa) / 2 + 1j * ee.imag * (fr - fl)).imag
                            xndif1 = xndif + a * bb
                            a = a - xndif1 / bb
                        xmunew = ((1 - a) * ee + a * ep).real
                        a_fermi = a
                    break
                if iflr == iflrp:
                    if xndifp * xndif <= 0:
                        upok = False
                        iflrp = iflr
                        iflr = iflr - 1
                        ep = ee
                        ee = complex(ee.real, 4 * step[iflr - 1])
                    elif abs(xndif) > 10.0 * abs(xndif - xndifp) and upok:
                        iflrp = iflr
                        if iflr < nflrx:
                            iflr = iflr + 1
                            ep = ee
                            ee = complex(ee.real, 4 * step[iflr - 1])
                        else:
                            ep = ee
                            ee = ee + idir * step[iflr - 1]
                    else:
                        ep = ee
                        ee = ee + idir * step[iflr - 1]
                else:
                    idir = -1
                    if xndif < 0:
                        idir = 1
                    iflrp = iflr
                    ep = ee
                    ee = ee + idir * step[iflr - 1]
                continue
        a = a_fermi
        for iph in range(n1):
            for il in range(LX + 1):
                if il <= 2 or iunf != 0:
                    fl_ = xrhocp[il, iph] * 2
                    fr_ = xrhoce[il, iph] * 2
                    fxa = a * fl_ + (1 - a) * fr_
                    bb = ((ep - ee) * (fr_ + fxa) / 2 + 1j * ee.imag * (fr_ - fl_)).imag
                    xnmues[il, iph] = xnmues[il, iph] + a * bb
            n = nr05[iph]
            fl_ = yrhocp[:n, iph] * 2
            fr_ = yrhoce[:n, iph] * 2
            fxa = a * fl_ + (1 - a) * fr_
            bb = ((ep - ee) * (fr_ + fxa) / 2 + 1j * ee.imag * (fr_ - fl_)).imag
            rhoval[:n, iph] = rhoval[:n, iph] + a * bb
        if verbose:
            print(f"     Fermi level found after {ie} points: mu_new = {xmunew * HART:.4f} eV; counts " +
                  " ".join(f"{xnmues[il, ip]:.3f}" for ip in range(n1) for il in range(3)))
        S.xmu = xmunew
        rhoval, S.qnrm, dq = self.broyden.mix(iscmt, ca1, nph, inp["xnvmu"], nr05, xnatph, S.rnrm, S.qnrm, S.edenvl, rhoval)
        S.vclap = coulom(nph, nr05, rhoval, S.edenvl, S.edens, inp["iz"], S.vclap)
        for ip in range(n1):
            n = nr05[ip]
            S.edens[:n, ip] = S.edens[:n, ip] - S.edenvl[:n, ip] + rhoval[:n, ip]
            S.edens[n:, ip] = 0.0
            S.edenvl[n:, ip] = 0.0
        S.rhoval = rhoval
        S.nr05 = nr05
        return xnmues

    def corval(self, S, ecv, rgrd, lmaxsc, verbose=False):
        """Core-valence separation check (corval.f): may lower ecv and move core levels to valence (updating xnvmu,
        edenvl, xnval). Returns ecv."""
        inp = self.inp
        nph, iunf = inp["nph"], inp["iunf"]
        n1 = nph + 1
        tol = 5.0 / HART
        if S.vint - ecv < tol:
            ecv = S.vint - tol
        elow, ehigh = -70.0 / HART, -20.0 / HART
        eimag = 1j * 1.5 / HART
        ne = 1 + int(round((ehigh - elow) * 2 * HART))
        de = (ehigh - elow) / (ne - 1)
        eldos = np.zeros((LX + 1, n1))
        iiorb = np.zeros((LX + 1, n1), int)
        ival = np.zeros((LX + 1, n1), int)
        for iph in range(n1):
            for iorb in range(1, inp["norb"][iph] + 1):
                e = inp["eorb"][iorb - 1, iph]
                if ehigh - tol > e > elow:
                    k = inp["kappa"][iorb - 1, iph]
                    lll = -k - 1
                    if lll < 0:
                        lll = k
                    if (71 <= inp["iz"][iph] <= 73) and lll == 3:
                        continue
                    if iunf == 0 and lll == 3:
                        continue
                    eldos[lll, iph] = e
                    ival[lll, iph] = 1
                    if inp["xnval"][iorb - 1, iph] < 0.1:
                        ival[lll, iph] = -1
                    iiorb[lll, iph] = iorb
        xx_im = eimag.imag
        for iph in range(n1):
            xpeak = [(2 * il + 1.0) / (6 * xx_im * PI) for il in range(LX + 1)]
            xp = [0.0] * (LX + 1)
            ifound = [0 if ival[il, iph] != 0 else 1 for il in range(LX + 1)]
            if sum(ifound) == LX + 1:
                continue
            energies = [elow + de * (ie - 1) + eimag for ie in range(1, ne + 1)]
            P, v_rel, eref, itmp = self._prepare(S, iph, rgrd)
            em = torch.tensor(energies, dtype=CDTYPE, device=self.device)
            _, xc, _, _, _, _ = rholie(P, em, torch.full((ne,), eref, dtype=CDTYPE, device=self.device), v_rel, lmaxsc[iph], RI05)
            xc = xc.cpu().numpy()
            for ie in range(1, ne + 1):
                for il in range(LX + 1):
                    if ival[il, iph] != 0 and ifound[il] == 0:
                        v = xc[ie - 1, il].imag
                        if (ie == ne or v < xp[il]) and xp[il] > xpeak[il]:
                            ifound[il] = 1
                            eldos[il, iph] = elow + de * (ie - 2)
                        else:
                            xp[il] = v
                if sum(ifound) == LX + 1:
                    break
            if sum(ifound) < LX + 1:
                raise RuntimeError("CORVAL-1")
        en, icv, ll, ip = [], [], [], []
        for iph in range(n1):
            for il in range(LX + 1):
                if eldos[il, iph] < 0:
                    inew = len(en)
                    for k in range(len(en)):
                        if en[k] > eldos[il, iph] and inew == len(en):
                            inew = k
                    en.insert(inew, eldos[il, iph]); icv.insert(inew, ival[il, iph]); ll.insert(inew, il); ip.insert(inew, iph)
        nlev = len(en)
        if nlev == 0:
            return ecv
        ic, iv = 0, nlev + 1
        for k in range(1, nlev + 1):
            if icv[k - 1] == -1:
                ic = k
            elif k < iv:
                iv = k

        def to_valence(k):
            iph_, l_ = ip[k - 1], ll[k - 1]
            icv[k - 1] = 1
            ival[l_, iph_] = 1
            inp["xnvmu"][l_, iph_] += 4 * l_ + 2
            iorb = iiorb[l_, iph_]
            dgc, dpc = inp["dgc"][:, :, iph_], inp["dpc"][:, :, iph_]
            S.edenvl[:, iph_] += 2 * (l_ + 1) * (dgc[:, iorb - 1] ** 2 + dpc[:, iorb - 1] ** 2) / RI05 ** 2
            if l_ != 0:
                S.edenvl[:, iph_] += 2 * l_ * (dgc[:, iorb - 2] ** 2 + dpc[:, iorb - 2] ** 2) / RI05 ** 2

        for k in range(iv + 1, ic + 1):
            if icv[k - 1] < 0:
                to_valence(k)
        ic = iv - 1
        ok = False
        if ic > 0:
            if iv <= nlev:
                ok = (ecv - en[ic - 1] > tol) and (en[iv - 1] - ecv > tol)
            else:
                ok = ecv - en[ic - 1] > tol
        elif iv <= nlev:
            ok = en[iv - 1] - ecv > tol
        if not ok:
            while True:
                ecv = S.vint - tol
                if iv <= nlev:
                    ecv = min(ecv, en[iv - 1] - tol)
                if ic == 0 or ecv - en[ic - 1] > tol:
                    break
                ic -= 1
                iv -= 1
                to_valence(iv)
            for k in range(iv, nlev + 1):
                iph_, lll = ip[k - 1], ll[k - 1]
                iorb = iiorb[lll, iph_]
                if inp["xnval"][iorb - 1, iph_] < 0.1:
                    inp["xnval"][iorb - 1, iph_] = 2 * lll + 2
                    if lll > 0:
                        inp["xnval"][iorb - 2, iph_] = 2 * lll
        if verbose:
            print(f"     core-valence separation: ecv = {ecv * HART:.3f} eV, levels examined {[(round(e * HART, 1), c) for e, c in zip(en, icv)]}")
        return ecv
