"""Full multiple scattering of a cluster for the self-consistency loop (feff85exafs FMS: yprep.f, xstaff.f,
fmspack.f, gglu.f), in double precision and batched over energies. For a central atom and all atoms within rfms,
builds the free propagator G0 between all (atom, l, m) states from Rehr-Albers polynomials and rotation
matrices, the diagonal t-matrix from the phase shifts, solves (1 - G0 T) X = G0 for the central atom's block and
returns gtr(l) = sum_m X_(lm,lm) exp(2 i delta_l) / (2l+1), the quantity ff2g multiplies into the embedded-atom
densities. FEFF does this in single precision; expect agreement at the 1e-6 level."""
from __future__ import annotations
import math
import numpy as np
import torch
from .xsph import CDTYPE, RDTYPE

LX = 4
NCLUSX = 100


def atheap(rat, iphat):
    """FEFF's heap sort of the cluster by distance from the centre, with its tie-breaking (xstaff.f: atheap).
    rat (nat, 3) relative to the centre; returns the sorted copies."""
    nat = len(rat)
    rat = [list(map(float, r)) for r in rat]
    iphat = list(map(int, iphat))
    ra = [float(np.float32(np.float32(r[0]) ** 2 + np.float32(r[1]) ** 2 + np.float32(r[2]) ** 2)) + (i + 1) * 1e-8 for i, r in enumerate(rat)]
    if nat < 2:
        return rat, iphat
    inorder = all(ra[i] >= ra[i - 1] for i in range(1, nat))
    if inorder:
        return rat, iphat
    # 1-based heapsort exactly as the Fortran
    R = [None] + rat
    I = [None] + iphat
    A = [None] + ra
    l = nat // 2 + 1
    ir = nat
    while True:
        if l > 1:
            l -= 1
            toss, itoss, dum = R[l], I[l], A[l]
        else:
            toss, itoss, dum = R[ir], I[ir], A[ir]
            R[ir], I[ir], A[ir] = R[1], I[1], A[1]
            ir -= 1
            if ir == 1:
                R[1], I[1], A[1] = toss, itoss, dum
                break
        i, j = l, l + l
        while j <= ir:
            if j < ir and A[j] < A[j + 1]:
                j += 1
            if dum < A[j]:
                R[i], I[i], A[i] = R[j], I[j], A[j]
                i, j = j, j + j
            else:
                j = ir + 1
        R[i], I[i], A[i] = toss, itoss, dum
    return R[1:], I[1:]


def getang(rat, i, j):
    tiny, pi = 1.0e-7, 3.141592654
    x, y, z = rat[i][0] - rat[j][0], rat[i][1] - rat[j][1], rat[i][2] - rat[j][2]
    r = math.sqrt(x * x + y * y + z * z)
    phi = theta = 0.0
    if i != j:
        if abs(x) < tiny:
            phi = 0.0 if abs(y) < tiny else (pi / 2 if y > tiny else -pi / 2)
        else:
            phi = math.atan2(y, x)
        if r > tiny:
            if z <= -r:
                theta = pi
            elif z < r:
                theta = math.acos(z / r)
    return theta, phi


def rotmat_small(betax, lxp1, mxp1):
    """d^l_{m n}(beta) for l < lxp1 by rotxan's recursion; returns dri0[l][m+l][n+l] (0-based lists)."""
    lxx = 24
    d = np.zeros((lxx + 1, 2 * lxx + 1, 2 * lxx + 1))
    nm = mxp1
    ndm = lxp1 + nm - 1
    xc, xs, s = math.cos(betax / 2), math.sin(betax / 2), math.sin(betax)
    d[0, 0, 0] = 1
    d[1, 0, 0] = xc ** 2
    d[1, 0, 1] = s / math.sqrt(2.0)
    d[1, 0, 2] = xs ** 2
    d[1, 1, 0] = -d[1, 0, 1]
    d[1, 1, 1] = math.cos(betax)
    d[1, 1, 2] = d[1, 0, 1]
    d[1, 2, 0] = d[1, 0, 2]
    d[1, 2, 1] = -d[1, 1, 2]
    d[1, 2, 2] = d[1, 0, 0]
    for l in range(3, lxp1 + 1):
        ln, lm = 2 * l - 1, 2 * l - 3
        ln, lm = min(ln, ndm), min(lm, ndm)
        for n in range(1, ln + 1):
            for m in range(1, lm + 1):
                t1 = (2 * l - 1 - n) * (2 * l - 2 - n)
                t = (2 * l - 1 - m) * (2 * l - 2 - m)
                f1 = math.sqrt(t1 / t)
                f2 = math.sqrt((2 * l - 1 - n) * (n - 1) / t)
                t3 = (n - 2) * (n - 1)
                f3 = math.sqrt(t3 / t)
                dlnm = f1 * xc ** 2 * d[l - 2, n - 1, m - 1]
                if n - 1 > 0:
                    dlnm = dlnm - f2 * s * d[l - 2, n - 2, m - 1]
                if n - 2 > 0:
                    dlnm = dlnm + f3 * xs ** 2 * d[l - 2, n - 3, m - 1]
                d[l - 1, n - 1, m - 1] = dlnm
                if n > 2 * l - 3:
                    d[l - 1, m - 1, n - 1] = (-1) ** (n - m) * d[l - 1, n - 1, m - 1]
            if n > 2 * l - 3:
                d[l - 1, 2 * l - 3, 2 * l - 3] = d[l - 1, 1, 1]
                d[l - 1, 2 * l - 2, 2 * l - 3] = -d[l - 1, 0, 1]
                d[l - 1, 2 * l - 3, 2 * l - 2] = -d[l - 1, 1, 0]
                d[l - 1, 2 * l - 2, 2 * l - 2] = d[l - 1, 0, 0]
    return d


class Cluster:
    """The atoms within rfms of the first atom carrying potential iph0, in FEFF's order, with the rotation matrices
    of every pair (yprep.f) and the Legendre normalisations (xanlm)."""

    def __init__(self, iph0, rat, iphat, rfms, lxp1=LX + 1, device="cpu"):
        rat = np.asarray(rat, float)
        iphat = np.asarray(iphat, int)
        icen = int(np.nonzero(iphat == iph0)[0][0])
        rat2 = [(np.float32(rat[i] - rat[icen])).astype(float).tolist() for i in range(len(rat))]
        rat2, iphat2 = atheap(rat2, iphat.tolist())
        rmax2 = float(np.float32(rfms)) ** 2
        inclus = 0
        for i, r in enumerate(rat2):
            if r[0] ** 2 + r[1] ** 2 + r[2] ** 2 > rmax2:
                inclus = i
                break
        if inclus == 0:
            inclus = len(rat2)
        inclus = min(inclus, NCLUSX)
        self.n = inclus
        self.xrat = [rat2[i] for i in range(inclus)]
        self.iphx = [iphat2[i] for i in range(inclus)]
        self.lxp1 = lxp1
        lx = lxp1 - 1
        pi = 3.14159265358979323846
        n = inclus
        # drix[k][j][i][l] : (2l+1, 2l+1) complex with FEFF's index order [m2 + l, m1 + l]
        self.drix = [[[None] * n for _ in range(n)] for _ in range(2)]
        self.phi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                beta, phi = getang(self.xrat, i, j)
                self.phi[i, j] = phi
                if i == j:
                    continue
                for k in (0, 1):
                    b = beta if k == 0 else -beta
                    d = rotmat_small(b, lxp1, lxp1)
                    mats = []
                    for il in range(lxp1):
                        m = np.zeros((2 * il + 1, 2 * il + 1), complex)
                        for m1 in range(-il, il + 1):
                            for m2 in range(-il, il + 1):
                                m[m2 + il, m1 + il] = d[il, m1 + il, m2 + il]
                        for m1 in range(-il, il + 1):
                            dum = 1j * m1 * (phi - pi)
                            if k == 1:
                                dum = -dum
                            dum = np.exp(dum)
                            if k == 1:
                                m[:, m1 + il] *= dum               # drix(m2, m1) *= dum
                            else:
                                m[m1 + il, :] *= dum               # drix(m1, m2) *= dum
                        mats.append(m)
                    self.drix[k][j][i] = mats
        # xnlm(m, l) = sqrt((2l+1)(l-m)!/(l+m)!) with FEFF's scaled factorials
        a = 0.03125
        flg = [1.0, a]
        for i in range(2, 51):
            flg.append(flg[-1] * i * a)
        self.xnlm = np.zeros((lxp1, lxp1))
        for il in range(1, lxp1 + 1):
            for im in range(1, min(lxp1, il) + 1):
                l, m = il - 1, im - 1
                c = (2 * l + 1) * flg[l - m] / flg[l + m]
                self.xnlm[m, l] = math.sqrt(c) * a ** m
        self.device = torch.device(device)
        self.r = np.array([[math.sqrt(sum((self.xrat[i][q] - self.xrat[j][q]) ** 2 for q in range(3))) for j in range(n)] for i in range(n)])


def xclmz(lmaxp1, mmaxp1, rho):
    """Rehr-Albers polynomials c_lm(z), z = -i/rho, for a batch rho (E,) -> clm (E, lmaxp1, mmaxp1) 0-based [l, m]."""
    E = rho.shape[0]
    z = (-1j) / rho
    lmax = lmaxp1 - 1
    clm = torch.zeros(E, lmaxp1 + 2, mmaxp1 + 2, dtype=rho.dtype, device=rho.device)   # 1-based copy of the Fortran array (ltotb+1 rows)
    clm[:, 1, 1] = 1
    clm[:, 2, 1] = clm[:, 1, 1] - z
    for il in range(2, lmax + 1):
        clm[:, il + 1, 1] = clm[:, il - 1, 1] - (z * (2 * il - 1) * clm[:, il, 1])
    cmm = torch.ones_like(z)
    mmxp1 = min(lmaxp1, mmaxp1)
    for im in range(2, mmxp1 + 1):
        m = im - 1
        imp1 = im + 1
        cmm = (-cmm) * (2 * m - 1) * z
        clm[:, im, im] = cmm
        clm[:, imp1, im] = cmm * (2 * m + 1) * (1 - im * z)
        for il in range(imp1, lmax + 1):
            clm[:, il + 1, im] = clm[:, il - 1, im] - (2 * il - 1) * z * (clm[:, il, im] + clm[:, il, m])
    return clm[:, 1:lmaxp1 + 1, 1:mmaxp1 + 1]                            # [l, m] 0-based, shape (E, lmaxp1, mmaxp1)


def solve_batch(a, b):
    """Solve ``a x = b`` for a batch: a (E, n, n), b (E, n, m).

    In some PyTorch builds the batched LU factorization on the CPU returns invalid pivots when the number of threads
    was changed (``torch.set_num_threads``) before the first linear-algebra call of the process, and the batched solve
    then raises. Solving the matrices one at a time is not affected, so that is the fallback here; its result is
    checked against the equations before it is used.
    """
    try:
        return torch.linalg.solve(a, b)
    except torch.linalg.LinAlgError:
        raise                                                       # a singular matrix is a real error
    except RuntimeError as err:
        x = torch.stack([torch.linalg.solve(a[i], b[i]) for i in range(a.shape[0])])
        scale = max(float(b.abs().max()), 1e-300)
        if not bool(torch.isfinite(x).all()) or float((a @ x - b).abs().max()) > 1e-8 * scale * a.shape[-1]:
            raise RuntimeError("the linear solve of the full multiple-scattering matrix failed, in a batch and one matrix at a time. "
                               "If torch.set_num_threads() was called in this process, set the number of threads with the environment "
                               "variable OMP_NUM_THREADS before Python starts instead") from err
        return x


def fms(cl: Cluster, ck, lipotx, xphase, iph0, lfms=0):
    """G for the central atom's block. ck (E,) complex momentum; xphase (E, lx+1, nph+1) phase shifts; lipotx (nph+1,)
    l limits. Returns gtr (E, lx+1) for potential iph0 (lfms = 0)."""
    dev = cl.device
    E = ck.shape[0]
    lx = cl.lxp1 - 1
    n = cl.n
    states, i0, first = [], {}, {}
    for iat in range(n):
        ip = cl.iphx[iat]
        if ip not in i0:
            i0[ip] = len(states)
        first[iat] = len(states)
        lim = min(lx, lipotx[ip])
        for l in range(lim + 1):
            for m in range(-l, l + 1):
                states.append((iat, l, m))
    ns = len(states)
    lim_of = [min(lx, lipotx[cl.iphx[iat]]) for iat in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    P = len(pairs)
    rpair = torch.tensor([cl.r[i, j] for i, j in pairs], dtype=RDTYPE, device=dev)
    rho = ck[:, None] * rpair[None, :]                                         # (E, P)
    c = xclmz(cl.lxp1, cl.lxp1, rho.reshape(-1)).reshape(E, P, cl.lxp1, cl.lxp1)   # (E, P, l, m)
    prefac = torch.exp(1j * rho) / rho
    xnlm = torch.tensor(cl.xnlm, dtype=RDTYPE, device=dev)
    g0 = torch.zeros(E, ns, ns, dtype=CDTYPE, device=dev)
    lmaxc = max(lim_of)
    for l1 in range(lmaxc + 1):
        for l2 in range(lmaxc + 1):
            sel = [k for k, (i, j) in enumerate(pairs) if lim_of[i] >= l1 and lim_of[j] >= l2]
            if not sel:
                continue
            ks = torch.tensor(sel, device=dev)
            mumax = min(l1, l2)
            # gllmz(mu) for mu = 0..mumax: sum_nu (2l1+1) c[l1, nu]/xnlm[mu, l1] * (-1)^mu c[l2, mu+nu] xnlm[mu, l2]
            G = torch.zeros(E, len(sel), 2 * l1 + 1, dtype=CDTYPE, device=dev)
            cs = c[:, ks]                                                     # (E, S, l, m)
            for mu in range(mumax + 1):
                numax = min(l1, l2 - mu)
                acc = torch.zeros(E, len(sel), dtype=CDTYPE, device=dev)
                for nu in range(numax + 1):
                    gamtl = (2 * l1 + 1) * cs[:, :, l1, nu] / xnlm[mu, l1]
                    gam = (-1) ** mu * cs[:, :, l2, mu + nu] * xnlm[mu, l2]
                    acc = acc + gamtl * gam
                G[:, :, mu + l1] = acc
                if mu > 0:
                    G[:, :, -mu + l1] = acc
            A = torch.tensor(np.stack([cl.drix[1][j][i][l1] for (i, j) in (pairs[k] for k in sel)]), dtype=CDTYPE, device=dev)  # (S, mu, m1)
            B = torch.tensor(np.stack([cl.drix[0][j][i][l2] for (i, j) in (pairs[k] for k in sel)]), dtype=CDTYPE, device=dev)  # (S, m2, mu)
            # block[m1, m2] = sum_mu A[mu, m1] G[mu] B[m2, mu]; mu limited to |mu| <= min(l1, l2) (G is zero elsewhere)
            mu_lo, mu_hi = -mumax, mumax
            Ag = A[:, mu_lo + l1:mu_hi + l1 + 1, :]                           # (S, mu, m1)
            Bg = B[:, :, mu_lo + l2:mu_hi + l2 + 1]                           # (S, m2, mu)
            Gg = G[:, :, mu_lo + l1:mu_hi + l1 + 1]                           # (E, S, mu)
            block = torch.einsum("sum,esu,snu->esmn", Ag, Gg, Bg) * prefac[:, ks, None, None]
            rows = torch.tensor([[first[pairs[k][0]] + l1 * l1 + mm for mm in range(2 * l1 + 1)] for k in sel], device=dev)   # (S, 2l1+1)
            cols = torch.tensor([[first[pairs[k][1]] + l2 * l2 + mm for mm in range(2 * l2 + 1)] for k in sel], device=dev)   # (S, 2l2+1)
            g0[:, rows[:, :, None], cols[:, None, :]] = block
    tdiag = torch.zeros(E, ns, dtype=CDTYPE, device=dev)
    for a, (iat1, l1, m1) in enumerate(states):
        tdiag[:, a] = (torch.exp(2j * xphase[:, l1, cl.iphx[iat1]]) - 1.0) / (2j)
    g0t = torch.eye(ns, dtype=CDTYPE, device=dev)[None] - g0 * tdiag[:, None, :]
    ip = iph0
    ipart = (lipotx[ip] + 1) ** 2
    cols = g0[:, :, i0[ip]:i0[ip] + ipart]
    x = solve_batch(g0t, cols)
    gg = x[:, i0[ip]:i0[ip] + ipart, :]
    gtr = torch.zeros(E, lx + 1, dtype=CDTYPE, device=dev)
    for il in range(lipotx[ip] + 1):
        ix = il ** 2
        s_ = torch.zeros(E, dtype=CDTYPE, device=dev)
        for im in range(2 * il + 1):
            s_ = s_ + gg[:, ix + im, ix + im]
        gtr[:, il] = s_ * torch.exp(2j * xphase[:, il, ip]) / (2 * il + 1)
    return gtr
