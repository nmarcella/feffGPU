"""FEFF's phase-shift stage (feff85exafs XSPH, FOVRG, EXCH) in torch, double precision, batched over energies and
partial waves. From the potentials stage (pot.pad) it produces what the path stage needs: the complex energy grid,
the energy reference (interstitial potential plus self-energy), the partial-wave phase shifts of every unique
potential and the absorber's reduced dipole matrix elements, i.e. the content of phase.pad.

The port is literal: the same grids (including the single-precision constants FEFF's Fortran mixes in), the same
self-energy (Hedin-Lundqvist with Quinn's low-energy losses), the same radial solver (Desclaux power expansion at
the origin, Runge-Kutta start, Milne predictor-corrector, exact flat-potential steps where the grid is too coarse),
the same phase matching and the same relativistic dipole integrals. Covered: unpolarised, spin-independent,
Hedin-Lundqvist or ground-state exchange (ixc 0 or 2), K, L and M edges, no PLASMON / TDLDA / nonlocal exchange.
Units are FEFF's code units (Hartree, bohr).
"""
from __future__ import annotations
import json, math, os
import numpy as np
import torch
from .pad import PotData, PhaseData, BOHR, HART

ALPINV = 137.035999139
ALPHFS = 1.0 / ALPINV
PI = 3.1415926535897932384626433
FA = 1.919158292677512811
THIRD = 1.0 / 3.0
CL_W = 137.0373            # wfirdc.f: speed of light used by the radial solver and the origin expansion
X0 = 8.8
NRPTX = 1251
LTOT = 80                  # dim.h of the rebuilt FEFF; besjn evaluates ltot + 2 orders
NPI = 6
MILNE_TEST = 1.0e5
USE_KERNELS = os.environ.get("FEFFGPU_KERNELS", "1").strip().lower() not in ("0", "no", "off", "false")
# Triton kernels for the radial recursions on CUDA tensors (FEFFGPU_KERNELS=0 switches them off); see kernels_usable
CDTYPE = torch.complex128
RDTYPE = torch.float64


def f32(x):
    """A Fortran single-precision literal, as the double it becomes inside a double-precision expression."""
    return float(np.float32(x))


def powi(x, n):
    """x ** n for an integer n, by binary exponentiation (libgfortran pow_*_i4 and GCC's powi expansion)."""
    if n == 0:
        return torch.ones_like(x) if torch.is_tensor(x) else 1.0
    if n < 0:
        return powi(1.0 / x, -n)
    out, base, u = None, x, n
    while True:
        if u & 1:
            out = base if out is None else out * base
        u >>= 1
        if u:
            base = base * base
        else:
            break
    return out


# ------------------------------------------------------------------ interpolation (feff6l/terp.f, polint)
def _locat(x, xx):
    n, lo, u = len(xx), 0, len(xx) + 1
    while u - lo > 1:
        m = (u + lo) // 2
        if x < xx[m - 1]:
            u = m
        else:
            lo = m
    return lo


def _polint(xa, ya, x):
    n = len(xa)
    c, d = list(ya), list(ya)
    ns, dif = 1, abs(x - xa[0])
    for i in range(1, n + 1):
        dift = abs(x - xa[i - 1])
        if dift < dif:
            ns, dif = i, dift
    y = ya[ns - 1]
    ns -= 1
    for m in range(1, n):
        for i in range(1, n - m + 1):
            ho, hp = xa[i - 1] - x, xa[i + m - 1] - x
            w = c[i] - d[i - 1]
            den = w / (ho - hp)
            d[i - 1], c[i - 1] = hp * den, ho * den
        if 2 * ns < n - m:
            dy = c[ns]
        else:
            dy = d[ns - 1]
            ns -= 1
        y = y + dy
    return y


def terp(x, y, n, m, x0):
    """FEFF's terp: polynomial of order m through the m+1 points around x0 (x, y of length >= n)."""
    i = _locat(x0, x[:n])
    k = min(max(i - m // 2, 1), n - m)
    return _polint(x[k - 1:k + m], y[k - 1:k + m], x0)


# ------------------------------------------------------------------ grids (COMMON/fixvar.f, fixdsp.f, XSPH/phmesh.f)
def fixvar(rmt, edens, vtot, dmag, vint, rhoint, dxorg, dxnew, jumprm):
    """Potential and density of one unique potential moved from the 0.05 grid to the phase grid (rgrd); values
    beyond the muffin tin set to the interstitial ones. Returns ri, vtotph, rhoph, dmagx (nrptx each) and jrinew."""
    xxx = lambda j, delta: -X0 + (j - 1) * delta
    jjj = lambda r, delta: int((math.log(r) + X0) / delta) + 1
    jriorg = jjj(rmt, dxorg) + 1
    jrior1 = jriorg + 1
    xorg = [xxx(j, dxorg) for j in range(1, jrior1 + 1)]
    jrinew = jjj(rmt, dxnew) + 1
    xnew = [xxx(j, dxnew) for j in range(1, jrinew + 2)]
    vtotph, rhoph, dmagx = np.zeros(NRPTX), np.zeros(NRPTX), np.zeros(NRPTX)
    for j in range(jrinew):
        vtotph[j] = terp(xorg, vtot, jriorg, 3, xnew[j])
        rhoph[j] = terp(xorg, edens, jrior1, 3, xnew[j])
        dmagx[j] = terp(xorg, dmag, jrior1, 3, xnew[j])
    if jumprm == 1:
        vmt = terp(xorg, vtot, jriorg, 3, math.log(rmt))
        vjump = vint - vmt
    if jumprm > 0:
        vtotph[:jrinew] += vjump
    ri = np.array([math.exp(-X0 + (j - 1) * dxnew) for j in range(1, NRPTX + 1)])
    rhoph[:jrinew] = rhoph[:jrinew] / (4 * PI)
    vtotph[jrinew:] = vint
    rhoph[jrinew:] = rhoint / (4 * PI)
    dmagx[jrinew:] = 0.0
    return ri, vtotph, rhoph, dmagx, jrinew


def fixdsp(dxorg, dxnew, dgc0, dpc0):
    """The core-hole orbital on the phase grid (zero beyond its last point). Returns dgcx, dpcx, jnew."""
    imax = 0
    for i in range(251, 0, -1):
        if abs(dgc0[i - 1]) >= 1.0e-11 or abs(dpc0[i - 1]) >= 1.0e-11:
            imax = i
            break
    jmax = min(imax, 250) + 1
    xorg = [-X0 + (j - 1) * dxorg for j in range(1, jmax + 1)]
    rmax = math.exp(-X0 + (jmax - 1) * dxorg)
    jnew = int((math.log(rmax) + X0) / dxnew) + 1
    xnew = [-X0 + (j - 1) * dxnew for j in range(1, jnew + 1)]
    dgcx, dpcx = np.zeros(NRPTX), np.zeros(NRPTX)
    for j in range(jnew):
        dgcx[j] = terp(xorg, dgc0, jmax, 3, xnew[j])
        dpcx[j] = terp(xorg, dpc0, jmax, 3, xnew[j])
    return dgcx, dpcx, jnew


def phmesh_exafs(edge, emu, vi0, gamach, xkmax):
    """FEFF's energy mesh for EXAFS (ispec = 0): the horizontal grid in k, then the vertical grid above the edge.
    Returns em (complex, Hartree), ne, ne1, ik0 (1-based like FEFF)."""
    xloss = gamach / 2 + vi0
    if xloss < 0:
        xloss = 0.0
    xvert = max(xloss, f32(0.02) / HART)
    xloss = xvert
    em = []
    ik0 = 0
    delk = BOHR / 10
    for i in range(1, 21):
        tempk = (i - 1) * delk
        em.append(tempk ** 2 / 2 + edge + 1j * xloss)
        if i == 1:
            ik0 = len(em)
    delk = BOHR / 5
    for i in range(1, 21):
        tempk = 2 * BOHR + (i - 1) * delk
        em.append(tempk ** 2 / 2 + edge + 1j * xloss)
    delk = BOHR / 2
    for i in range(1, 10):
        tempk = 6 * BOHR + (i - 1) * delk
        em.append(tempk ** 2 / 2 + edge + 1j * xloss)
    delk = BOHR
    for i in range(1, 11):
        tempk = 11 * BOHR + (i - 1) * delk
        em.append(tempk ** 2 / 2 + edge + 1j * xloss)
    while tempk < xkmax:
        tempk = tempk + delk
        em.append(tempk ** 2 / 2 + edge + 1j * xloss)
    em = em[:100]
    ne1 = len(em)
    tempk = 0.005 / HART
    em.append(edge + 1j * tempk)
    tempk = tempk * 2
    em.append(edge + 1j * tempk)
    delk = 0.4
    n1 = int(round(math.log(xloss / tempk) / delk - 0.5))
    if n1 <= 0:
        n1 = 1
    bb = math.exp(delk)
    aa = 2 * xloss / (1 + bb)
    aa = aa / bb ** n1
    if aa <= tempk:
        aa = aa * bb
    ee = min(50.0 / HART, 20.0 * xloss)
    n1 = int(round(math.log(ee / aa) / delk))
    for i in range(0, n1 + 1):
        em.append(edge + 1j * aa * math.exp(delk * i))
    return np.array(em), len(em), ne1, ik0


# ------------------------------------------------------------------ self-energy (EXCH: rhl, imhl, quinn, cubic, ffq, edp, xcpot)
_RCFR = np.array([-0.173963, -0.173678, -0.142040, -0.101030, -0.838843e-01, -0.807046e-01, -0.135577, -0.177556,
                  -0.645803e-01, -0.731172e-01, -0.498823e-01, -0.393108e-01, -0.116431, -0.909300e-01, -0.886979e-01,
                  -0.702319e-01, 0.791051e-01, -0.359401e-01, -0.379584e-01, -0.419807e-01, -0.628162e-01,
                  0.669257e-01, 0.667119e-01, 0.648175e-01]).reshape(2, 3, 4).transpose(2, 1, 0)      # (lrs, nrs, nright)
_RCFL = np.array([0.590195e+02, 0.478860e+01, 0.812813, 0.191145, -0.291180e+03, -0.926539e+01, -0.858348, -0.246947,
                  0.363830e+03, 0.460433e+01, 0.173067, 0.239738e-01, -0.181726e+03, -0.169709e+02, -0.409425e+01,
                  -0.173077e+01, 0.886023e+03, 0.301808e+02, 0.305836e+01, 0.743167, -0.110486e+04, -0.149086e+02,
                  -0.662794, -0.100106, 0.184417e+03, 0.180204e+02, 0.450425e+01, 0.184349e+01, -0.895807e+03,
                  -0.318696e+02, -0.345827e+01, -0.855367, 0.111549e+04, 0.156448e+02, 0.749582, 0.117680,
                  -0.620411e+02, -0.616427e+01, -0.153874e+01, -0.609114, 0.300946e+03, 0.109158e+02, 0.120028e+01,
                  0.290985, -0.374494e+03, -0.535127e+01, -0.261260, -0.405337e-01]).reshape(4, 3, 4).transpose(2, 1, 0)
_F1P = f32(1.00001)
_ALPH = 4.0 / 3.0


def ffq(q, ef, xk, wp, alph=_ALPH):
    wq = torch.sqrt(wp * wp + alph * (q * q) + powi(q, 4))
    f = (wp + wq) / (q * q) + alph / (2 * wp)
    return ((ef * wp) / (4 * xk)) * torch.log(f)


def cubic(xk0, wp, alph=_ALPH):
    a2 = (alph / (4 * (xk0 * xk0)) - 1) * xk0
    a0 = wp * wp / (4 * xk0)
    q = 0.0 / 3 - a2 * a2 / 9
    r = (0.0 * a2 - 3 * a0) / 6 - powi(a2, 3) / 27
    rad = powi(q, 3) + r * r
    neg = rad <= 0
    s13 = torch.complex(r, torch.sqrt(torch.where(neg, -rad, torch.zeros_like(rad))))
    s1 = s13 ** THIRD
    qz1 = 2 * s1.real - a2 / 3
    qz3 = -(s1.real - math.sqrt(3.0) * s1.imag + a2 / 3)
    zero = torch.zeros_like(rad)
    return rad, torch.where(neg, qz1, zero), torch.where(neg, qz3, zero)


def quinn(x, rs, wp, ef):
    alphaq = 1.0 / FA
    pisqrt = math.sqrt(PI)
    pfq = pisqrt / (32 * (alphaq * rs) ** 1.5)
    temp1 = torch.atan(torch.sqrt(PI / (alphaq * rs)))
    temp2 = torch.sqrt(alphaq * rs / PI) / (1 + alphaq * rs / PI)
    pfq = pfq * (temp1 + temp2)
    wkc = (torch.sqrt(1 + wp) - 1) ** 2
    wkc = (1 + f32(6.0 / 5.0) * wkc / (wp * wp)) * wp * ef
    ekc = wkc + ef
    gam = (pfq / x) * (x * x - 1) ** 2
    eabs = ef * (x * x)
    arg = (eabs - ekc) / (f32(0.3) * ekc)
    f = torch.where(arg < 80, 1 / (1 + torch.exp(torch.clamp(arg, max=80.0))), torch.zeros_like(arg))
    return -gam * f / 2


def imhl(rs, xk):
    alph = _ALPH
    xf = FA / rs
    ef = xf * xf / 2
    xk0 = xk / xf
    xk0 = torch.where(xk0 < _F1P, torch.full_like(xk0, _F1P), xk0)
    wp = torch.sqrt(3 / powi(rs, 3)) / ef
    xs = wp * wp - (xk0 * xk0 - 1) ** 2
    eim = torch.zeros_like(rs)
    icusp = torch.zeros_like(rs, dtype=torch.long)
    # xs < 0 branch
    neg = xs < 0
    xs_s = torch.where(neg, xs, torch.zeros_like(xs))
    q2 = torch.sqrt(torch.clamp((torch.sqrt(alph * alph - 4 * xs_s) - alph) / 2, min=0.0))
    qu = torch.minimum(q2, 1 + xk0)
    d1 = qu - (xk0 - 1)
    use1 = neg & (d1 > 0)
    eim = torch.where(use1, ffq(qu, ef, xk, wp) - ffq(xk0 - 1, ef, xk, wp), eim)
    rad, qplus, qminus = cubic(xk0, wp, alph)
    r0 = rad <= 0
    d2 = qplus - (xk0 + 1)
    use2 = r0 & (d2 > 0)
    qp_s = torch.where(use2, qplus, xk0 + 1)
    eim = torch.where(use2, eim + (ffq(qp_s, ef, xk, wp) - ffq(xk0 + 1, ef, xk, wp)), eim)
    d3 = (xk0 - 1) - qminus
    use3 = r0 & (d3 > 0)
    qm_s = torch.where(use3, qminus, xk0 - 1)
    eim = torch.where(use3, eim + (ffq(xk0 - 1, ef, xk, wp) - ffq(qm_s, ef, xk, wp)), eim)
    icusp = torch.where(use3, torch.ones_like(icusp), icusp)
    ei = quinn(xk0, rs, wp, ef)
    eim = torch.where(eim >= ei, ei, eim)
    return eim, icusp


def rhl(rs, xk):
    """Hedin-Lundqvist self-energy, real part by FEFF's interpolation, imaginary part analytic. (erl, eim)."""
    dev = rs.device
    rkf = FA / rs
    ef = rkf * rkf / 2
    wp = torch.sqrt(3 / powi(rs, 3))
    dwp = wp / 3.0
    eim, icusp = imhl(rs, xk)
    xx = xk / rkf
    xx = torch.where(xx < _F1P, torch.full_like(xx, _F1P), xx)
    deltae = ((xx * xx - 1.0) * ef - wp - dwp) / dwp
    mrs = torch.where(rs < 0.2, 0, torch.where(rs < 1.0, 1, torch.where(rs < 5.0, 2, 3)))
    rcfr = torch.tensor(_RCFR, device=dev, dtype=RDTYPE)[mrs]              # (..., 3, 2)
    rcfl = torch.tensor(_RCFL, device=dev, dtype=RDTYPE)[mrs]              # (..., 3, 4)
    sq = torch.sqrt(rs)
    cright = [rcfr[..., 0, j] * rs + rcfr[..., 1, j] * rs * sq + rcfr[..., 2, j] * (rs * rs) for j in range(2)]
    eee = -PI * wp / (4 * rkf * ef)
    cleft = [rcfl[..., 0, j] * rs + rcfl[..., 1, j] * rs ** 1.5 + rcfl[..., 2, j] * (rs * rs) for j in range(4)]
    erl = cleft[0]
    for j in range(2, 5):
        erl = erl + cleft[j - 1] * powi(xx, j - 1)
    erlr = eee / xx
    for j in range(1, 3):
        erlr = erlr + cright[j - 1] / powi(xx, j + 1)
    blend = deltae.abs() < 1.0
    wr = torch.where(deltae < 0, (1.0 + deltae) ** 2 / 2.0, 1.0 - (1.0 - deltae) ** 2 / 2.0)
    erl_b = wr * erlr + (1.0 - wr) * erl
    erl = torch.where(blend, erl_b, torch.where(icusp == 1, erlr, erl))
    return erl * ef, eim


def edp(rs, xk):
    xf = FA / rs
    x = xk / xf
    x = x + f32(1.0e-5)
    x = torch.where(x < _F1P, torch.full_like(x, _F1P), x)
    c = torch.log(((1 + x) / (1 - x)).abs())
    vr = -(xf / PI) * (1 + c * (1 - x * x) / (2 * x))
    return torch.where(rs <= 100.0, vr, torch.zeros_like(vr))


def sigma(ixc, rs, xk):
    if ixc in (0, 5, 6, 7, 8, 9):
        return rhl(rs, xk)
    if ixc == 1:
        return edp(rs, xk), torch.zeros_like(rs)
    if ixc == 3:
        return edp(rs, xk), imhl(rs, xk)[0]
    raise NotImplementedError(f"exchange model ixc={ixc}")


def xcpot(ixc, lreal, em, xmu, vtot, densty, jri):
    """Complex potential v(r) - eref and eref for all energies (EXCH/xcpot.f for ixc 0, 1, 2, 3; no PLASMON).
    em (E,) complex tensor, vtot/densty (nrptx,) real tensors on the phase grid. Returns v (E, jri+1), eref (E,)."""
    E = em.shape[0]
    jri1 = jri + 1
    vt = vtot[:jri1]
    dens = densty[:jri1]
    rs = torch.where(dens <= 0, torch.full_like(dens, 10.0), (3 / (4 * PI * dens)) ** THIRD)
    xf = FA / rs
    ground = (em.real <= xmu) | (ixc == 2)
    v = vt[None, :].expand(E, -1).to(CDTYPE).clone()
    if not bool(ground.all()):
        xk = xf * _F1P
        vxcrmu, vxcimu = sigma(ixc, rs, xk)
        er = em.real[:, None]
        xk2 = 2 * (er - xmu) + (xf * xf)[None]
        xk = torch.sqrt(xk2.clamp(min=0))
        rs2, xf2 = rs[None].expand(E, -1), xf[None].expand(E, -1)
        vxcr, vxci = sigma(ixc, rs2, xk)
        del1r = 1.0 * (vxcr - vxcrmu[None])
        for _ in range(2):
            xk2 = 2 * (er - xmu - del1r) + (xf * xf)[None]
            if bool(((xk2 < 0) & ~ground[:, None]).any()):
                raise RuntimeError("XCPOT-2: negative local momentum squared")
            xk = torch.sqrt(xk2.clamp(min=0))
            vxcr, vxci = sigma(ixc, rs2, xk)
            delr = 1.0 * (vxcr - vxcrmu[None])
            deli = vxci - vxcimu[None]
            del1r = delr
        v = torch.where(ground[:, None], v, vt[None] + torch.complex(delr, deli))
    eref = v[:, jri1 - 1].clone()
    v = v - eref[:, None]
    if lreal > 0:
        v = v.real.to(CDTYPE)
        eref = eref.real.to(CDTYPE)
    return v, eref


# ------------------------------------------------------------------ spherical Bessel functions (feff6l/besjn.f, bjnser.f, MATH/exjlnl.f)
_TOL = f32(1.0e-15)


def bjnser(x, l, want=0):
    """Series 10.1.2/10.1.3 for one order l on a batch; want 0 both, 1 jl only, 2 nl only."""
    u = x * x / 2
    djl, fac = 1.0, -1
    for _ in range(l + 1):
        fac += 2
        djl = fac * djl
    dnl = djl / (2 * l + 1)

    def series(nfac):
        p = torch.ones_like(x)
        nf, den, sgn, ux = 1, float(nfac), -1.0, u
        done = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        sync = not x.is_cuda                                  # on the GPU the early exit would cost a sync per term
        for _ in range(20):
            delta = sgn * ux / den
            pn = p + delta
            trel = (delta / pn).abs()
            p = torch.where(done, p, pn)
            done = done | (trel <= _TOL)
            if sync and bool(done.all()):
                break
            sgn = -sgn
            ux = u * ux
            nf += 1
            nfac += 2
            den = nf * nfac * den
        return p

    jl = nl = None
    if want != 2:
        jl = series(2 * l + 3) * powi(x, l) / djl
    if want != 1:
        nl = -series(1 - 2 * l) * dnl / powi(x, l + 1)
    return jl, nl


def besjn(x, lmaxp1=LTOT + 2):
    """jl and nl (Abramowitz conventions) for orders 0..lmaxp1-1 on a batch of complex arguments (M,) -> (M, lmaxp1)."""
    M, dev = x.shape[0], x.device
    jl = torch.zeros(M, lmaxp1, dtype=x.dtype, device=dev)
    nl = torch.zeros_like(jl)
    rex = x.real
    A, C = rex < 1.0, rex >= 7.51
    B = ~A & ~C
    if bool(A.any()):
        xa = x[A]
        for il in range(lmaxp1):
            j_, n_ = bjnser(xa, il, 0)
            jl[A, il], nl[A, il] = j_, n_
    if bool(B.any()):
        xb = x[B]
        jb = [None] * lmaxp1
        nb = [None] * lmaxp1
        jb[lmaxp1 - 1] = bjnser(xb, lmaxp1 - 1, 1)[0]
        jb[lmaxp1 - 2] = bjnser(xb, lmaxp1 - 2, 1)[0]
        small = xb.real < 5.01
        n0s, n1s = torch.zeros_like(xb), torch.zeros_like(xb)
        if bool(small.any()):
            n0s[small] = bjnser(xb[small], 0, 2)[1]
            n1s[small] = bjnser(xb[small], 1, 2)[1]
        asx, acx = torch.sin(xb), torch.cos(xb)
        xi = 1 / xb
        xi2 = xi * xi
        n0f, n1f = -acx * xi, -acx * xi2 - asx * xi
        nb[0] = torch.where(small, n0s, n0f)
        nb[1] = torch.where(small, n1s, n1f)
        for lp1 in range(3, lmaxp1 + 1):
            tlxp1 = 2 * lp1 - 3
            nb[lp1 - 1] = tlxp1 * nb[lp1 - 2] / xb - nb[lp1 - 3]
        for lx in range(3, lmaxp1 + 1):
            lp1 = lmaxp1 + 1 - lx
            l = lp1 - 1
            tlxp3 = 2 * l + 3
            jb[lp1 - 1] = tlxp3 * jb[lp1] / xb - jb[lp1 + 1]
        jl[B] = torch.stack(jb, 1)
        nl[B] = torch.stack(nb, 1)
    if bool(C.any()):
        xc = x[C]
        xi = 1 / xc
        xi2 = xi * xi; xi3 = xi * xi2; xi4 = xi * xi3; xi5 = xi * xi4; xi6 = xi * xi5; xi7 = xi * xi6
        xi8 = xi * xi7; xi9 = xi * xi8; xi10 = xi * xi9; xi11 = xi * xi10
        sjl = [xi, xi2, 3.0 * xi3 - xi, 15.0 * xi4 - 6.0 * xi2, 105.0 * xi5 - 45.0 * xi3 + xi,
               945.0 * xi6 - 420.0 * xi4 + 15.0 * xi2, 10395.0 * xi7 - 4725.0 * xi5 + 210.0 * xi3 - xi,
               135135.0 * xi8 - 62370.0 * xi6 + 3150.0 * xi4 - 28.0 * xi2,
               2027025.0 * xi9 - 945945.0 * xi7 + 51975.0 * xi5 - 630.0 * xi3 + xi,
               34459425.0 * xi10 - 16216200.0 * xi8 + 945945.0 * xi6 - 13860.0 * xi4 + 45.0 * xi2,
               654729075.0 * xi11 - 310134825.0 * xi9 + 18918900.0 * xi7 - 315315.0 * xi5 + 1485.0 * xi3 - xi]
        cjl = [torch.zeros_like(xi), -xi, -3.0 * xi2, -15.0 * xi3 + xi, -105.0 * xi4 + 10.0 * xi2,
               -945.0 * xi5 + 105.0 * xi3 - xi, -10395.0 * xi6 + 1260.0 * xi4 - 21.0 * xi2,
               -135135.0 * xi7 + 17325.0 * xi5 - 378.0 * xi3 + xi,
               -2027025.0 * xi8 + 270270.0 * xi6 - 6930.0 * xi4 + 36.0 * xi2,
               -34459425.0 * xi9 + 4729725.0 * xi7 - 135135.0 * xi5 + 990.0 * xi3 - xi,
               -654729075.0 * xi10 + 91891800.0 * xi8 - 2837835.0 * xi6 + 25740.0 * xi4 - 55.0 * xi2]
        snl = [c for c in cjl]
        cnl = [-s for s in sjl]
        for lp1 in range(12, lmaxp1 + 1):
            tlxp1 = 2 * lp1 - 3
            sjl.append(tlxp1 * xi * sjl[lp1 - 2] - sjl[lp1 - 3])
            cjl.append(tlxp1 * xi * cjl[lp1 - 2] - cjl[lp1 - 3])
            snl.append(tlxp1 * xi * snl[lp1 - 2] - snl[lp1 - 3])
            cnl.append(tlxp1 * xi * cnl[lp1 - 2] - cnl[lp1 - 3])
        asx, acx = torch.sin(xc), torch.cos(xc)
        jl[C] = torch.stack([asx * sjl[i] + acx * cjl[i] for i in range(lmaxp1)], 1)
        nl[C] = torch.stack([asx * snl[i] + acx * cnl[i] for i in range(lmaxp1)], 1)
    return jl, nl


def exjlnl(z, l):
    """Analytic jl, nl for l <= 6 on a batch (series below |z| = 0.3)."""
    small = z.abs() < f32(0.3)
    jl, nl = torch.zeros_like(z), torch.zeros_like(z)
    if bool(small.any()):
        j_, n_ = bjnser(z[small], l, 0)
        jl[small], nl[small] = j_, n_
    big = ~small
    if bool(big.any()):
        zb = z[big]
        s, c = torch.sin(zb), torch.cos(zb)
        if l == 0:
            jb, nb = s / zb, -c / zb
        elif l == 1:
            jb, nb = s / (zb * zb) - c / zb, -c / (zb * zb) - s / zb
        elif l == 2:
            jb = (3 / powi(zb, 3) - 1 / zb) * s - 3 * c / (zb * zb)
            nb = (-3 / powi(zb, 3) + 1 / zb) * c - 3 * s / (zb * zb)
        elif l == 3:
            jb = (15 / powi(zb, 4) - 6 / (zb * zb)) * s + (-15 / powi(zb, 3) + 1 / zb) * c
            nb = (-15 / powi(zb, 4) + 6 / (zb * zb)) * c + (-15 / powi(zb, 3) + 1 / zb) * s
        elif l == 4:
            jb = (105 / powi(zb, 5) - 45 / powi(zb, 3) + 1 / zb) * s + (-105 / powi(zb, 4) + 10 / (zb * zb)) * c
            nb = (-105 / powi(zb, 5) + 45 / powi(zb, 3) - 1 / zb) * c + (-105 / powi(zb, 4) + 10 / (zb * zb)) * s
        else:
            raise NotImplementedError("exjlnl l > 4")
        jl[big], nl[big] = jb, nb
    return jl, nl


# ------------------------------------------------------------------ phase matching (MATH/phamp.f)
def atancc(temp):
    xx, yy = temp.real, temp.imag
    alph = (1 - xx * xx - yy * yy)
    alph = torch.sqrt(alph * alph + 4 * (xx * xx)) - alph
    alph = torch.where(xx != 0, alph / (2 * torch.where(xx != 0, xx, torch.ones_like(xx))), torch.zeros_like(xx))
    alph = torch.atan(alph)
    beta = (xx * xx + (yy + 1) ** 2) / (xx * xx + (yy - 1) ** 2)
    beta = torch.log(beta) / 4
    return torch.complex(alph, beta)


def atan2c(a, b):
    """Complex amplitude and phase with a = ampl cos(phx), b = ampl sin(phx)."""
    aa, bb = a.abs(), b.abs()
    zero = (aa + bb) == 0
    big_a = aa > bb
    safe_a = torch.where(big_a, a, torch.ones_like(a))
    safe_b = torch.where(big_a, torch.ones_like(b), b)
    ph_a = atancc(b / safe_a)
    amp_a = a / torch.cos(ph_a)
    ph_b = PI / 2 - atancc(a / safe_b)
    amp_b = b / torch.sin(ph_b)
    phx = torch.where(big_a, ph_a, ph_b)
    ampl = torch.where(big_a, amp_a, amp_b)
    phx = torch.where(zero, torch.zeros_like(phx), phx)
    ampl = torch.where(zero, torch.zeros_like(ampl), ampl)
    flip = ampl.real < 0
    ampl = torch.where(flip, -ampl, ampl)
    phx = torch.where(flip, phx + PI, phx)
    return ampl, phx


def phamp(rmt, pu, qu, ck, jl, nl, jlp, nlp, ikap):
    xkr = ck * rmt
    isign = torch.where(ikap < 0, -1.0, 1.0).to(RDTYPE)
    a = ck * ALPHFS
    factor = isign * a / (1 + torch.sqrt(1 + a * a))
    a = isign * ck * xkr * (pu * nlp - qu * nl / factor)
    b = isign * ck * xkr * (qu * jl / factor - pu * jlp)
    b = -b
    amp, ph = atan2c(a, b)
    return ph, amp


def flatv(r1, r2, p1, q1, en, vav, ikap, lmaxp1=LTOT + 2):
    """Dirac solution in a flat potential vav from r1 to r2 (FOVRG/dfovrg.f: flatv), batched."""
    ev = en - vav
    ck = torch.sqrt(2 * ev + (ALPHFS * ev) ** 2)
    xkr = ck * r1
    neg = ikap < 0
    lp = torch.where(neg, -ikap - 1, ikap)
    lq = torch.where(neg, lp + 1, lp - 1)
    isign = torch.where(neg, -1.0, 1.0).to(RDTYPE)
    a = ck * ALPHFS
    factor = isign * a / (1 + torch.sqrt(1 + a * a))
    jl, nl = besjn(xkr, lmaxp1)
    ar = torch.arange(xkr.shape[0], device=xkr.device)
    a = isign * ck * xkr * (p1 * nl[ar, lq] - q1 * nl[ar, lp] / factor)
    b = isign * ck * xkr * (q1 * jl[ar, lp] / factor - p1 * jl[ar, lq])
    xkr = ck * r2
    jl, nl = besjn(xkr, lmaxp1)
    p2 = r2 * (jl[ar, lp] * a + nl[ar, lp] * b)
    q2 = r2 * factor * (jl[ar, lq] * a + nl[ar, lq] * b)
    return p2, q2


# ------------------------------------------------------------------ orbital tables (COMMON/getorb.f, setkap.f)
_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = json.load(open(os.path.join(os.path.dirname(__file__), "data", "getorb.json")))
    return _TABLES


def getorb(iz, ihole, xion, iunf):
    """FEFF's atomic configuration with the core hole and its screening electron: orbital list (kappa, principal
    quantum number, occupation, valence occupation, spin), the hole's position in the list and iorb(-4..3)."""
    T = _tables()
    kappa, nnum = T["kappa"], T["nnum"]
    iocc = lambda idx, i: T["iocc"][idx - 1][i - 1]
    ival = lambda idx, i: T["ival"][idx - 1][i - 1]
    ispn = lambda idx, i: T["ispn"][idx - 1][i - 1]
    ion = int(math.floor(xion + 0.5)) if xion >= 0 else -int(math.floor(-xion + 0.5))
    delion = xion - ion
    index = iz - ion
    ilast = iscr = iion = 0
    iholep = ihole
    for i in range(29, 0, -1):
        if iion == 0 and iocc(index, i) > delion:
            iion = i
        if ilast == 0 and iocc(index, i) > 0:
            ilast = i
    if ihole > 0 and iocc(index, ihole) < 1:
        raise ValueError("cannot remove an electron from this level")
    index1 = index + 1
    for i in range(1, 30):
        if iscr == 0 and (iocc(index1, i) - iocc(index, i)) > 0.5:
            iscr = i
    if delion < 0:
        iion = iscr
        if ihole != 0 and iocc(index, iscr) + 1 - delion > 2 * abs(kappa[iscr - 1]):
            iion = ilast
            if ilast == iscr or iocc(index, ilast) - delion > 2 * abs(kappa[ilast - 1]):
                iion = ilast + 1
    norb = 0
    iorb = {i: 0 for i in range(-4, 4)}
    nqn, nk, xnel, xnval, xmag = [], [], [], [], []
    for i in range(1, 30):
        if iocc(index, i) > 0 or (i == iscr and ihole > 0) or (i == iion and iocc(index, i) - delion > 0):
            if i != ihole or iocc(index, i) >= 1:
                norb += 1
                nqn.append(nnum[i - 1])
                nk.append(kappa[i - 1])
                xn = iocc(index, i)
                if i == ihole:
                    xn = xn - 1
                    iholep = norb
                if i == iscr and ihole > 0:
                    xn = xn + 1
                xv = ival(index, i)
                if kappa[i - 1] in (-4, 3) and iunf == 0:
                    xv = 0.0
                xmag.append(ispn(index, i))
                iorb[kappa[i - 1]] = norb
                if i == ihole and xv >= 1:
                    xv = xv - 1
                if i == iscr and ihole > 0:
                    xv = xv + 1
                if i == iion:
                    xn = xn - delion
                    xv = xv - delion
                xnel.append(xn)
                xnval.append(xv)
    return dict(norb=norb, iorb=iorb, iholep=iholep, nqn=nqn, nk=nk, xnel=xnel, xnval=xnval, xmag=xmag)


def setkap(ihole):
    """(kinit, linit) of the core hole (COMMON/setkap.f)."""
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
    raise ValueError("invalid hole number")


# ------------------------------------------------------------------ Dirac radial solver (FOVRG: dfovrg, wfirdc, potdvp, nucdec, solout, intout, diff)
def nucdec_grid(iz, hx, n):
    """nucdec's radial mesh for a point nucleus: dr(1) = (nz exp(-8.8), in single precision as wfirdc computes
    it) / nz, then dr(l) = dr(1) exp(hx (l-1))."""
    e = np.float32(np.exp(np.float64(np.float32(-8.8))))
    dr1 = float(np.float32(iz) * e)
    d1 = dr1 / float(iz)
    return np.array([d1 * math.exp(hx * (l - 1)) for l in range(1, n + 1)])


def aprdep(a, b, l):
    s = 0.0
    for m in range(1, l + 1):
        s = s + a[m - 1] * b[l - m]
    return s


def origin_density_term(adgc, adpc, nk, xnel, dz, dr1, ndor=3):
    """potdvp: ap(1), the core density's contribution to the potential's constant term at the origin."""
    a = (dz / CL_W) ** 2
    ag = [0.0] * ndor
    for j in range(len(nk)):
        kap = nk[j]
        fl = math.sqrt(kap * kap - a)
        fix = dr1 ** (fl - abs(kap))
        bg, bp = adgc[:, j], adpc[:, j]
        n = 2 * abs(kap)
        l = ndor + 2 - n
        if l > 0:
            for i in range(1, l + 1):
                m = n - 2 + i
                ag[m - 1] = ag[m - 1] + xnel[j] * (aprdep(bg, bg, i) + aprdep(bp, bp, i)) * (fix * fix)
    ap1 = 0.0
    for i in range(1, ndor + 1):
        ag[i - 1] = ag[i - 1] / (i + 2) / (i + 1)
        ap1 = ap1 + ag[i - 1] * powi(dr1, i + 1)
    return ap1


def vm_derivative(v, ri, hx, n):
    """diff.f: (d(v r^2)/dx - 2 v r^2)/r for the first n points (the (kappa+1)/c factor is applied per lane)."""
    vt = v[:, :n] * (ri[:n] * ri[:n])[None]
    dx = hx
    out = torch.zeros_like(vt)
    c1, c2, c3, c4, c5, c6, c7 = 6.0, f32(6.66666666667), f32(1.2), f32(2.45), 7.5, 3.75, f32(.166666666667)
    out[:, 0] = ((c1 * vt[:, 1] + c2 * vt[:, 3] + c3 * vt[:, 5]) - (c4 * vt[:, 0] + c5 * vt[:, 2] + c6 * vt[:, 4] + c7 * vt[:, 6])) / dx
    out[:, 1] = ((c1 * vt[:, 2] + c2 * vt[:, 4] + c3 * vt[:, 6]) - (c4 * vt[:, 1] + c5 * vt[:, 3] + c6 * vt[:, 5] + c7 * vt[:, 7])) / dx
    out[:, 2:n - 2] = ((vt[:, 0:n - 4] + 8.0 * vt[:, 3:n - 1]) - (8.0 * vt[:, 1:n - 3] + vt[:, 4:n])) / 12.0 / dx
    out[:, n - 2] = (vt[:, n - 1] - vt[:, n - 3]) / (2.0 * dx)
    out[:, n - 1] = (vt[:, n - 3] * 0.5 - 2.0 * vt[:, n - 2] + 1.5 * vt[:, n - 1]) / dx
    return (out - 2 * vt) / ri[:n][None]


def _lane_grid(dr, L, dev):
    """dr as (L, N): one radial grid per lane (lanes of different species have different first points)."""
    if dr.dim() == 1:
        return dr[None, :].expand(L, -1)
    return dr


_KERNELS = {"checked": False, "usable": False, "note": "not decided yet (decided at the first calculation on a GPU)"}


def _kernel_test_problem(dev):
    """A small made-up set of lanes for the two Milne routines: benign numbers of the size the solver meets, three
    kernel blocks with a part-filled last one, lanes that stop at different points."""
    gen = torch.Generator().manual_seed(7)

    def rnd(*shape):
        return torch.rand(*shape, generator=gen, dtype=RDTYPE)

    L, N, hx = 150, 56, 0.05
    dr = (1.0e-3 * torch.exp(hx * torch.arange(N, dtype=RDTYPE)))[None].expand(L, N).contiguous()
    kapf = torch.tensor([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0], dtype=RDTYPE).repeat(L // 7 + 1)[:L].contiguous()
    ic3f = (torch.arange(L) % 2).to(RDTYPE)
    zeff = 20.0 + 40.0 * rnd(L)
    av1 = -zeff / CL_W
    dv = torch.complex(-zeff[:, None] / (dr * CL_W) + 0.3 * rnd(L, N), 0.01 * rnd(L, N))
    vm = torch.complex(0.05 * rnd(L, N), 0.01 * rnd(L, N))
    ec = torch.complex(0.5 * rnd(L) / CL_W, 0.01 * rnd(L) / CL_W)
    iflat0 = 40 + (torch.arange(L) % 7)
    start = torch.complex(0.5 + rnd(L, 2), 0.2 * rnd(L, 2))
    outer = torch.complex(0.5 + rnd(L, N), 0.2 * rnd(L, N))
    seeds = torch.complex(rnd(L, 2, 6) - 0.5, 0.2 * rnd(L, 2, 6))
    t = dict(dr=dr, kapf=kapf, ic3f=ic3f, av1=av1, dv=dv, vm=vm, ec=ec, iflat0=iflat0, start=start, outer=outer, seeds=seeds)
    return {k: v.to(dev) for k, v in t.items()}, L, N, hx


def kernel_difference(dev="cuda"):
    """The largest relative difference between the Triton kernels and the torch code on a small made-up problem:
    (outward, inward). Rounding gives about 1e-15."""
    from .kernels import milne_outward, milne_inward
    t, L, N, hx = _kernel_test_problem(dev)
    consts = _milne_constants(hx, CL_W + CL_W)

    def rel(a, b):
        return float(((a - b).abs() / b.abs().clamp_min(1e-300))[b.abs() > 0].max())

    def start():
        gg = torch.zeros(L, N, dtype=CDTYPE, device=t["dv"].device)
        gp = torch.zeros_like(gg)
        gg[:, 0], gp[:, 0] = t["start"][:, 0], t["start"][:, 1]
        return gg, gp

    k = milne_outward(*start(), t["dv"], t["vm"], t["dr"], t["ec"], t["kapf"], t["ic3f"], t["av1"].contiguous(), t["iflat0"], hx, consts)
    p = _milne_outward_plain(*start(), t["dv"], t["vm"], t["dr"], t["ec"], t["kapf"], t["ic3f"], t["av1"], t["iflat0"], hx, consts)
    d_out = max(rel(k[0], p[0]), rel(k[1], p[1]))

    def outer():
        keep = torch.arange(N, device=t["dv"].device)[None] >= t["iflat0"][:, None]
        return torch.where(keep, t["outer"], torch.zeros_like(t["outer"])), torch.where(keep, t["outer"].conj() * 0.7, torch.zeros_like(t["outer"]))

    D, P = t["seeds"][:, 0], t["seeds"][:, 1]
    k = milne_inward(*outer(), D.contiguous(), P.contiguous(), t["dv"], t["vm"], t["dr"], t["ec"], t["kapf"], t["ic3f"], t["iflat0"], hx, consts)
    p = _milne_inward_plain(*outer(), [D[:, j] for j in range(6)], [P[:, j] for j in range(6)], t["dv"], t["vm"], t["dr"], t["ec"],
                            t["kapf"], t["ic3f"], t["iflat0"], hx, consts)
    d_in = max(rel(k[0], p[0]), rel(k[1], p[1]))
    return d_out, d_in


class KernelCheckError(RuntimeError):
    """The compiled GPU kernels of the radial solver cannot be trusted in this environment."""


def kernels_usable(t):
    """Whether the Triton kernels are used for the tensor t. False on a CPU and when the kernels were switched off
    on purpose (FEFFGPU_KERNELS=0). On a CUDA device the kernels are used. A kernel is compiled by the installed
    Triton, and a compiler can change what a kernel computes (Triton 3.8 did, against 3.1). So: with a Triton
    version that the kernels were tested with before the release (`kernels.TESTED_TRITON`) nothing is checked; with
    any other version the kernels are first compared with the torch code on a small made-up problem, once per
    process. There is no automatic change to the slower torch code: if Triton is missing, or the kernels do not
    reproduce the torch code, or the comparison cannot run, the calculation stops with a KernelCheckError that says
    what to do."""
    if not (t.is_cuda and USE_KERNELS):
        return False
    if not _KERNELS["checked"]:
        _check_kernels(t.device)
    if not _KERNELS["usable"]:
        raise KernelCheckError(_KERNELS["note"])
    return True


_KERNEL_HELP = ("feffgpu does not continue with other code on its own. Report this with the output of `feffgpu doctor`. To run now with the "
                "plain PyTorch code of the radial solver (same results, potentials and phase shifts take longer), set the environment "
                "variable FEFFGPU_KERNELS=0.")


def _check_kernels(dev, test=False):
    """Decide once per process whether the kernels are used: by the Triton version if it is a tested one, else (or
    with test=True) by comparing the kernels with the torch code now."""
    _KERNELS["checked"] = True
    try:
        from .kernels import HAVE_TRITON, TESTED_TRITON
        if not HAVE_TRITON:
            _KERNELS.update(usable=False, note="the GPU kernels of the radial solver need Triton, which is not installed (it comes with "
                            "the CUDA builds of PyTorch on Linux: pip install triton). " + _KERNEL_HELP)
            return
        import triton
        version = triton.__version__
        tested = version in TESTED_TRITON
        if tested and not test:
            _KERNELS.update(usable=True, note=f"in use (Triton {version}, a version they were tested with)")
            return
        d_out, d_in = kernel_difference(dev)
        if max(d_out, d_in) < 1.0e-9:
            which = "a version they were tested with" if tested else "not a version they were released with, so they were tested now"
            _KERNELS.update(usable=True, note=f"in use (Triton {version}, {which}); they reproduce the plain code to {max(d_out, d_in):.0e}")
            return
        note = (f"the GPU kernels of the radial solver do not reproduce the plain code with Triton {version}: on the built-in test "
                f"they differ by {d_out:.1e} (outward) and {d_in:.1e} (inward), where rounding gives about 1e-14. The kernels were "
                f"released with Triton {', '.join(TESTED_TRITON)}. ")
    except Exception as err:
        note = f"the built-in test of the GPU kernels of the radial solver could not run ({err!r}). "[:400]
    _KERNELS.update(usable=False, note=note + _KERNEL_HELP)


def kernel_status(dev=None, test=False):
    """dict(usable, note) for `feffgpu doctor`. With a CUDA device the decision is made if it has not been made yet;
    test=True runs the comparison of the kernels with the torch code now, whatever the Triton version. Never raises."""
    if not USE_KERNELS:
        return {"usable": False, "note": "switched off on purpose (FEFFGPU_KERNELS=0): the plain code is used"}
    if dev is not None and (test or not _KERNELS["checked"]):
        _check_kernels(torch.device(dev), test=test)
    return {"usable": _KERNELS["usable"], "note": _KERNELS["note"]}


def _milne_constants(hx, ccl):
    """The constants of the Runge-Kutta start and of the Milne predictor-corrector (intout.f, solin.f); FEFF writes
    3.3, 4.2 and 7.8 as single-precision numbers."""
    a1, a2, a3 = hx * f32(3.3), -(hx * f32(4.2)), hx * f32(7.8)
    a4, a5, a6 = hx * 14.0 / 45.0, hx * 64.0 / 45.0, hx * 24.0 / 45.0
    return dict(ccl=ccl, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, test=MILNE_TEST)


def _milne_outward_plain(gg, gp, dv, vm, dr, ec, kapf, ic3f, av1, iflat0, hx, consts):
    """The Runge-Kutta start and the Milne predictor-corrector of the regular solution (intout.f) in torch: fills
    gg, gp (L, N; point 0 set) up to iflat0 of each lane. `kernels.milne_outward` does the same on the GPU, one
    thread per lane, operation by operation."""
    ccl_i, a1, a2, a3, a4, a5, a6 = (consts[k] for k in ("ccl", "a1", "a2", "a3", "a4", "a5", "a6"))
    exphx = math.exp(hx / 2)
    cz = torch.zeros(gg.shape[0], dtype=CDTYPE, device=gg.device)

    def fg(i):
        f = (ec - dv[:, i]) * dr[:, i]
        return f, f + ccl_i * dr[:, i]

    f, g = fg(0)
    c3 = ic3f * vm[:, 0] / (g * g)
    D = [hx * (g * gp[:, 0] - kapf * gg[:, 0])]
    P = [hx * (kapf * gp[:, 0] - (f - c3) * gg[:, 0])]
    for i in range(NPI - 1):
        ac = gg[:, i] + 0.5 * D[-1]
        bc = gp[:, i] + 0.5 * P[-1]
        rh = dr[:, i] * exphx
        xm1 = (dr[:, i + 1] - rh) / (dr[:, i + 1] - dr[:, i])
        xm2 = (rh - dr[:, i]) / (dr[:, i + 1] - dr[:, i])
        dv1 = dv[:, i] - av1 / dr[:, i]
        dv2 = dv[:, i + 1] - av1 / dr[:, i + 1]
        vh = dv1 * xm1 + dv2 * xm2
        vh = vh + av1 / rh
        vmh = (xm1 * vm[:, i] * dr[:, i] + xm2 * vm[:, i + 1] * dr[:, i + 1]) / rh
        f = (ec - vh) * rh
        g = f + ccl_i * rh
        c3 = ic3f * vmh / (g * g)
        dg2 = hx * (g * bc - kapf * ac)
        dp2 = hx * (kapf * bc - (f - c3) * ac)
        ac = ac + 0.5 * (dg2 - D[-1])
        bc = bc + 0.5 * (dp2 - P[-1])
        dg3 = hx * (g * bc - kapf * ac)
        dp3 = hx * (kapf * bc - (f - c3) * ac)
        ac = ac + dg3 - 0.5 * dg2
        bc = bc + dp3 - 0.5 * dp2
        f, g = fg(i + 1)
        c3 = ic3f * vm[:, i + 1] / (g * g)
        dg4 = hx * (g * bc - kapf * ac)
        dp4 = hx * (kapf * bc - (f - c3) * ac)
        gg[:, i + 1] = gg[:, i] + (D[-1] + 2.0 * (dg2 + dg3) + dg4) / 6.0
        gp[:, i + 1] = gp[:, i] + (P[-1] + 2.0 * (dp2 + dp3) + dp4) / 6.0
        D.append(hx * (g * gp[:, i + 1] - kapf * gg[:, i + 1]))
        P.append(hx * (kapf * gp[:, i + 1] - (f - c3) * gg[:, i + 1]))
    # --- Milne predictor-corrector to iflat
    D = [d / hx for d in D]
    P = [p / hx for p in P]
    for i in range(NPI - 1, int(iflat0.max())):
        acp = gg[:, i - 5] + a1 * (D[5] + D[1]) + a2 * (D[4] + D[2]) + a3 * D[3]
        bcp = gp[:, i - 5] + a1 * (P[5] + P[1]) + a2 * (P[4] + P[2]) + a3 * P[3]
        ac = gg[:, i - 3] + a4 * D[2] + a5 * (D[5] + D[3]) + a6 * D[4]
        bc = gp[:, i - 3] + a4 * P[2] + a5 * (P[5] + P[3]) + a6 * P[4]
        D, P = D[1:], P[1:]
        f, g = fg(i + 1)
        c3 = ic3f * vm[:, i + 1] / (g * g)
        fc = f - c3
        dg6 = g * bcp - kapf * acp
        dp6 = kapf * bcp - fc * acp
        ggn = ac + a4 * dg6
        gpn = bc + a4 * dp6
        valid = (i + 1) <= iflat0
        nit = 0
        while True:
            fail = ((MILNE_TEST * (ggn - acp)).abs() > ggn.abs()) | ((MILNE_TEST * (gpn - bcp)).abs() > gpn.abs())
            fail = fail & valid
            if nit >= 40 or not bool(fail.any()):
                break
            acp = torch.where(fail, ggn, acp)
            bcp = torch.where(fail, gpn, bcp)
            nit += 1
            dg6n = g * bcp - kapf * acp
            dp6n = kapf * bcp - fc * acp
            dg6 = torch.where(fail, dg6n, dg6)
            dp6 = torch.where(fail, dp6n, dp6)
            ggn = torch.where(fail, ac + a4 * dg6n, ggn)
            gpn = torch.where(fail, bc + a4 * dp6n, gpn)
        gg[:, i + 1] = torch.where(valid, ggn, cz)
        gp[:, i + 1] = torch.where(valid, gpn, cz)
        D.append(torch.where(valid, dg6, cz))
        P.append(torch.where(valid, dp6, cz))
    return gg, gp


def dirac_outward(ec, en, kap, ic3, fl, av1, av2, dv, vm, dr, ri, hx, iflat0, iwkb0, jri0, jlast0, rmt, lmaxp1=LTOT + 2):
    """Regular solution of the Dirac equation for a batch of lanes (dfovrg with ncycle = 0, irr = -1): power
    expansion at the origin (solout), Runge-Kutta then Milne predictor-corrector (intout) to iflat, exact flat-
    potential steps to jlast, then the flat step from ri(jri) to the muffin-tin radius. All indices 0-based.
    dr (N,) or (L, N); av1, rmt scalars or (L,); jri0 int or (L,). Returns gg, gp (L, N) and pu, qu (L,)."""
    L, N = dv.shape
    dev = dv.device
    kapf = kap.to(RDTYPE)
    ic3f = ic3.to(RDTYPE)
    dr = _lane_grid(dr, L, dev)
    dr1 = dr[:, 0]
    av1 = torch.as_tensor(av1, dtype=RDTYPE, device=dev)
    rmt = torch.as_tensor(rmt, dtype=RDTYPE, device=dev)
    jri0 = torch.as_tensor(jri0, dtype=torch.long, device=dev)
    if jri0.dim() == 0:
        jri0 = jri0.expand(L)
    ccl_s = 2 * ALPINV
    ccl_i = CL_W + CL_W
    cz = torch.zeros(L, dtype=CDTYPE, device=dev)
    # --- values at the first point
    ap1 = torch.where(kap > 0, -(kapf + fl) / av1, -av1 / (kapf - fl)).to(CDTYPE)
    ag = [torch.ones(L, dtype=CDTYPE, device=dev), cz.clone(), cz.clone()]
    ap = [ap1, cz.clone(), cz.clone()]
    av = [av1, av2, cz]
    for j in (2, 3):
        k = j - 1
        a = fl + kapf + k
        b = fl - kapf + k
        eph = a * b + av1 * av1
        f = (ec + ccl_s) * ap[k - 1] + ap[j - 1]
        g = ec * ag[k - 1] + ag[j - 1]
        for i in range(1, k + 1):
            f = f - av[i] * ap[j - i - 1]
            g = g - av[i] * ag[j - i - 1]
        ag[j - 1] = (b * f + av1 * g) / eph
        ap[j - 1] = (av1 * f - a * g) / eph
    gg1d, gp1d = cz.clone(), cz.clone()
    for j in (1, 2, 3):
        bpow = dr1 ** (fl + j - 1)
        gg1d = gg1d + bpow * ag[j - 1]
        gp1d = gp1d + bpow * ap[j - 1]
    twoz = -av1 * 2.0 * CL_W
    rat1 = twoz / ccl_s
    rat2 = rat1 * rat1
    rat3 = (ccl_s * ccl_s) / twoz
    il = torch.where(kap > 0, kapf + 1, -kapf)
    l0 = il - 1
    ap1f = (fl - il) * rat3 * 1.0
    ag2 = (3.0 * fl - rat2) / (2.0 * fl + 1.0) * 1.0
    ap2 = rat3 * ((fl - l0) * ag2 - 1.0) - ap1f
    ag3 = ((fl + 3.0 * il) * ag2 - 3.0 * l0 * 1.0 + (fl + il + 3.0) / rat3 * ap2) / (fl + 1.0) / 4.0
    ap3 = (rat3 * (2.0 * l0 * (fl + 2.0 - il) - l0 - rat2) * ag2 - 3.0 * l0 * rat3 * (fl + 2.0 - il) * 1.0
           + (fl + 3.0 - 2.0 * il - rat2) * ap2) / (fl + 1.0) / 4.0
    ap1f = ap1f / ccl_s
    ag2 = ag2 * rat3
    ap2 = ap2 * rat3 / ccl_s
    ag3 = ag3 * (rat3 * rat3)
    ap3 = ap3 * (rat3 * rat3) / ccl_s
    dfl = dr1 ** fl
    gg1f = dfl * (1.0 + dr1 * (ag2 + dr1 * ag3))
    gp1f = dfl * (ap1f + dr1 * (ap2 + dr1 * ap3))
    use_d = ic3 == 0
    gg = torch.zeros(L, N, dtype=CDTYPE, device=dev)
    gp = torch.zeros_like(gg)
    gg[:, 0] = torch.where(use_d, gg1d, gg1f.to(CDTYPE))
    gp[:, 0] = torch.where(use_d, gp1d, gp1f.to(CDTYPE))
    # --- Runge-Kutta for the first NPI points (intout), then Milne to iflat: on the GPU one kernel does both
    consts = _milne_constants(hx, ccl_i)
    if kernels_usable(dv):
        from .kernels import milne_outward
        gg, gp = milne_outward(gg, gp, dv, vm, dr.contiguous(), ec, kapf, ic3f, av1.expand(L).contiguous(), iflat0, hx, consts)
    else:
        gg, gp = _milne_outward_plain(gg, gp, dv, vm, dr, ec, kapf, ic3f, av1, iflat0, hx, consts)
    return _flat_steps_out(gg, gp, ec, en, kap, ic3, ic3f, dv, vm, dr, ri, hx, iflat0, iwkb0, jri0, jlast0, rmt, lmaxp1, L, N, dev, ccl_s)


def _free_region(gg, gp, en, kap, dr, jri0, jlast0, lmaxp1, L, dev):
    """The flat-potential steps beyond the muffin tin (potential zero): the chained steps of solout equal one
    evaluation of the free solution from the point jri; done here for all target points in one Bessel call."""
    ar = torch.arange(L, device=dev)
    kmax = int((jlast0 - jri0).max())
    if kmax <= 0:
        return gg, gp
    r1 = dr[ar, jri0]
    p1, q1 = gg[ar, jri0], gp[ar, jri0]
    ck = torch.sqrt(2 * en + (ALPHFS * en) ** 2)
    xkr = ck * r1
    neg = kap < 0
    lp = torch.where(neg, -kap - 1, kap)
    lq = torch.where(neg, lp + 1, lp - 1)
    isign = torch.where(neg, -1.0, 1.0).to(RDTYPE)
    a_ = ck * ALPHFS
    factor = isign * a_ / (1 + torch.sqrt(1 + a_ * a_))
    jl, nl = besjn(xkr, lmaxp1)
    a = isign * ck * xkr * (p1 * nl[ar, lq] - q1 * nl[ar, lp] / factor)
    b = isign * ck * xkr * (q1 * jl[ar, lp] / factor - p1 * jl[ar, lq])
    ks = torch.arange(1, kmax + 1, device=dev)
    idx = jri0[:, None] + ks[None, :]                                        # (L, K) target points
    act = idx <= jlast0[:, None]
    idxc = torch.minimum(idx, torch.full_like(idx, gg.shape[1] - 1))
    r2 = dr[ar[:, None], idxc]
    jl2, nl2 = besjn((ck[:, None] * r2).reshape(-1), lmaxp1)
    arK = torch.arange(L * kmax, device=dev)
    lpK, lqK = lp.repeat_interleave(kmax), lq.repeat_interleave(kmax)
    p2 = (r2.reshape(-1) * (jl2[arK, lpK] * a.repeat_interleave(kmax) + nl2[arK, lpK] * b.repeat_interleave(kmax))).reshape(L, kmax)
    q2 = (r2.reshape(-1) * factor.repeat_interleave(kmax) * (jl2[arK, lqK] * a.repeat_interleave(kmax) + nl2[arK, lqK] * b.repeat_interleave(kmax))).reshape(L, kmax)
    gg[ar[:, None], idxc] = torch.where(act, p2, gg[ar[:, None], idxc])
    gp[ar[:, None], idxc] = torch.where(act, q2, gp[ar[:, None], idxc])
    return gg, gp


def _flat_steps_out(gg, gp, ec, en, kap, ic3, ic3f, dv, vm, dr, ri, hx, iflat0, iwkb0, jri0, jlast0, rmt, lmaxp1, L, N, dev, ccl_s):
    # --- flat-potential steps from iflat to jri inside the muffin tin (solout), chained
    for i in range(int(iflat0.min()), int(jri0.max())):
        act = (i >= iflat0) & (i < jlast0) & (i < jri0)
        if not bool(act.any()):
            continue
        idx = act.nonzero().squeeze(1)
        eph_a = CL_W * (dv[idx, i] + dv[idx, i + 1]) / 2
        if i + 2 < N:
            eph_w = CL_W * (3 * dv[idx, i + 1] - dv[idx, i + 2]) / 2
            eph = torch.where((iwkb0[idx] == i) & (iwkb0[idx] != jri0[idx] - 1), eph_w, eph_a)
        else:
            eph = eph_a
        rav = (dr[idx, i] + dr[idx, i + 1]) / 2
        t = ccl_s + (en[idx] - eph) / CL_W
        ec2 = powi(rav, 3) * (t * t)
        eph_c = eph + ic3f[idx] * CL_W / ec2 * (vm[idx, i] + vm[idx, i + 1]) / 2
        eph = torch.where((ic3[idx] > 0) & (i < jri0[idx]), eph_c, eph)
        p2n, q2n = flatv(dr[idx, i], dr[idx, i + 1], gg[idx, i], gp[idx, i], en[idx], eph, kap[idx], lmaxp1)
        gg[idx, i + 1] = p2n
        gp[idx, i + 1] = q2n
    gg, gp = _free_region(gg, gp, en, kap, dr, jri0, jlast0, lmaxp1, L, dev)
    ar = torch.arange(L, device=dev)
    pu, qu = flatv(ri[jri0], rmt, gg[ar, jri0], gp[ar, jri0], en, torch.zeros_like(en), kap, lmaxp1)
    return gg, gp, pu, qu


def _lane_setup(p2, dx, idim, jri, jlast):
    """iwkb and iflat (1-based like FEFF) per energy, for a given last point of the integration."""
    rwkb = 0.5 / dx / torch.sqrt((2 * p2 + (p2 / ALPINV) * (p2 / ALPINV)).abs())
    iwkb = torch.floor((torch.log(rwkb) + f32(8.8)) / dx).long() + 2
    iwkb = torch.clamp(iwkb, min=10, max=idim)
    iwkb = torch.where(iwkb >= jlast - 1, torch.full_like(iwkb, idim), iwkb)
    iflat = torch.minimum(iwkb, torch.as_tensor(jri, device=iwkb.device, dtype=iwkb.dtype))
    return iwkb, iflat


class _Potential:
    """One unique potential on the phase grid with the pieces the solver needs."""

    def __init__(self, pot: PotData, iph, ihole_eff, dx, dxnew, device):
        self.iph = iph
        self.rmt = float(pot.rmt[iph])
        self.rnrm = float(pot.rnrm[iph])
        self.iz = int(pot.iz[iph])
        ri, vtotph, rhoph, dmagx, self.jrinew = fixvar(self.rmt, pot.edens[:, iph], pot.vtot[:, iph], pot.dmag[:, iph],
                                                       pot.vint, pot.rhoint, dx, dxnew, pot.jumprm)
        self.ri = torch.tensor(ri, dtype=RDTYPE, device=device)
        self.vtotph = torch.tensor(vtotph, dtype=RDTYPE, device=device)
        self.rhoph = torch.tensor(rhoph, dtype=RDTYPE, device=device)
        self.imt = int((math.log(self.rmt) + X0) / dxnew) + 1
        self.jri = self.imt + 1
        orb = getorb(self.iz, ihole_eff, float(pot.xion[iph]), pot.iunf)
        self.norbco = orb["norb"]
        self.nk, self.xnel = orb["nk"], orb["xnel"]
        self.idim = 1 + int(round(250 * 0.05 / dxnew))
        if self.idim > NRPTX:
            self.idim = NRPTX
        if self.idim % 2 == 0:
            self.idim -= 1
        self.np_ = 1 + int((f32(8.8) + f32(math.log(10.0))) / dxnew)
        if self.idim < self.np_:
            self.np_ = self.idim
        self.dr = torch.tensor(nucdec_grid(self.iz, dxnew, self.idim), dtype=RDTYPE, device=device)
        self.dz = float(self.iz)
        self.av1 = (-self.dz) / CL_W
        self.ap1 = origin_density_term(pot.adgc[:, :, iph], pot.adpc[:, :, iph], self.nk, self.xnel, self.dz, float(self.dr[0]))
        self.hx = dxnew
        self.device = device

    def potential(self, ixc, lreal, em, xmu):
        """Complex potential relative to the interstitial, its reference, and the solver's dv/vm bases."""
        v, eref = xcpot(ixc, lreal, em, xmu, self.vtotph, self.rhoph, self.jri)
        dv, vmb, av2 = self.bases(v)
        return v, eref, dv, vmb, av2

    def bases(self, v):
        """The solver's inputs from a potential v (E, jri+1) relative to its interstitial value."""
        E = v.shape[0]
        jri0 = self.jri - 1
        N = self.idim
        dv = torch.zeros(E, N, dtype=CDTYPE, device=self.device)
        dv[:, :jri0] = v[:, :jri0] / CL_W
        dv[:, jri0:] = (v[:, jri0 + 1] / CL_W)[:, None]
        vmb = torch.zeros(E, N, dtype=CDTYPE, device=self.device)
        vmb[:, :self.jri] = vm_derivative(v, self.ri, self.hx, self.jri)
        vmb[:, jri0:] = 0
        av2 = (self.ap1 / CL_W) + (v[:, 0] - (-self.dz / self.dr[0])) / CL_W
        return dv, vmb, av2

    def solve(self, en, dv, vmb, av2, kap, ic3, jlast, ie_of):
        """Lanes: en (L,) complex energies (p2), kap, ic3 (L,) long, jlast (L,) 1-based, ie_of (L,) energy index."""
        a = (self.dz / CL_W) ** 2
        kapf, ic3f = kap.to(RDTYPE), ic3.to(RDTYPE)
        fl = torch.sqrt((kapf * kapf - a) + (kapf + 1) * ic3f)
        jlast = torch.clamp(jlast, max=self.np_)
        iwkb, iflat = _lane_setup(en, self.hx, self.idim, self.jri, jlast)
        vm = vmb[ie_of] * (kapf + 1.0)[:, None] / ALPINV
        gg, gp, pu, qu = dirac_outward(en / CL_W, en, kap, ic3, fl, self.av1, av2[ie_of], dv[ie_of], vm, self.dr,
                                       self.ri, self.hx, iflat - 1, iwkb - 1, self.jri - 1, jlast - 1, self.rmt,
                                       getattr(self, "bessel_orders", LTOT + 2))
        return gg, gp, pu, qu


def phase_shifts(P: _Potential, ixc, lreal, em, xmu, lmax, ne1):
    """eref (E,) and the phase shifts ph (E, 2 lmax + 1) of one unique potential (XSPH/phase.f)."""
    E, dev = em.shape[0], P.device
    v, eref, dv, vmb, av2 = P.potential(ixc, lreal, em, xmu)
    p2 = em - eref
    if lreal > 1:
        p2 = torch.cat([p2[:ne1].real.to(CDTYPE), p2[ne1:]])
    ck = torch.sqrt(2 * p2 + (p2 * ALPHFS) * (p2 * ALPHFS))
    xkmt = P.rmt * ck
    skip = (em.real < -10.0) | (em.real > 300.0) | ((p2.real <= 0) & (p2.imag <= 0))
    jl_all, nl_all = besjn(xkmt)
    ls = torch.arange(lmax + 1, device=dev)
    ie_of = torch.arange(E, device=dev).repeat_interleave(lmax + 1)
    l_of = ls.repeat(E)
    kap = -(l_of + 1)
    ic3 = (l_of != 0).long()
    jlast = torch.full_like(kap, P.jri)
    gg, gp, pu, qu = P.solve(p2[ie_of], dv, vmb, av2, kap, ic3, jlast, ie_of)
    ph, _ = phamp(P.rmt, pu, qu, ck[ie_of], jl_all[ie_of, l_of], nl_all[ie_of, l_of], jl_all[ie_of, l_of + 1],
                  nl_all[ie_of, l_of + 1], kap)
    ph = ph.reshape(E, lmax + 1)
    ph = torch.where(skip[:, None], torch.zeros_like(ph), ph)
    out = torch.zeros(E, 2 * lmax + 1, dtype=CDTYPE, device=dev)
    broken = torch.zeros(E, dtype=torch.bool, device=dev)
    zero = torch.zeros(E, dtype=CDTYPE, device=dev)
    for ll in range(-lmax, lmax + 1):
        val = ph[:, abs(ll)]
        cond1 = (val.abs() < f32(1.0e-6)) & (ll >= 4)
        rivas = (torch.exp(2j * val) - 1.0).abs() < f32(1.0e-5)
        val2 = torch.where(rivas, zero, val)
        cond2 = (val2.abs() < f32(1.0e-5)) & (ll >= 4)
        keep = torch.where(cond1, val, val2)
        out[:, ll + lmax] = torch.where(broken, zero, keep)
        broken = broken | cond1 | cond2
    # lanes whose radial solution overflowed (very high l at high energy, inside the flat-potential steps) are
    # not finite in FEFF either; its packed writer stores them as zero
    out = torch.where(torch.isfinite(out.real) & torch.isfinite(out.imag), out, torch.zeros_like(out))
    return eref, out


# ------------------------------------------------------------------ dipole matrix elements (XSPH/xsect.f, radint.f, xmult.f, MATH/cwig3j.f, csomm.f)
_AL = None


def cwig3j(j1, j2, j3, m1, m2, ient):
    global _AL
    idim = 58
    if _AL is None:
        _AL = [0.0]
        for i in range(1, idim + 1):
            _AL.append(_AL[-1] + math.log(float(i)))
    al = _AL
    m3 = -m1 - m2
    if ient not in (1, 2):
        raise ValueError("cwig3j ient")
    ii = ient + ient
    if (abs(m1) + abs(m2)) == 0 and (j1 + j2 + j3) % ii != 0:
        return 0.0
    m = [j1 + j2 - j3, j2 + j3 - j1, j3 + j1 - j2, j1 + m1, j1 - m1, j2 + m2, j2 - m2, j3 + m3, j3 - m3,
         j1 + j2 + j3 + ient, j2 - j3 - m1, j1 - j3 + m2]
    for i in range(1, 13):
        if i <= 10 and m[i - 1] < 0:
            return 0.0
        if m[i - 1] % ient != 0:
            raise ValueError("cwig3j parity")
        m[i - 1] = m[i - 1] // ient if m[i - 1] >= 0 else -((-m[i - 1]) // ient)
        if m[i - 1] > idim:
            raise ValueError("cwig3j range")
    max0 = max(m[10], m[11], 0) + 1
    min0 = min(m[0], m[4], m[5]) + 1
    isig = 1
    if (max0 - 1) % 2 != 0:
        isig = -isig
    c = -al[m[9]]
    for i in range(1, 10):
        c = c + al[m[i - 1]]
    c = c / 2.0
    out = 0.0
    for i in range(max0, min0 + 1):
        j = 2 - i
        b = al[i - 1] + al[j + m[0] - 1] + al[j + m[4] - 1] + al[j + m[5] - 1] + al[i - m[10] - 1] + al[i - m[11] - 1]
        out = out + isig * math.exp(c - b)
        isig = -isig
    if (j1 - j2 - m3) % ii != 0:
        out = -out
    return out


def sixj(j1, j2, j3, j4, j5):
    aa = 0.0
    if j2 == j1 + 1:
        if j4 == j3 + 1:
            g2 = j5 - 1
            if abs(j1 - j3) <= g2 <= j1 + j3:
                aa = (1.0 + (g2 + j1 - j3) / 2.0) * (1.0 + (g2 - j1 + j3) / 2.0) / (j1 + 1) / (j1 + 2) / (j3 + 1) / (j3 + 2)
                aa = math.sqrt(aa) * (-1) ** (int(round(1 + (g2 + j1 + j3) / 2.0)))
        elif j3 == j4 + 1:
            g2 = j5
            if abs(j1 - j4) <= g2 <= j1 + j4:
                aa = (1.0 - (g2 - j1 - j4) / 2.0) * (2.0 + (g2 + j1 + j4) / 2.0) / (j1 + 1) / (j1 + 2) / (j4 + 1) / (j4 + 2)
                aa = math.sqrt(aa) * (-1) ** (int(round(1 + (g2 + j1 + j4) / 2.0)))
    return aa


def ninej(lam, lamp, ls, j2, jp2, lb):
    if ls > lb:
        return -(ls + lb + 1) * sixj(1, 2, 2 * lb, ls + lb, 2 * ls) * sixj(2 * lb, ls + lb, 2 * lamp, jp2, j2) * \
            sixj(ls + lb, 2 * ls, 2 * lam, j2, 2 * lamp)
    if ls < lb:
        return -(ls + lb + 1) * sixj(1, 2, 2 * lb, ls + lb, 2 * ls) * sixj(ls + lb, 2 * lb, jp2, 2 * lamp, j2) * \
            sixj(2 * ls, ls + lb, j2, 2 * lam, 2 * lamp)
    aa = -(2 * ls + 2) * sixj(1, 2, 2 * lb, 2 * lb + 1, 2 * lb) * sixj(2 * lb, 2 * lb + 1, 2 * lamp, jp2, j2) * \
        sixj(2 * lb, 2 * lb + 1, j2, 2 * lam, 2 * lamp)
    aa = aa - (2 * ls) * sixj(1, 2, 2 * lb, 2 * lb - 1, 2 * lb) * sixj(2 * lb - 1, 2 * lb, jp2, 2 * lamp, j2) * \
        sixj(2 * lb - 1, 2 * lb, 2 * lam, j2, 2 * lamp)
    return aa


def xmult(k, kp, ls, lb):
    """Grant's angular factors for the multipole matrix elements (XSPH/xmult.f): (xm1, xm2)."""
    coni = 1j
    if ls + 1 == lb:
        aa = (2 * lb - 1) * (lb + 1) / 2.0
        alslb = powi(coni, ls) * math.sqrt(aa)
    elif ls - 1 == lb:
        aa = (2 * lb + 3) * lb / 2.0
        alslb = powi(coni, ls) * math.sqrt(aa)
    elif ls == lb:
        alslb = powi(coni, ls) * (2 * lb + 1) / math.sqrt(2.0)
    else:
        alslb = 0.0
    j2 = 2 * abs(k) - 1
    a = -1 if k > 0 else 1
    jp2 = 2 * abs(kp) - 1
    ap = -1 if kp > 0 else 1

    def part(lam2, lamp2):
        lam, lamp = lam2 // 2, lamp2 // 2
        if 2 * lam == lam2 and 2 * lamp == lamp2:
            aa = ninej(lam, lamp, ls, j2, jp2, lb)
            return alslb * aa * cwig3j(lam, ls, lamp, 0, 0, 1) * (-1) ** lam * \
                math.sqrt(6.0 * (j2 + 1) * (jp2 + 1) * (2 * lb + 1) * (2 * lam + 1) * (2 * lamp + 1))
        return 0.0

    xm1 = part(j2 - a, jp2 + ap) * coni
    xm2 = -coni * part(j2 + a, jp2 - ap)
    return complex(xm1), complex(xm2)


def csomm(dr, dp, dpas, da, m, npts):
    """Simpson integration of dp dr^m from 0 to dr(npts) with FEFF's end corrections; dp (E, >= npts) complex,
    da the power of dp near the origin. Returns (E,)."""
    mm = m + 1
    d1 = float(da + mm)
    acc = torch.zeros(dp.shape[0], dtype=dp.dtype, device=dp.device)
    for i in range(1, npts + 1):
        dl = powi(float(dr[i - 1]), mm)
        if i != 1 and i != npts:
            dl = dl + dl
            if i % 2 == 0:
                dl = dl + dl
        acc = acc + dp[:, i - 1] * dl
    acc = dpas * acc / 3
    dd = math.exp(dpas) - 1.0
    db = d1 * (d1 + 1.0) * dd * math.exp((d1 - 1.0) * dpas)
    db = float(dr[0]) * powi(float(dr[1]), m) / db
    dd = powi(float(dr[0]), mm) * (1.0 + 1.0 / (dd * (d1 + 1.0))) / d1
    return acc + dd * dp[:, 0] - db * dp[:, 1]


def dipole_elements(P: _Potential, ixc, lreal, em, edge, emu, xmu, ihole, dgcx, dpcx, jnew, le2=0):
    """rkk (E, 8) and xsnorm (E,) of the absorber (XSPH/xsect.f, electric dipole; quadrupole not ported)."""
    E, dev = em.shape[0], P.device
    v, eref, dv, vmb, av2 = P.potential(ixc, lreal, em, xmu)
    p2 = em - eref
    ck = torch.sqrt(2 * p2 + (p2 * ALPHFS) * (p2 * ALPHFS))
    xkmt = P.rmt * ck
    omega = (em.real - edge) + emu
    omega = torch.clamp(omega, min=0.001 / HART)
    xk0 = omega * ALPHFS
    inrm = int((math.log(P.rnrm) + X0) / P.hx) + 1
    jnrm = inrm + 1
    ilast = max(jnrm + 6, jnew)
    ilast = min(ilast, NRPTX)
    ri = P.ri
    temp = xk0[:, None] * ri[None, :ilast]
    small = temp.abs() < 1.0
    bf = []
    x = temp
    sinx, cosx = torch.sin(x), torch.cos(x)
    xs = torch.where(small, torch.ones_like(x), x)
    formulas = [sinx / xs, sinx / (xs * xs) - cosx / xs, sinx * (3 / powi(xs, 3) - 1 / xs) - 3 * cosx / (xs * xs)]
    for ll in range(3):
        ser = bjnser(temp.to(CDTYPE), ll, 1)[0].real
        bf.append(torch.where(small, ser, formulas[ll]))
    kinit, linit = setkap(ihole)
    kinds = []
    for k in (-1, 0, 1):
        kap = kinit + k
        if k == 0:
            kap = -kap
        lkap = kap if kap > 0 else abs(kap) - 1
        if kap == 0 or lkap > 4:
            kap = 0
        kinds.append(kap)
    rkk = torch.zeros(E, 8, dtype=CDTYPE, device=dev)
    phx = torch.zeros(E, 8, dtype=CDTYPE, device=dev)
    xsnorm = torch.zeros(E, dtype=RDTYPE, device=dev)
    skip = (em.real < -10.0) | ((p2.imag <= 0) & (p2.real <= 0))
    ie_of = torch.arange(E, device=dev)
    for ind, ikap in enumerate(kinds, start=1):
        if ikap == 0:
            continue
        lfin = ikap if ikap > 0 else -ikap - 1
        ilp = lfin - 1 if ikap > 0 else lfin + 1
        kap = torch.full((E,), ikap, dtype=torch.long, device=dev)
        ic3 = torch.zeros(E, dtype=torch.long, device=dev)
        jlast = torch.full((E,), ilast, dtype=torch.long, device=dev)
        gg, gp, pu, qu = P.solve(p2, dv, vmb, av2, kap, ic3, jlast, ie_of)
        jl, nl = exjlnl(xkmt, lfin)
        jlp1, nlp1 = exjlnl(xkmt, ilp)
        ph0, amp = phamp(P.rmt, pu, qu, ck, jl, nl, jlp1, nlp1, kap)
        sign = 1.0 if ikap > 0 else -1.0
        factor = ck * ALPHFS
        factor = sign * factor / (1 + torch.sqrt(1 + factor * factor))
        dum1 = 1 / torch.sqrt(1 + factor * factor)
        xfnorm = 1 / amp * dum1
        p = gg[:, :ilast] * xfnorm[:, None]
        q = gp[:, :ilast] * xfnorm[:, None]
        xm1, xm2 = xmult(ikap, kinit, 0, 1)
        xm3, xm4 = xmult(ikap, kinit, 2, 1)
        dg = torch.tensor(dgcx[:ilast], dtype=RDTYPE, device=dev)[None]
        dp_ = torch.tensor(dpcx[:ilast], dtype=RDTYPE, device=dev)[None]
        xrc = dg * q * (xm2 * bf[0] + xm4 * bf[2]) + dp_ * p * (xm1 * bf[0] + xm3 * bf[2])
        xirf = csomm(ri, xrc, P.hx, float(lfin + linit + 2), 0, ilast)
        xirf = torch.where(skip, torch.zeros_like(xirf), xirf)
        rkk[:, ind - 1] = xirf
        phx[:, ind - 1] = torch.where(skip, torch.zeros_like(ph0), ph0)
        xsnorm = xsnorm + (xirf.real * xirf.real + xirf.imag * xirf.imag) / 3
    prefac = 4 * PI * ALPINV / omega * (BOHR * BOHR)
    xsnorm = xsnorm * prefac * 2 * ck.abs()
    xnorm = torch.sqrt(xsnorm)
    ckp = torch.sqrt(prefac * (2 * ck))
    ckp = torch.where(ckp.imag < 0, -ckp, ckp)
    ok = ~skip & (omega > 0)
    for kdif in range(8):
        rkk[:, kdif] = torch.where(ok, rkk[:, kdif] * ckp / xnorm * torch.exp(1j * phx[:, kdif]), torch.zeros_like(ckp))
    xsnorm = torch.where(ok, xsnorm, torch.zeros_like(xsnorm))
    return rkk, xsnorm


# ------------------------------------------------------------------ driver (XSPH/xsph.f)
def xsph(pot: PotData, gamach, xkmax, ixc=0, vr0=0.0, vi0=0.0, lreal=0, rgrd=f32(0.05), lmax_override=None, ixc0=0,
         potlbl=None, device="cpu"):
    """The phase-shift stage: pot.pad content in, phase.pad content out. gamach in Hartree, xkmax in 1/bohr,
    vr0/vi0 in Hartree (FEFF converts the input-file values before this point)."""
    dx, dxnew = 0.05, float(rgrd)
    edge = pot.xmu - vr0
    emu = pot.emu - vr0
    xmuvr = pot.xmu - vr0
    em_np, ne, ne1, ik0 = phmesh_exafs(edge, emu, vi0, gamach, xkmax)
    em = torch.tensor(em_np, dtype=CDTYPE, device=device)
    xkmax_e = math.sqrt(max(em_np.real[:ne].max(), 0.0) * 2)
    phs, lmaxs, eref = [], [], None
    for iph in range(pot.nph + 1):
        itmp = pot.ihole if iph == 0 else 0
        P = _Potential(pot, iph, itmp, dx, dxnew, device)
        if lmax_override is None:
            lmax = max(int(0.7 * P.rmt * xkmax_e), 5)
        else:
            lmax = int(lmax_override)
        lmax = min(lmax, LTOT)
        if iph == 0:
            dgcx, dpcx, jnew = fixdsp(dx, dxnew, pot.dgc0, pot.dpc0)
            # FEFF computes the matrix elements with the EXCHANGE card's fourth value ixc0 (default 0, Hedin-Lundqvist),
            # whatever the exchange model of the phase shifts (xsph.f: xsect(..., ixc0, ...))
            rkk, xsnorm = dipole_elements(P, ixc0, lreal, em, edge, emu, xmuvr, pot.ihole, dgcx, dpcx, jnew)
        eref, ph = phase_shifts(P, ixc, lreal, em, xmuvr, lmax, ne1)
        phs.append(ph.cpu().numpy())
        lmaxs.append(lmax)
    ltot = max(lmaxs)
    ph = np.zeros((ne, 2 * ltot + 1, 1, pot.nph + 1), complex)
    for iph in range(pot.nph + 1):
        lo = ltot - lmaxs[iph]
        ph[:, lo:lo + 2 * lmaxs[iph] + 1, 0, iph] = phs[iph]
    labels = potlbl if potlbl is not None else [""] * (pot.nph + 1)
    return PhaseData(1, ne, ne1, 0, pot.nph, pot.ihole, ik0, ixc, pot.rs, pot.vint, pot.rnrmav, xmuvr, edge, em_np,
                     eref.cpu().numpy()[:, None], np.array(lmaxs), np.array(pot.iz[:pot.nph + 1]), labels, ph,
                     rkk.cpu().numpy()[:, :, None], ltot), xsnorm.cpu().numpy()


# ------------------------------------------------------------------ irregular solution (FOVRG/solin.f, MATH/besjh.f)
def besjh(x, lbmax):
    """jl and hl^(+/-) (Messiah, sign by Im x) for orders 0..lbmax on a batch (M,) -> (M, lbmax+1)."""
    M, dev = x.shape[0], x.device
    lmax = min(lbmax, LTOT + 1)
    lmaxp1 = lmax + 1
    jl = torch.zeros(M, lmaxp1, dtype=x.dtype, device=dev)
    hl = torch.zeros_like(jl)
    rex, aix = x.real, x.imag.abs()
    A = (rex < 1.0) & (aix < 1.0)
    B = ~A & (rex < 7.51) & (aix < 7.51)
    Cc = ~A & ~B
    if bool(A.any()):
        xa = x[A]
        for ll in range(lmaxp1):
            j_, n_ = bjnser(xa, ll, 0)
            jl[A, ll] = j_
            hl[A, ll] = -n_ + 1j * j_
    if bool(B.any()):
        xb = x[B]
        jb = [None] * lmaxp1
        nb = [None] * lmaxp1
        jb[lmax] = bjnser(xb, lmax, 1)[0]
        jb[lmax - 1] = bjnser(xb, lmax - 1, 1)[0]
        small = (xb.real < 5.01) & (xb.imag.abs() < 5.01)
        n0s, n1s = torch.zeros_like(xb), torch.zeros_like(xb)
        if bool(small.any()):
            n0s[small] = bjnser(xb[small], 0, 2)[1]
            n1s[small] = bjnser(xb[small], 1, 2)[1]
        asx, acx = torch.sin(xb), torch.cos(xb)
        xi = 1 / xb
        xi2 = xi * xi
        nb[0] = torch.where(small, n0s, -acx * xi)
        nb[1] = torch.where(small, n1s, -acx * xi2 - asx * xi)
        for lp1 in range(3, lmaxp1 + 1):
            l = lp1 - 2
            tlxp1 = 2 * l + 1
            nb[lp1 - 1] = tlxp1 * nb[lp1 - 2] / xb - nb[lp1 - 3]
        for lxx in range(3, lmaxp1 + 1):
            lp1 = lmaxp1 + 1 - lxx
            l = lp1 - 1
            tlxp3 = 2 * l + 3
            jb[l] = tlxp3 * jb[l + 1] / xb - jb[l + 2]
        for il in range(1, lmaxp1 + 1):
            l = il - 1
            jl[B, l] = jb[l]
            hl[B, l] = -nb[il - 1] + 1j * jb[l]
    if bool(Cc.any()):
        xc = x[Cc]
        xi = 1 / xc
        xi2 = xi * xi; xi3 = xi * xi2; xi4 = xi * xi3; xi5 = xi * xi4; xi6 = xi * xi5; xi7 = xi * xi6
        xi8 = xi * xi7; xi9 = xi * xi8; xi10 = xi * xi9; xi11 = xi * xi10
        sjl = [xi, xi2, 3 * xi3 - xi, 15 * xi4 - 6 * xi2, 105 * xi5 - 45 * xi3 + xi, 945 * xi6 - 420 * xi4 + 15 * xi2,
               10395 * xi7 - 4725 * xi5 + 210 * xi3 - xi, 135135 * xi8 - 62370 * xi6 + 3150 * xi4 - 28 * xi2,
               2027025 * xi9 - 945945 * xi7 + 51975 * xi5 - 630 * xi3 + xi,
               34459425 * xi10 - 16216200 * xi8 + 945945 * xi6 - 13860 * xi4 + 45 * xi2,
               654729075 * xi11 - 310134825 * xi9 + 18918900 * xi7 - 315315 * xi5 + 1485 * xi3 - xi]
        cjl = [torch.zeros_like(xi), -xi, -3 * xi2, -15 * xi3 + xi, -105 * xi4 + 10 * xi2, -945 * xi5 + 105 * xi3 - xi,
               -10395 * xi6 + 1260 * xi4 - 21 * xi2, -135135 * xi7 + 17325 * xi5 - 378 * xi3 + xi,
               -2027025 * xi8 + 270270 * xi6 - 6930 * xi4 + 36 * xi2,
               -34459425 * xi9 + 4729725 * xi7 - 135135 * xi5 + 990 * xi3 - xi,
               -654729075 * xi10 + 91891800 * xi8 - 2837835 * xi6 + 25740 * xi4 - 55 * xi2]
        for lp1 in range(12, lmaxp1 + 1):
            l = lp1 - 2
            tlxp1 = float(2 * l + 1)
            sjl.append(tlxp1 * xi * sjl[lp1 - 2] - sjl[lp1 - 3])
            cjl.append(tlxp1 * xi * cjl[lp1 - 2] - cjl[lp1 - 3])
        asx, acx = torch.sin(xc), torch.cos(xc)
        pos = xc.imag >= 0
        epx = torch.where(pos, torch.exp(1j * xc), torch.exp(-1j * xc))
        for ll in range(lmaxp1):
            jl[Cc, ll] = asx * sjl[ll] + acx * cjl[ll]
            hl[Cc, ll] = torch.where(pos, (sjl[ll] + 1j * cjl[ll]) * epx, (sjl[ll] - 1j * cjl[ll]) * epx)
    return jl, hl


def _milne_inward_plain(gg, gp, D, P, dv, vm, dr, ec, kapf, ic3f, iflat0, hx, consts):
    """The Milne predictor-corrector of the irregular solution from iflat0 of each lane to the origin (solin.f) in
    torch. gg, gp (L, N) are set from iflat0 outward; D, P: the six derivative seeds, lists of (L,).
    `kernels.milne_inward` does the same on the GPU."""
    ccl, a1, a2, a3, a4, a5, a6 = (consts[k] for k in ("ccl", "a1", "a2", "a3", "a4", "a5", "a6"))

    def fg(i):
        f = (ec - dv[:, i]) * dr[:, i]
        return f, f + ccl * dr[:, i]

    for i in range(int(iflat0.max()), 0, -1):                   # 0-based i: computes point i - 1
        valid = i <= iflat0
        acp = gg[:, i + 5] + a1 * (D[5] + D[1]) + a2 * (D[4] + D[2]) + a3 * D[3]
        bcp = gp[:, i + 5] + a1 * (P[5] + P[1]) + a2 * (P[4] + P[2]) + a3 * P[3]
        ac = gg[:, i + 3] + a4 * D[2] + a5 * (D[5] + D[3]) + a6 * D[4]
        bc = gp[:, i + 3] + a4 * P[2] + a5 * (P[5] + P[3]) + a6 * P[4]
        Dn, Pn = D[1:], P[1:]
        f, g = fg(i - 1)
        c3 = ic3f * vm[:, i - 1] / (g * g)
        fc = f - c3
        dg6 = -(g * bcp - kapf * acp)
        dp6 = -(kapf * bcp - fc * acp)
        ggn = ac + a4 * dg6
        gpn = bc + a4 * dp6
        nit = 0
        while True:
            fail = ((MILNE_TEST * (ggn - acp)).abs() > ggn.abs()) | ((MILNE_TEST * (gpn - bcp)).abs() > gpn.abs())
            fail = fail & valid
            if nit >= 40 or not bool(fail.any()):
                break
            acp = torch.where(fail, ggn, acp)
            bcp = torch.where(fail, gpn, bcp)
            nit += 1
            dg6n = -(g * bcp - kapf * acp)
            dp6n = -(kapf * bcp - fc * acp)
            dg6 = torch.where(fail, dg6n, dg6)
            dp6 = torch.where(fail, dp6n, dp6)
            ggn = torch.where(fail, ac + a4 * dg6n, ggn)
            gpn = torch.where(fail, bc + a4 * dp6n, gpn)
        gg[:, i - 1] = torch.where(valid, ggn, gg[:, i - 1])
        gp[:, i - 1] = torch.where(valid, gpn, gp[:, i - 1])
        # lanes that have not started their inward Milne yet (i > iflat0) keep their derivative seeds
        D = [torch.where(valid, Dn[k], D[k]) for k in range(5)] + [torch.where(valid, dg6, D[5])]
        P = [torch.where(valid, Pn[k], P[k]) for k in range(5)] + [torch.where(valid, dp6, P[5])]
    return gg, gp


def dirac_inward(ec, en, kap, ic3, dv, vm, dr, hx, iflat0, iwkb0, jri0, imax0, lmaxp1=LTOT + 2):
    """Irregular solution (dfovrg with irr = 1): outgoing Hankel solution from jri to imax, flat-potential steps
    inward to iflat, Milne predictor-corrector inward to the origin (solin.f). 0-based indices. Returns gg, gp."""
    L, N = dv.shape
    dev = dv.device
    kapf = kap.to(RDTYPE)
    ic3f = ic3.to(RDTYPE)
    ccl = CL_W + CL_W
    cz = torch.zeros(L, dtype=CDTYPE, device=dev)
    gg = torch.zeros(L, N, dtype=CDTYPE, device=dev)
    gp = torch.zeros_like(gg)
    dr = _lane_grid(dr, L, dev)
    jri0 = torch.as_tensor(jri0, dtype=torch.long, device=dev)
    if jri0.dim() == 0:
        jri0 = jri0.expand(L)
    ar = torch.arange(L, device=dev)
    vmh = CL_W * dv[ar, jri0 + 1]
    ev = en - vmh
    ck = torch.sqrt(2 * ev + (ALPHFS * ev) ** 2)
    neg = kap < 0
    il = torch.where(neg, kap.abs() - 1, kap.abs())
    ilp = torch.where(neg, il + 1, il - 1)
    ilx = int(torch.maximum(il, ilp).max()) + 1
    xsign = torch.where(neg, -1.0, 1.0).to(RDTYPE)
    factor = ck * ALPHFS
    factor = xsign * factor / (1 + torch.sqrt(1 + factor * factor))
    dum1 = 1 / torch.sqrt(1 + factor * factor)
    D = [None] * NPI
    P = [None] * NPI

    def fg(i):
        f = (ec - dv[:, i]) * dr[:, i]
        return f, f + ccl * dr[:, i]

    imax_all = int(imax0.max())
    i_lo = int(jri0.min())
    K = imax_all - i_lo + 1
    ks = torch.arange(i_lo, imax_all + 1, device=dev)
    idx = ks[None, :].expand(L, -1)
    actK = (idx <= imax0[:, None]) & (idx >= jri0[:, None])
    rK = dr[ar[:, None], idx]
    jlK, hlK = besjh((ck[:, None] * rK).reshape(-1), ilx)
    arK = torch.arange(L * K, device=dev)
    hl_il = hlK[arK, il.repeat_interleave(K)].reshape(L, K)
    hl_ilp = hlK[arK, ilp.repeat_interleave(K)].reshape(L, K)
    g_new = hl_il * rK * dum1[:, None]
    p_new = hl_ilp * rK * (dum1 * factor)[:, None]
    gg[ar[:, None], idx] = torch.where(actK, g_new, torch.zeros_like(g_new))
    gp[ar[:, None], idx] = torch.where(actK, p_new, torch.zeros_like(p_new))
    for i in range(i_lo, imax_all + 1):
        act = actK[:, i - i_lo]
        j = iflat0 + NPI - i                                     # 1-based j of FEFF: iflat + npi - i with both 1-based
        f, g = fg(i)
        c3 = ic3f * vm[:, i] / (g * g)
        dgi = -(g * gp[:, i] - kapf * gg[:, i])
        dpi = -(kapf * gp[:, i] - (f - c3) * gg[:, i])
        for jj in range(1, NPI + 1):
            sel = (j == jj) & act
            D[jj - 1] = torch.where(sel, dgi, D[jj - 1] if D[jj - 1] is not None else cz)
            P[jj - 1] = torch.where(sel, dpi, P[jj - 1] if P[jj - 1] is not None else cz)
    for jj in range(NPI):
        if D[jj] is None:
            D[jj], P[jj] = cz.clone(), cz.clone()
    for i in range(int(jri0.max()) - 1, int(iflat0.min()) - 1, -1):
        act = (i >= iflat0) & (i < jri0)
        if not bool(act.any()):
            continue
        idx = act.nonzero().squeeze(1)
        eph_a = CL_W * (dv[idx, i] + dv[idx, i + 1]) / 2
        if i + 2 < N:
            eph_w = CL_W * (3 * dv[idx, i + 1] - dv[idx, i + 2]) / 2
            eph = torch.where((iwkb0[idx] == i) & (iwkb0[idx] != jri0[idx] - 1), eph_w, eph_a)
        else:
            eph = eph_a
        rav = (dr[idx, i] + dr[idx, i + 1]) / 2
        t = ccl + (en[idx] - eph) / CL_W
        ec2 = powi(rav, 3) * (t * t)
        eph_c = eph + ic3f[idx] * CL_W / ec2 * (vm[idx, i] + vm[idx, i + 1]) / 2
        eph = torch.where(ic3[idx] > 0, eph_c, eph)
        p2n, q2n = flatv(dr[idx, i + 1], dr[idx, i], gg[idx, i + 1], gp[idx, i + 1], en[idx], eph, kap[idx], lmaxp1)
        gg[idx, i] = p2n
        gp[idx, i] = q2n
        j = iflat0 + NPI - i
        f, g = fg(i)
        c3 = ic3f * vm[:, i] / (g * g)
        dgi = -(g * gp[:, i] - kapf * gg[:, i])
        dpi = -(kapf * gp[:, i] - (f - c3) * gg[:, i])
        for jj in range(1, NPI + 1):
            sel = (j == jj) & act
            if bool(sel.any()):
                D[jj - 1] = torch.where(sel, dgi, D[jj - 1])
                P[jj - 1] = torch.where(sel, dpi, P[jj - 1])
    consts = _milne_constants(hx, ccl)
    if kernels_usable(dv):
        from .kernels import milne_inward
        return milne_inward(gg, gp, torch.stack(D, 1), torch.stack(P, 1), dv, vm, dr.contiguous(), ec, kapf, ic3f, iflat0, hx, consts)
    return _milne_inward_plain(gg, gp, D, P, dv, vm, dr, ec, kapf, ic3f, iflat0, hx, consts)


def _solve_irregular(self, en, dv, vmb, kap, ic3, jlast, ie_of):
    """Irregular solution for a batch of lanes of this potential (dfovrg with irr = 1)."""
    kapf, ic3f = kap.to(RDTYPE), ic3.to(RDTYPE)
    jlast = torch.clamp(jlast, max=self.np_)
    iwkb, iflat = _lane_setup(en, self.hx, self.idim, self.jri, jlast)
    vm = vmb[ie_of] * (kapf + 1.0)[:, None] / ALPINV
    return dirac_inward(en / CL_W, en, kap, ic3, dv[ie_of], vm, self.dr, self.hx, iflat - 1, iwkb - 1, self.jri - 1, jlast - 1,
                        getattr(self, "bessel_orders", LTOT + 2))


_Potential.solve_irregular = _solve_irregular
