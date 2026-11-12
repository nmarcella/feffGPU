"""ctypes access to routines of FEFF's potentials-and-phases library (libfeff8lpotph.so, built with the path
stage's rebuilt FEFF) for use as full-precision oracles in tests: fixvar, phmesh, xcpot, dfovrg, phase, xsect.
Arrays are passed in Fortran (column-major) order with FEFF's fixed dimensions (nrptx 1251, nex 150, ltot 80)."""
from __future__ import annotations
import ctypes as C
import os
import numpy as np

NRPTX, NEX, LTOT, NPHX, MXPOLE = 1251, 150, 80, 11, 1000
_LIB = None


def find_library(name="libfeff8lpotph.so"):
    """The rebuilt FEFF library: $FEFFGPU_POTPH_LIB (a file), $FEFF_HOME/lib/<name>, or fortran/lib/<name> next to a source
    checkout of this repository. Needed only for ION / UNFREEZEF inputs, for rebuilding the free-atom table and for the
    oracle tests; see fortran/README.md."""
    cands = [os.environ.get("FEFFGPU_POTPH_LIB", ""), os.path.join(os.environ.get("FEFF_HOME", ""), "lib", name),
             os.path.join(os.path.dirname(__file__), "..", "..", "fortran", "lib", name)]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(f"{name} not found: build it from feff85exafs with fortran/feff85exafs_ltot80.patch (fortran/README.md) "
                            f"and point FEFFGPU_POTPH_LIB or FEFF_HOME at it")


def lib(path=None):
    global _LIB
    if _LIB is None:
        if path is None:
            path = find_library("libfeff8lpotph.so")
        _LIB = C.CDLL(os.path.abspath(path))
    return _LIB


def _i(v):
    return C.byref(C.c_int(int(v)))


def _d(v):
    return C.byref(C.c_double(float(v)))


def _arr(a, dtype, shape=None):
    """A Fortran-ordered array of the given dtype (copy of a, zero-padded to shape) and its pointer."""
    out = np.zeros(shape if shape is not None else np.shape(a), dtype=dtype, order="F")
    if a is not None:
        a = np.asarray(a)
        sl = tuple(slice(0, n) for n in a.shape)
        out[sl] = a
    return out, out.ctypes.data_as(C.c_void_p)


def fixvar(rmt, edens, vtot, dmag, vint, rhoint, dxorg, dxnew, jumprm):
    L = lib()
    ri, pri = _arr(None, np.float64, (NRPTX,))
    vtotph, pv = _arr(None, np.float64, (NRPTX,))
    rhoph, pr = _arr(None, np.float64, (NRPTX,))
    dmagx, pm = _arr(None, np.float64, (NRPTX,))
    e, pe = _arr(edens[:251], np.float64, (251,))
    v, pvt = _arr(vtot[:251], np.float64, (251,))
    m, pdm = _arr(dmag[:251], np.float64, (251,))
    vjump = C.c_double(0.0)
    L.fixvar_(_d(rmt), pe, pvt, pdm, _d(vint), _d(rhoint), _d(dxorg), _d(dxnew), _i(jumprm), C.byref(vjump), pri, pv, pr, pm)
    return ri, vtotph, rhoph, dmagx


def phmesh(iprint, ispec, edge, emu, vi0, gamach, xkmax, xkstep, vixan):
    L = lib()
    em, pem = _arr(None, np.complex128, (NEX,))
    ne, ne1, ik0, ne3 = C.c_int(0), C.c_int(0), C.c_int(0), C.c_int(0)
    L.phmesh_(_i(iprint), _i(ispec), _d(edge), _d(emu), _d(vi0), _d(gamach), _d(xkmax), _d(xkstep), _d(vixan),
              C.byref(ne), C.byref(ne1), pem, C.byref(ik0), C.byref(ne3))
    return em[:ne.value], ne.value, ne1.value, ik0.value


def phase(iph, dx, x0, ri, em, ixc, nsp, lreal, rmt, rnrm, xmu, vtotph, vvalph, rhoph, dmagx, rhphvl, dgcn, dpcn, adgc, adpc,
          iz, ihole, xion, iunf, xnval, ispin=0, iPl=0, ne3=0):
    """FEFF's phase.f for one potential: eref (ne,) and ph (ne, 2 ltot + 1) at full double precision, lmax."""
    L = lib()
    ne = len(em)
    emf, pem = _arr(em, np.complex128, (NEX,))
    rif, pri = _arr(ri, np.float64, (NRPTX,))
    arrs = [_arr(a, np.float64, (NRPTX,)) for a in (vtotph, vvalph, rhoph, dmagx, rhphvl)]
    dg, pdg = _arr(dgcn, np.float64, (NRPTX, 30))
    dp, pdp = _arr(dpcn, np.float64, (NRPTX, 30))
    ag, pag = _arr(adgc, np.float64, (10, 30))
    ap, pap = _arr(adpc, np.float64, (10, 30))
    eref, peref = _arr(None, np.complex128, (NEX,))
    ph, pph = _arr(None, np.complex128, (NEX, 2 * LTOT + 1))
    xnv, pxn = _arr(xnval, np.float64, (30,))
    lmax = C.c_int(0)
    L.phase_(_i(iph), _d(dx), _d(x0), pri, _i(ne), _i(len(em) - ne3), _i(ne3), pem, _i(ixc), _i(nsp), _i(lreal), _d(rmt), _d(rnrm),
             _d(xmu), _i(iPl), arrs[0][1], arrs[1][1], arrs[2][1], arrs[3][1], arrs[4][1], pdg, pdp, pag, pap, peref, pph,
             C.byref(lmax), _i(iz), _i(ihole), _d(xion), _i(iunf), pxn, _i(ispin))
    return eref[:ne].copy(), ph[:ne].copy(), lmax.value


def xsect(dx, x0, ri, em, ne1, ik0, edge, ihole, emu, dgcx, dpcx, jnew, ixc, lreal, rmt, rnrm, xmu, vtotph, vvalph, rhoph, dmagx,
          rhphvl, dgcn, dpcn, adgc, adpc, iz, xion, iunf, xnval, iorb, ipr2=0, iPl=0, izstd=0, l2lp=0, ipol=0, ispin=0, le2=0,
          angks=0.0, corr=1.0):
    """FEFF's xsect.f: xsec (ne,), xsnorm (ne,), rkk (ne, 8) at full precision."""
    L = lib()
    ne = len(em)
    emf, pem = _arr(em, np.complex128, (NEX,))
    rif, pri = _arr(ri, np.float64, (NRPTX,))
    g0, pg0 = _arr(dgcx, np.float64, (NRPTX,))
    p0, pp0 = _arr(dpcx, np.float64, (NRPTX,))
    arrs = [_arr(a, np.float64, (NRPTX,)) for a in (vtotph, vvalph, rhoph, dmagx, rhphvl)]
    dg, pdg = _arr(dgcn, np.float64, (NRPTX, 30))
    dp, pdp = _arr(dpcn, np.float64, (NRPTX, 30))
    ag, pag = _arr(adgc, np.float64, (10, 30))
    ap, pap = _arr(adpc, np.float64, (10, 30))
    xsec, pxs = _arr(None, np.complex128, (NEX,))
    xsnorm, pxn = _arr(None, np.float64, (NEX,))
    rkk, prk = _arr(None, np.complex128, (NEX, 8))
    xnv, pxv = _arr(xnval, np.float64, (30,))
    io, pio = _arr(iorb, np.int32, (8,))
    ptz = np.zeros((3, 3), dtype=np.complex128, order="F")
    for k in range(3):
        ptz[k, k] = 1.0 / 3.0
    L.xsect_(_i(ipr2), _d(dx), _d(x0), pri, _i(ne), _i(ne1), _i(ik0), pem, _d(edge), _i(ihole), _d(emu), _d(corr), pg0, pp0,
             _i(jnew), _i(ixc), _i(lreal), _d(rmt), _d(rnrm), _d(xmu), _i(iPl), arrs[0][1], arrs[1][1], arrs[2][1], arrs[3][1],
             arrs[4][1], pdg, pdp, pag, pap, pxs, pxn, prk, _i(iz), _d(xion), _i(iunf), pxv, _i(izstd), pio, _i(l2lp), _i(ipol),
             _i(ispin), _i(le2), _d(angks), ptz.ctypes.data_as(C.c_void_p))
    return xsec[:ne].copy(), xsnorm[:ne].copy(), rkk[:ne].copy()


def xcpot(iph, ie, index, lreal, jri, em, xmu, vtot, vvalgs, densty, dmag, denval, rnrm):
    """FEFF's xcpot.f for one energy (ifirst = 0): eref, v (nrptx,), vval."""
    L = lib()
    arrs = [_arr(a, np.float64, (NRPTX,)) for a in (vtot, vvalgs, densty, dmag, denval)]
    eref = np.zeros(1, dtype=np.complex128)
    v, pv = _arr(None, np.complex128, (NRPTX,))
    vval, pvv = _arr(None, np.complex128, (NRPTX,))
    wp, pwp = _arr(np.full(MXPOLE, -1.0e30), np.float64, (MXPOLE,))
    amp, pamp = _arr(None, np.float64, (MXPOLE,))
    work = [_arr(None, np.float64, (NRPTX,)) for _ in range(5)]
    emc = np.array([em], dtype=np.complex128)
    L.xcpot_(_i(iph), _i(ie), _i(index), _i(lreal), _i(0), _i(jri), emc.ctypes.data_as(C.c_void_p), _d(xmu),
             arrs[0][1], arrs[1][1], arrs[2][1], arrs[3][1], arrs[4][1], eref.ctypes.data_as(C.c_void_p), pv, pvv, _i(0), pwp, pamp,
             work[0][1], work[1][1], work[2][1], work[3][1], work[4][1], _d(rnrm))
    return eref[0], v, vval


def dfovrg(ncycle, ikap, rmt, jlast, jri, p2, dx, ri, vxc, vxcval, dgcn, dpcn, adgc, adpc, xnval, iz, ihole, xion, iunf, irr, ic3,
           pu=0.0, qu=0.0, ps=None, qs=None):
    """FEFF's Dirac solver for one lane: pu, qu at rmt and ps, qs (nrptx,)."""
    L = lib()
    rif, pri = _arr(ri, np.float64, (NRPTX,))
    v, pv = _arr(vxc, np.complex128, (NRPTX,))
    vv, pvv = _arr(vxcval, np.complex128, (NRPTX,))
    dg, pdg = _arr(dgcn, np.float64, (NRPTX, 30))
    dp, pdp = _arr(dpcn, np.float64, (NRPTX, 30))
    ag, pag = _arr(adgc, np.float64, (10, 30))
    ap, pap = _arr(adpc, np.float64, (10, 30))
    xnv, pxn = _arr(xnval, np.float64, (30,))
    puc, quc = np.array([pu], dtype=np.complex128), np.array([qu], dtype=np.complex128)
    p, pp = _arr(ps, np.complex128, (NRPTX,))
    q, pq = _arr(qs, np.complex128, (NRPTX,))
    p2c = np.array([p2], dtype=np.complex128)
    jl = C.c_int(int(jlast))
    L.dfovrg_(_i(ncycle), _i(ikap), _d(rmt), C.byref(jl), _i(jri), p2c.ctypes.data_as(C.c_void_p), _d(dx), pri, pv, pvv, pdg, pdp,
              pag, pap, pxn, puc.ctypes.data_as(C.c_void_p), quc.ctypes.data_as(C.c_void_p), pp, pq, _i(iz), _i(ihole), _d(xion),
              _i(iunf), _i(irr), _i(ic3))
    return puc[0], quc[0], p, q, jl.value


def scfdat(iz, ihole, xion, iunf, ispinr=0):
    """FEFF's single-configuration Dirac-Fock atom (ATOM/scfdat.f) for one species: Coulomb potential, densities
    (times 4 pi), orbitals and their origin expansions, total energy, etc. ispinr = the hole index saves that
    orbital as the core-hole spinor (dgc0, dpc0)."""
    L = lib()
    vcoul, pvc = _arr(None, np.float64, (251,))
    srho, psr = _arr(None, np.float64, (251,))
    dmag, pdm = _arr(None, np.float64, (251,))
    srhovl, psv = _arr(None, np.float64, (251,))
    dgc0, pg0 = _arr(None, np.float64, (251,))
    dpc0, pp0 = _arr(None, np.float64, (251,))
    dgc, pdg = _arr(None, np.float64, (251, 30, NPHX + 1))
    dpc, pdp = _arr(None, np.float64, (251, 30, NPHX + 1))
    adgc, pag = _arr(None, np.float64, (10, 30, NPHX + 1))
    adpc, pap = _arr(None, np.float64, (10, 30, NPHX + 1))
    s02, efrozn, eatom = C.c_double(0), C.c_double(0), C.c_double(0)
    xntot, pxt = _arr(None, np.float64, (5,))
    xnval, pxv = _arr(None, np.float64, (30,))
    indorb, pio = _arr(None, np.int32, (8,))
    norbp = C.c_int(0)
    eorb, peo = _arr(None, np.float64, (30,))
    kappa, pka = _arr(None, np.int32, (30,))
    L.scfdat_(_i(0), _i(0), _i(0), _i(iz), _i(ihole), _d(xion), _i(iunf), pvc, psr, pdm, psv, _i(ispinr), pg0, pp0, pdg, pdp,
              pag, pap, C.byref(s02), C.byref(efrozn), C.byref(eatom), pxt, pxv, pio, C.byref(norbp), peo, pka)
    return dict(vcoul=vcoul.copy(), rho=srho.copy(), dmag=dmag.copy(), rhoval=srhovl.copy(), dgc0=dgc0.copy(), dpc0=dpc0.copy(),
                dgc=dgc[:, :, 0].copy(), dpc=dpc[:, :, 0].copy(), adgc=adgc[:, :, 0].copy(), adpc=adpc[:, :, 0].copy(),
                s02=s02.value, efrozn=efrozn.value, eatom=eatom.value, xnvmu=xntot.copy(), xnval=xnval.copy(),
                iorb=indorb.copy(), norb=norbp.value, eorb=eorb.copy(), kappa=kappa.copy())


def rholie(ri05, dx, x0, ri, em, ixc, rmt, rnrm, vtot, vvalgs, xnval, dgcn, dpcn, eref, adgc, adpc, iz, xion, iunf, ihole, lmaxsc):
    """FEFF's rholie.f at one complex energy: xrhole (5,), xrhoce (5,), yrhole (251, 5), yrhoce (251,), ph (5,), nr05."""
    L = lib()
    r05, pr05 = _arr(ri05, np.float64, (251,))
    rif, pri = _arr(ri, np.float64, (NRPTX,))
    vt, pvt = _arr(vtot, np.float64, (NRPTX,))
    vv, pvv = _arr(vvalgs, np.float64, (NRPTX,))
    xnv, pxn = _arr(xnval, np.float64, (30,))
    dg, pdg = _arr(dgcn, np.float64, (NRPTX, 30))
    dp, pdp = _arr(dpcn, np.float64, (NRPTX, 30))
    ag, pag = _arr(adgc, np.float64, (10, 30))
    ap, pap = _arr(adpc, np.float64, (10, 30))
    xrhole, p1 = _arr(None, np.complex128, (5,))
    xrhoce, p2 = _arr(None, np.complex128, (5,))
    yrhole, p3 = _arr(None, np.complex128, (251, 5))
    yrhoce, p4 = _arr(None, np.complex128, (251,))
    ph, p5 = _arr(None, np.complex128, (5,))
    emc = np.array([em], dtype=np.complex128)
    erc = np.array([eref], dtype=np.complex128)
    nr05 = C.c_int(0)
    L.rholie_(pr05, C.byref(nr05), _d(dx), _d(x0), pri, emc.ctypes.data_as(C.c_void_p), _i(ixc), _d(rmt), _d(rnrm), pvt, pvv, pxn,
              pdg, pdp, erc.ctypes.data_as(C.c_void_p), pag, pap, p1, p2, p3, p4, p5, _i(iz), _d(xion), _i(iunf), _i(ihole), _i(lmaxsc))
    return xrhole.copy(), xrhoce.copy(), yrhole.copy(), yrhoce.copy(), ph.copy(), nr05.value


def fmsie(iph0, nph, lipotx, ie, em, eref, ph, rfms, lfms, iphat, rath):
    """FEFF's fmsie.f (single-precision FMS): gtr (5, nph+1) complex64 for one energy. ph (5, nph+1) complex phase
    shifts, rath (nat, 3) bohr."""
    L = lib()
    nat = len(rath)
    lip, plip = _arr(lipotx, np.int32, (NPHX + 1,))
    phf, pph = _arr(ph, np.complex128, (5, NPHX + 1))
    iph, piph = _arr(iphat, np.int32, (1000,))
    rt, prt = _arr(np.asarray(rath).T, np.float64, (3, 1000))
    gtr, pg = _arr(None, np.complex64, (5, NPHX + 1))
    emc = np.array([em], dtype=np.complex128)
    erc = np.array([eref], dtype=np.complex128)
    verbse = C.c_int(0)
    L.fmsie_(C.byref(verbse), _i(iph0), _i(nph), plip, _i(ie), emc.ctypes.data_as(C.c_void_p), erc.ctypes.data_as(C.c_void_p), pph,
             C.byref(C.c_float(rfms)), _i(lfms), _i(nat), piph, prt, pg)
    return gtr.copy()
