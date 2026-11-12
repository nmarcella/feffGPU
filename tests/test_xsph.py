"""The ported phase-shift stage (feffgpu.xsph) against FEFF's own routines (libfeff8lpotph.so through ctypes)
and against the W set's phase.pad and chi.dat: energy mesh, self-energy potential, phase shifts, dipole matrix
elements, and chi through the exact path stage on FEFF's path list."""
import json, os
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tests/data/W"
LIB = Path(os.environ.get("FEFFGPU_POTPH_LIB", ROOT / "fortran/lib/libfeff8lpotph.so"))
pytestmark = [pytest.mark.oracle, pytest.mark.skipif(not LIB.exists(), reason="needs the FEFF potentials library")]


@pytest.fixture(scope="module")
def setup():
    os.environ["FEFF_LMAX"] = "56"
    from feffgpu.pad import read_pot_pad, read_phase_pad, HART, BOHR
    from feffgpu import xsph as X
    pot = read_pot_pad(D / "pot.pad")
    J = json.load(open(D / "xsph.json"))
    args = dict(gamach=J["gamach"] / HART, xkmax=J["xkmax"] * BOHR, rgrd=float(J["rgrd"]))
    mine, xsnorm = X.xsph(pot, lmax_override=56, potlbl=["W", "W"], **args)
    return pot, read_phase_pad(D / "phase.pad"), J, args, mine, xsnorm


def test_energy_mesh_and_self_energy(setup):
    from feffgpu import xsph as X, potph_lib as F
    pot, ref, J, args, mine, _ = setup
    em, ne, ne1, ik0 = F.phmesh(0, 0, pot.xmu, pot.emu, 0.0, args["gamach"], args["xkmax"], J["xkstep"] * 0.52917721067, 0.0)
    assert (ne, ne1, ik0) == (mine.ne, mine.ne1, mine.ik0) and np.abs(em - mine.em).max() < 1e-13
    P = X._Potential(pot, 0, pot.ihole, 0.05, args["rgrd"], "cpu")
    v, eref, *_ = P.potential(0, 0, torch.tensor(mine.em), pot.xmu)
    for ie in (0, 10, 44, 80):
        eref_l, v_l, _ = F.xcpot(0, ie + 1, 0, 0, P.jri, mine.em[ie], pot.xmu, P.vtotph.numpy(), P.vtotph.numpy(), P.rhoph.numpy(),
                                 np.zeros(1251), P.rhoph.numpy(), pot.rnrm[0])
        assert abs(eref_l - eref[ie].item()) < 1e-13
        assert np.abs(v_l[:P.jri + 1] - v[ie].numpy()).max() / np.abs(v_l[:P.jri + 1]).max() < 1e-14


def test_phase_shifts_and_matrix_elements(setup):
    from feffgpu import xsph as X, potph_lib as F
    pot, ref, J, args, mine, xsnorm = setup
    em = mine.em
    for iph in range(2):
        ri, vt, rh, dm = F.fixvar(pot.rmt[iph], pot.edens[:, iph], pot.vtot[:, iph], pot.dmag[:, iph], pot.vint, pot.rhoint, 0.05,
                                  args["rgrd"], pot.jumprm)
        eref_l, ph_l, lmax_l = F.phase(iph, args["rgrd"], 8.8, ri, em, 0, 1, 0, pot.rmt[iph], pot.rnrm[iph], pot.xmu, vt, vt, rh, dm, rh,
                                       pot.dgc[:, :, 0], pot.dpc[:, :, 0], pot.adgc[:, :, iph], pot.adpc[:, :, iph], int(pot.iz[iph]),
                                       pot.ihole if iph == 0 else 0, float(pot.xion[iph]), pot.iunf, pot.xnval[:, iph])
        assert lmax_l == 56 and np.abs(eref_l - mine.eref[:, 0]).max() < 1e-13
        lo = F.LTOT - 56
        ph_l = np.where(np.isfinite(ph_l), ph_l, 0)[:, lo:lo + 113]
        t_l, t_m = np.exp(2j * ph_l) - 1, np.exp(2j * mine.ph[:, :, 0, iph]) - 1      # the scattering amplitude's phase factor
        # the path stage uses l up to the last non-negligible phase on the positive side (rdxsph's rule)
        used = np.zeros_like(t_l, dtype=bool)
        for ie in range(ph_l.shape[0]):
            nz = np.nonzero(np.abs(np.sin(ph_l[ie, 56:])) > 1e-7)[0]
            lm = int(nz.max()) if len(nz) else 0
            used[ie, 56 - lm:56 + lm + 1] = True
        d = np.abs(t_l - t_m)
        tiny = (np.abs(t_l) < 2e-5) & (np.abs(t_m) < 2e-5)       # FEFF's cut threshold; lanes it computes from rounding noise
        bad = (d > 1e-9) & ~tiny & used
        # Known difference: FEFF's partial-wave limit per energy is an integer part of a product with k. Where that product
        # lands on an integer (here k = 32 1/A of the W set) FEFF and the port can round to different sides, so FEFF keeps
        # one more partial wave than the port at that energy. Such an entry is the outermost used l of its energy, is zero
        # in the port and small in FEFF. Anything else is a failure.
        for ie, il in np.argwhere(bad):
            outer = abs(il - 56) == np.abs(np.nonzero(used[ie])[0] - 56).max()
            assert outer and t_m[ie, il] == 0 and abs(t_l[ie, il]) < 2e-3, (iph, ie, il - 56, abs(t_l[ie, il]), abs(t_m[ie, il]))
        assert bad.sum() <= 2, int(bad.sum())
        assert np.median(d[used]) < 1e-13
    ri, vt, rh, dm = F.fixvar(pot.rmt[0], pot.edens[:, 0], pot.vtot[:, 0], pot.dmag[:, 0], pot.vint, pot.rhoint, 0.05, args["rgrd"], pot.jumprm)
    dgcx, dpcx, jnew = X.fixdsp(0.05, args["rgrd"], pot.dgc0, pot.dpc0)
    xsec_l, xsn_l, rkk_l = F.xsect(args["rgrd"], 8.8, ri, em, mine.ne1, mine.ik0, pot.xmu, pot.ihole, pot.emu, dgcx, dpcx, jnew, 0, 0,
                                   pot.rmt[0], pot.rnrm[0], pot.xmu, vt, vt, rh, dm, rh, pot.dgc[:, :, 0], pot.dpc[:, :, 0],
                                   pot.adgc[:, :, 0], pot.adpc[:, :, 0], int(pot.iz[0]), float(pot.xion[0]), pot.iunf, pot.xnval[:, 0],
                                   pot.iorb[:, 0])
    assert (np.abs(rkk_l[:, :2] - mine.rkk[:, :2, 0]) / np.abs(rkk_l[:, :2])).max() < 1e-9
    assert (np.abs(xsn_l - xsnorm) / xsn_l).max() < 1e-9
    assert np.abs(ref.rkk[:, :2, 0] - mine.rkk[:, :2, 0]).max() < 1e-9           # the packed file itself


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_chi_through_path_stage(setup):
    """FEFF's 96-path list of the W set: chi from the ported stage equals chi from phase.pad."""
    from feffgpu.stage import GPUPathStage
    from feffgpu.compare import r_factor
    pot, ref, J, args, mine, _ = setup
    lines = (D / "paths.dat").read_text().splitlines()
    i, paths = 2, []
    while i < len(lines):
        if "index, nleg" not in lines[i]:
            i += 1
            continue
        w = lines[i].split()
        nleg, deg = int(w[1]), float(w[2])
        rows = [lines[i + 2 + j].split() for j in range(nleg)]
        pts = np.array([[float(v) for v in r[:3]] for r in rows])
        paths.append((nleg, deg, np.concatenate([np.zeros((1, 3)), pts[:-1]]), [0] + [int(r[3]) for r in rows][:-1]))
        i += 2 + nleg
    rows = [l.split() for l in (D / "chi.dat").read_text().splitlines() if l.strip() and not l.startswith("#")]
    tab = np.array([[float(r[0]), float(r[1])] for r in rows if len(r) >= 2 and r[0].replace(".", "", 1).isdigit()])[:601]
    kf, chi_ref = tab[:, 0], tab[:, 1]

    def chi_of(P):
        g = GPUPathStage(P, device="cuda", dtype=torch.complex128)
        S = g.linear_matrix(kf)
        acc = np.zeros(len(kf))
        with torch.no_grad():
            for nleg in (2, 3, 4):
                grp = [p for p in paths if p[0] == nleg]
                coords = torch.tensor(np.stack([p[2] for p in grp]), dtype=torch.float64, device="cuda")
                ipots = torch.tensor(np.array([p[3] for p in grp]), device="cuda")
                deg = torch.tensor([p[1] for p in grp], dtype=torch.float64, device="cuda")
                cc, rf = g.cchi(coords, ipots)
                acc += (deg[:, None] * g.chi_fine(cc, rf, kf, S, "feff")).sum(0).cpu().numpy()
        return acc

    k0 = np.concatenate([[0.0], kf])
    R = lambda a, b: r_factor(k0, np.concatenate([[0.0], a]), np.concatenate([[0.0], b]), kmin=3.0, kmax=23.0, rmin=1.0, rmax=9.3)
    a, b = chi_of(ref), chi_of(mine)
    assert R(chi_ref, a) < 1e-9 and R(chi_ref, b) < 1e-9 and R(a, b) < 1e-14


def test_matrix_elements_use_ixc0_not_ixc(setup):
    """FEFF computes the dipole matrix elements with the EXCHANGE card's fourth value (ixc0, default 0) whatever the
    exchange model of the phase shifts: with ground-state exchange (ixc 2) rkk is unchanged, the phases are not."""
    from feffgpu import xsph as X
    pot, ref, J, args, mine, xsnorm = setup
    gs, _ = X.xsph(pot, lmax_override=10, potlbl=["W", "W"], ixc=2, ixc0=0, **args)
    assert np.abs(gs.rkk[:gs.ne1, :2, 0] - mine.rkk[:gs.ne1, :2, 0]).max() < 1e-12
    assert np.abs(gs.ph[:gs.ne1, :, 0, 0] - mine.ph[:gs.ne1, mine.ltot - 10:mine.ltot + 11, 0, 0]).max() > 1e-3
