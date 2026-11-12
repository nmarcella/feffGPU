"""The ported potentials stage (feffgpu.pot, scf, fms) against FEFF: free-atom data through
FEFF's atom code equal the orbitals stored in pot.pad; the overlapped potentials, radii and interstitial of the W
set without self-consistency reproduce FEFF's pot.pad made with the SCF card removed; the first self-consistency
iteration reproduces the electron counts, Fermi level and charge transfer FEFF logged for that iteration; the
embedded-atom density of states and the cluster multiple scattering match FEFF's routines."""
import json, math, os
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tests/data/W"
LIB = Path(os.environ.get("FEFFGPU_POTPH_LIB", ROOT / "fortran/lib/libfeff8lpotph.so"))
pytestmark = [pytest.mark.oracle, pytest.mark.skipif(not LIB.exists(), reason="needs the FEFF potentials library")]


@pytest.fixture(scope="module")
def inputs():
    os.environ["FEFF_LMAX"] = "56"
    from feffgpu.pad import read_pot_pad, BOHR, HART
    g = json.load(open(D / "geom.json"))
    pj = json.load(open(D / "pot.json"))
    nat, nph = g["natt"], g["nph"]
    rat = np.array([g["x"], g["y"], g["z"]]).T[:nat] / BOHR
    iphat = np.array(g["iph"][:nat])
    iatph = [int(v) - 1 for v in g["iatph"][:nph + 1]]
    kw = dict(iz=list(pj["iz"]), ihole=pj["ihole"], xion=list(pj["xion"]), iunf=pj["iunf"], nph=nph, nat=nat, iphat=iphat, rat=rat,
              iatph=iatph, xnatph=list(pj["xnatph"]), folp0=list(pj["folp"]), iafolp=pj["iafolp"], ixc=pj["ixc"], nohole=pj["nohole"],
              inters=pj["inters"], totvol=pj["totvol"], ecv0=pj["ecv"] / HART)
    return kw, pj, read_pot_pad(D / "pot.pad"), read_pot_pad(D / "pot_noscf.pad")


def test_free_atoms(inputs):
    from feffgpu import pot as P
    kw, pj, ref, _ = inputs
    assert P._from_table(74, 1, 0) is not None and P._from_table(74, 0, 1) is not None       # the stored table covers W
    A = P.atomic_data(74, 1, 0.0, 0)
    assert np.abs(A["dgc"] - ref.dgc[:, :, 0]).max() < 1e-12
    assert (np.abs(A["adgc"] - ref.adgc[:, :, 0]) / np.abs(ref.adgc[:, :, 0]).clip(1e-30)).max() < 1e-12
    assert np.array_equal(A["kappa"][:22], ref.kappa[:22]) and np.abs(A["xnval"] - ref.xnval[:, 0]).max() == 0
    B = P.atomic_data(74, 0, 0.0, 0, ispinr=1)
    assert np.abs(B["dgc0"] - ref.dgc0).max() < 1e-12


def test_overlap_and_interstitial(inputs):
    from feffgpu import pot as P
    kw, pj, _, ref = inputs
    mine = P.potentials(**kw)
    for k, tol in (("xmu", 1e-6), ("vint", 1e-6), ("rhoint", 1e-6), ("emu", 1e-9), ("erelax", 1e-9), ("wp", 1e-6), ("rnrmav", 1e-12)):
        assert abs(getattr(mine, k) - getattr(ref, k)) < tol, k
    assert np.array_equal(mine.imt, ref.imt) and np.array_equal(mine.inrm, ref.inrm)
    assert np.abs(mine.rmt - ref.rmt).max() < 1e-12 and np.abs(mine.rnrm - ref.rnrm).max() < 1e-12 and np.abs(mine.folpx - ref.folpx).max() < 1e-12
    for k in ("vclap", "edens", "edenvl", "dmag"):
        a, b = getattr(mine, k), getattr(ref, k)
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-12, k
    assert np.abs(mine.vtot - ref.vtot).max() / np.abs(ref.vtot).max() < 1e-10         # FEFF's single-precision interstitial solve


def test_density_of_states_and_fms(inputs):
    from feffgpu.pad import BOHR
    from feffgpu import xsph as X, scf as SC, fms as FM, potph_lib as F
    kw, pj, pot, _ = inputs
    rgrd = float(pj["rgrd"])
    ri05 = SC.RI05
    emg, step = SC.grids(pot.ecv, pot.xmu)
    P = X._Potential(pot, 0, pot.ihole, 0.05, rgrd, "cpu")
    jri1 = int((math.log(pot.rmt[0]) + 8.8) / rgrd) + 3
    vt = P.vtotph.numpy().copy()
    eref = vt[jri1 - 1]
    vt[:jri1] -= eref
    ems = np.array([emg[0], emg[len(emg) // 2], emg[-1]])
    xl, xc, yl, yc, ph, nr05 = SC.rholie(P, torch.tensor(ems, dtype=torch.complex128), torch.full((3,), eref, dtype=torch.complex128), vt[:jri1], 2, ri05)
    for k, em in enumerate(ems):
        xl_l, xc_l, yl_l, yc_l, ph_l, n_l = F.rholie(ri05, rgrd, 8.8, P.ri.numpy(), em, 0, pot.rmt[0], pot.rnrm[0], vt, vt, pot.xnval[:, 0],
                                                    pot.dgc[:, :, 0], pot.dpc[:, :, 0], eref, pot.adgc[:, :, 0], pot.adpc[:, :, 0], 74, 0.0, 0, 1, 2)
        assert n_l == nr05
        for a, b in ((xl[k].numpy()[:3], xl_l[:3]), (xc[k].numpy()[:3], xc_l[:3]), (ph[k].numpy()[:3], ph_l[:3]),
                     (yl[k].numpy()[:nr05, :3], yl_l[:nr05, :3]), (yc[k].numpy()[:nr05], yc_l[:nr05])):
            assert np.abs(a - b).max() / np.abs(b).max() < 1e-10
    phs = np.zeros((5, 2), complex)
    phs[:3, 0] = [0.8 - 0.1j, -0.4 + 0.05j, 1.2 - 0.02j]
    phs[:3, 1] = phs[:3, 0] * 1.1
    em, er = 0.2 + 0.05j, -0.776
    for iph0 in (0, 1):
        cl = FM.Cluster(iph0, kw["rat"], kw["iphat"], pj["rfms1"] / BOHR)
        assert cl.n == 15
        gtr = FM.fms(cl, torch.tensor([np.sqrt(2 * (em - er))], dtype=torch.complex128), [2, 2], torch.tensor(phs[None], dtype=torch.complex128), iph0)
        gl = F.fmsie(iph0, 1, [2, 2], 1, em, er, phs, pj["rfms1"] / BOHR, 0, kw["iphat"], kw["rat"])
        assert np.abs(gtr[0, :3].numpy() - gl[:3, iph0]).max() / np.abs(gl[:3, iph0]).max() < 1e-5


def test_first_scf_iteration_matches_feff_log(inputs):
    """FEFF's log of the W run, iteration 1: counts 0.726 0.954 5.299 / 0.711 0.934 4.355, mu_new -8.180 eV,
    charge transfer 0.009 / -0.000."""
    from feffgpu.pad import BOHR, HART
    from feffgpu import pot as P
    kw, pj, ref, _ = inputs
    mine = P.potentials(**kw, nscmt=1, rfms1=pj["rfms1"] / BOHR, lfms1=pj["lfms1"], ca1=pj["ca1"], lmaxsc=pj["lmaxsc"], rgrd=float(pj["rgrd"]))
    assert abs(mine.xmu * HART - (-8.180)) < 1.5e-3
    counts = mine.xnmues[:3, :].T.ravel()
    assert np.abs(counts - np.array([0.726, 0.954, 5.299, 0.711, 0.934, 4.355])).max() < 1.5e-3
    assert abs(mine.qnrm[0] - 0.009) < 1.5e-3 and abs(mine.qnrm[1]) < 1.5e-3
