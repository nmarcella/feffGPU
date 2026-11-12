"""Radial distribution tools: coordination numbers and distances of an ideal lattice, periodic images beyond half the
cell, a planted Gaussian width recovered by integration and by the peak fits, projected RDFs, dynamic cells."""
import numpy as np
import pytest
from mdexafs_testing import A_W, bcc_sites, gaussian_frames, rock_salt

BCC_SHELLS = [(8, np.sqrt(3) / 2), (6, 1.0), (12, np.sqrt(2)), (24, np.sqrt(11) / 2), (8, np.sqrt(3))]   # N and R / a


def _shells(a):
    """Ranges around the first five bcc shells."""
    rs = [f * a for _n, f in BCC_SHELLS]
    edges = [rs[0] - 0.2] + [0.5 * (x + y) for x, y in zip(rs, rs[1:])] + [rs[-1] + 0.2]
    return [(edges[i], edges[i + 1]) for i in range(5)]


@pytest.mark.parametrize("L", [4, 2])
def test_ideal_bcc_coordination_numbers_and_distances(L):
    """With L = 2 the cell (6.3 A) is narrower than twice the largest distance: periodic images are counted."""
    from feffgpu.analysis.rdf import RDFAccumulator
    sites, box = bcc_sites(L)
    sym = np.array(["W"] * len(sites))
    shells = [("W", "W", lo, hi) for lo, hi in _shells(A_W)]
    acc = RDFAccumulator(sym, r_max=5.9, bin_width=0.01, shells=shells, device="cpu")
    acc.add(sites, np.eye(3) * box)
    res = acc.result()
    for (n, f), sh in zip(BCC_SHELLS, res["shells"]):
        assert sh["N"] == pytest.approx(n) and sh["R"] == pytest.approx(f * A_W) and sh["sigma2"] == pytest.approx(0.0, abs=1e-12)
    p = res["pairs"]["W-W"]
    assert p["coordination"][-1] == pytest.approx(58.0)                                     # 8 + 6 + 12 + 24 + 8 below 5.9 A
    assert p["pairs_per_center_per_A"].sum() * res["bin_width"] == pytest.approx(58.0)
    r = res["r"]
    tail = (r > 5.6) & (r < 5.9)
    assert np.all(p["g"][tail] == 0) and p["g"].max() > 10                                  # an ideal lattice: sharp peaks, nothing between


def test_integration_equals_shell_table_and_fits_recover_a_planted_width():
    """Frames of Gaussian displacements: the exact shell numbers equal feffgpu.analysis.shell_table on the same
    frames; the MSRD is 2 s^2 within the sampling error; both fitted forms recover it, with standard errors."""
    from feffgpu.analysis import shell_table
    from feffgpu.analysis.peaks import analyze_peak
    from feffgpu.analysis.rdf import RDFAccumulator
    s = 0.06
    sites, box = bcc_sites(4)
    sym = np.array(["W"] * len(sites))
    frames = gaussian_frames(sites, 12, s, seed=7)
    acc = RDFAccumulator(sym, r_max=4.0, bin_width=0.005, shells=[("W", "W", 2.2, 2.98)], device="cpu")
    for f in frames:
        acc.add(f, np.eye(3) * box)
    res = acc.result()
    exact = res["shells"][0]
    ref = shell_table(np.array(frames), sym, box, [("W", "W", 2.2, 2.98)])[0]
    for key in ("N", "R", "sigma2", "c3"):
        assert exact[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-12), key
    per_frame = [shell_table(f, sym, box, [("W", "W", 2.2, 2.98)])[0]["sigma2"] for f in frames]
    se = np.std(per_frame, ddof=1) / np.sqrt(len(frames))
    assert abs(exact["sigma2"] - 2 * s * s) < 3 * se
    an = analyze_peak(res["r"], res["pairs"]["W-W"]["pairs_per_center_per_A"], 2.2, 2.98, "both", exact=exact)
    assert an["integration"]["source"] == "distances"
    for form in ("gaussian", "skewed"):
        f = an[form]
        assert "error" not in f, f
        assert f["N"] == pytest.approx(8.0, rel=0.01) and f["R"] == pytest.approx(exact["R"], abs=0.003)
        assert f["sigma2"] == pytest.approx(exact["sigma2"], rel=0.05)
        assert 0 < f["sigma2_se"] < 0.1 * f["sigma2"] and f["N_se"] > 0 and f["r_squared"] > 0.98
    hist = analyze_peak(res["r"], res["pairs"]["W-W"]["pairs_per_center_per_A"], 2.2, 2.98, "none")["integration"]
    assert hist["source"] == "histogram" and hist["N"] == pytest.approx(exact["N"], rel=1e-3)
    assert hist["sigma2"] == pytest.approx(exact["sigma2"] + 0.005 ** 2 / 12, rel=0.02)          # binning adds bin_width^2 / 12


def test_skewed_fit_recovers_an_asymmetric_peak():
    from feffgpu.analysis.peaks import fit_peak, skewnormal_density, skewnormal_moments
    r = 0.005 * (np.arange(1200) + 0.5)
    dens = skewnormal_density(r, 6.0, 2.3, 0.12, 3.0)
    f = fit_peak(r, dens, 2.0, 3.2, "skewed")
    mean, var, c3 = skewnormal_moments(2.3, 0.12, 3.0)
    assert f["N"] == pytest.approx(6.0, rel=1e-3) and f["R"] == pytest.approx(mean, abs=1e-4)
    assert f["sigma2"] == pytest.approx(var, rel=1e-3) and f["c3"] == pytest.approx(c3, rel=1e-2)
    g = fit_peak(r, dens, 2.0, 3.2, "gaussian")
    assert g["c3"] == 0.0 and g["r_squared"] < f["r_squared"]


def test_projected_rdfs_count_the_bonds_along_a_direction():
    from feffgpu.analysis.rdf import RDFAccumulator, cubic_directions, solid_angle_fraction
    sites, box = bcc_sites(4)
    sym = np.array(["W"] * len(sites))
    r1, r2 = np.sqrt(3) / 2 * A_W, A_W
    proj = [{"center": "W", "neighbor": "W", "direction": [1, 1, 1], "angle": 10.0, "r_min": 2.5, "r_max": 2.9},
            {"center": "W", "neighbor": "W", "direction": [1, 1, 1], "angle": 10.0, "symmetry": True, "r_min": 2.5, "r_max": 2.9},
            {"center": "W", "neighbor": "W", "direction": [1, 0, 0], "angle": 10.0, "symmetry": True, "r_min": 3.0, "r_max": 3.3}]
    acc = RDFAccumulator(sym, r_max=5.0, bin_width=0.01, projections=proj, device="cpu")
    acc.add(sites, np.eye(3) * box)
    res = acc.result()["projections"]
    assert res[0]["shell"]["N"] == pytest.approx(2.0) and res[0]["shell"]["R"] == pytest.approx(r1)        # +-[111]
    assert res[1]["shell"]["N"] == pytest.approx(8.0)                                                         # all <111>
    assert res[2]["shell"]["N"] == pytest.approx(6.0) and res[2]["shell"]["R"] == pytest.approx(r2)        # all <100>
    assert len(cubic_directions([1, 1, 0])) == 12 and len(cubic_directions([1, 0, 0])) == 6
    cap = 1 - np.cos(np.radians(10.0))                                                       # two caps of half-angle 10 degrees
    assert solid_angle_fraction([[0, 0, 1]], np.cos(np.radians(10.0))) == pytest.approx(cap, rel=1e-3)
    assert res[0]["solid_angle_fraction"] == pytest.approx(cap, rel=1e-3)


def test_dynamic_cells_and_clusters():
    """Two frames with different cells (NPT): each contributes its own first-shell distance; the ideal-gas
    normalization uses each frame's volume. A cluster gives pairs per center but no g(r)."""
    from feffgpu.analysis.rdf import RDFAccumulator
    sites, box = bcc_sites(3)
    sym = np.array(["W"] * len(sites))
    acc = RDFAccumulator(sym, r_max=3.0, bin_width=0.01, shells=[("W", "W", 2.0, 3.0)], device="cpu")
    for scale in (1.0, 1.02):
        acc.add(sites * scale, np.eye(3) * box * scale)
    sh = acc.result()["shells"][0]
    assert sh["N"] == pytest.approx(8.0) and sh["R"] == pytest.approx(np.sqrt(3) / 2 * A_W * 1.01)
    assert sh["sigma2"] == pytest.approx((np.sqrt(3) / 2 * A_W * 0.01) ** 2)
    pos, rsym, rbox = rock_salt(2)
    c = RDFAccumulator(rsym, pairs=[("Zn", "C")], r_max=2.5, bin_width=0.01, shells=[("Zn", "C", 2.0, 2.4)], device="cpu")
    c.add(pos, None)
    out = c.result()
    assert out["pairs"]["Zn-C"]["g"] is None and 0 < out["shells"][0]["N"] < 6               # surface atoms of a cluster have fewer
    with pytest.raises(ValueError, match="periodic, or all clusters"):
        c.add(pos, np.eye(3) * rbox)


def test_bond_angles_accumulate_like_the_shared_function():
    from feffgpu.analysis import angle_histogram, bond_angles
    from feffgpu.analysis.rdf import AngleAccumulator
    sites, box = bcc_sites(3)
    sym = np.array(["W"] * len(sites))
    frames = gaussian_frames(sites, 3, 0.03, seed=1)
    acc = AngleAccumulator(sym, [{"center": "W", "neighbors": ["W", "W"], "ranges": [[2.4, 3.0], [2.4, 3.0]], "bin_width": 2.0}])
    total = 0
    for f in frames:
        acc.add(f, np.eye(3) * box)
        total = total + angle_histogram(bond_angles(f, sym, box, "W", ("W", "W"), ((2.4, 3.0), (2.4, 3.0))), 2.0)[1]
    res = acc.result()[0]
    assert np.array_equal(res["counts"], total) and res["angles"] == len(sites) * 28 * 3
    assert res["per_center"].sum() == pytest.approx(28.0)
