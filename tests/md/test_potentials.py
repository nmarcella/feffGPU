"""Potentials of an edge: the reference cluster (with periodic images), its FEFF input, the potential index of every
element, phase files given by the user, and one small reference calculation (slow; needs the free-atom table)."""
import pickle
import numpy as np
import pytest
from mdexafs_testing import A_W, PHASE_W, PHASE_ZNC, bcc_sites, gpu, rock_salt, write_config, write_extxyz


def test_cluster_with_periodic_images_and_its_feff_input(tmp_path):
    from feffgpu.inp import read_feff_inp
    from feffgpu.md.potentials import cut_cluster, reference_input
    sites, box = bcc_sites(2)                                                         # 6.3 A cell, radius 4.6 A: images needed
    rel, sym, d = cut_cluster(sites, np.array(["W"] * len(sites)), np.eye(3) * box, 0, 4.6)
    assert len(rel) == 1 + 8 + 6 + 12 and d[0] == 0 and np.all(np.diff(d) >= -1e-12)
    assert np.allclose(sorted(set(np.round(d[1:], 4))), np.round([np.sqrt(3) / 2 * A_W, A_W, np.sqrt(2) * A_W], 4))
    pos, rsym, rbox = rock_salt(2)
    center = int(np.nonzero(rsym == "C")[0][0])
    rel2, sym2, _ = cut_cluster(pos, rsym, np.eye(3) * rbox, center, 4.0)
    text, ipot_of = reference_input(rel2, sym2, "C", edge="K", scf_radius=3.0, phase_kmax=18.0, rpath=5.0, cards=["EXCHANGE 0 0.0 0.0"])
    assert ipot_of == {"C": 1, "Zn": 2}                                               # alphabetical, as md_exafs 2.0
    (tmp_path / "feff.inp").write_text(text)
    P = read_feff_inp(tmp_path / "feff.inp")
    assert list(P["iz"]) == [6, 6, 30] and P["nat"] == len(rel2) and P["ihole"] == 1 and P["nscmt"] == 30
    assert abs(P["xkmax"] - 18.0) < 1e-9


def test_potential_indices_of_user_phase_files(tmp_path):
    from feffgpu.md.potentials import load_phase, needed_elements, potentials_of_phase
    znc = load_phase(PHASE_ZNC)
    assert potentials_of_phase(znc, "C", ["C", "Zn"]) == {"C": 2, "Zn": 1}
    with pytest.raises(ValueError, match="not Zn"):
        potentials_of_phase(znc, "Zn", ["C"])
    with pytest.raises(ValueError, match="no scatterer potential for O"):
        potentials_of_phase(znc, "C", ["C", "O"])
    with pytest.raises(ValueError, match="potential 1 of the phase file is Z = 30"):
        potentials_of_phase(znc, "C", ["C"], override={"C": 1})
    with open(tmp_path / "phase.pkl", "wb") as f:                                     # the form written by `feffgpu phase`
        pickle.dump(load_phase(PHASE_W), f)
    w = load_phase(tmp_path / "phase.pkl")
    assert list(w.iz) == [74, 74] and potentials_of_phase(w, "W", ["W"]) == {"W": 1}
    assert needed_elements(np.array(["W", "Mo", "Mo"]), "W") == ["Mo"]               # a single absorbing atom needs no scatterer potential
    assert needed_elements(np.array(["W", "W", "Mo"]), "W") == ["Mo", "W"]
    with pytest.raises(FileNotFoundError):
        load_phase(tmp_path / "none.pad")


def test_file_mode_needs_no_calculation(tmp_path):
    from feffgpu.io.trajectory import TrajectoryReader
    from feffgpu.md.config import parse_config
    from feffgpu.md.potentials import PhaseProvider
    pos, sym, box = rock_salt(2)
    write_extxyz(tmp_path / "t.extxyz", sym, [pos], np.eye(3) * box)
    cfg = parse_config({"system": {"trajectory": str(tmp_path / "t.extxyz")}, "lattice": {"dynamic": True},
                        "edge": [{"element": "C", "potentials": "file", "phase": str(PHASE_ZNC)}]})
    prov = PhaseProvider(cfg.edge[0], tmp_path / "out", [0], sym, TrajectoryReader(tmp_path / "t.extxyz", dynamic=True))
    phase, ipot_of, key = prov.phase_for(0)
    assert key == "file" and ipot_of == {"C": 2, "Zn": 1} and prov.source_frame(0) is None
    per = parse_config({"system": {"trajectory": "t"}, "lattice": {"dynamic": True},
                        "edge": [{"element": "C", "potentials": "per_frame", "potentials_every": 3}]}).edge[0]
    p2 = PhaseProvider(per, tmp_path / "out", [0, 2, 4, 6, 8, 10, 12], sym, None)
    assert [p2.key(f) for f in (0, 4, 6, 12)] == ["frame_0", "frame_0", "frame_6", "frame_12"]


def _atoms_table_complete():
    try:
        from feffgpu.atoms_table import table_status
        return bool(table_status()["complete"])
    except Exception:
        return False


@gpu
@pytest.mark.slow
@pytest.mark.skipif(not _atoms_table_complete(), reason="needs the free-atom table (feffgpu fetch-atoms; FEFFGPU_ATOMS)")
def test_one_small_reference_calculation(tmp_path):
    """Potentials and phase shifts of a 15-atom bcc W cluster (no self-consistency), kept in the output folder and
    reused; a frame's spectrum with them is finite and dominated by the first shell."""
    from feffgpu.exafs import k_grid
    from feffgpu.md import load_config
    from feffgpu.md.run import make_reader
    from feffgpu.md.potentials import PhaseProvider
    from feffgpu.md.spectrum import EdgeSpectra
    sites, box = bcc_sites(3)
    sym = np.array(["W"] * len(sites))
    write_extxyz(tmp_path / "t.extxyz", sym, [sites], np.eye(3) * box)
    p = write_config(tmp_path / "c.toml", {"system": {"trajectory": "t.extxyz"}, "lattice": {"dynamic": True},
                                           "edge": [{"element": "W", "potentials": "reference", "cluster_radius": 3.3, "kmax": 12.0,
                                                     "rpath": 4.0, "nleg": 2}]})
    cfg = load_config(p)
    lines = []
    prov = PhaseProvider(cfg.edge[0], cfg.output_path, [0], sym, make_reader(cfg), device="cuda", log=lines.append)
    phase, ipot_of, key = prov.phase_for(0)
    info = prov.info(key)
    assert key == "reference" and ipot_of == {"W": 1} and info["cluster_atoms"] == 15 and list(phase.iz) == [74, 74]
    assert (cfg.output_path / "potentials/W/reference.pkl").is_file() and (cfg.output_path / "potentials/W/reference.feff.inp").is_file()
    again = PhaseProvider(cfg.edge[0], cfg.output_path, [0], sym, make_reader(cfg), device="cuda", log=lines.append)
    again.phase_for(0)
    assert any("using potentials/W/reference.pkl" in ln for ln in lines)                # computed once, then reused
    calc = EdgeSpectra(phase, "W", ipot_of, ["W"], k_grid(12.0), rpath=4.0, nleg=2, device="cuda")
    s = calc.spectrum(sites, np.eye(3) * box, np.arange(0, len(sites), 9), sym)
    assert np.all(np.isfinite(s.chi)) and np.abs(s.chi[(calc.k > 3) & (calc.k < 11)]).max() > 1e-3
