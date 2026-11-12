"""Reading trajectories without OVITO: formats, type maps, static and dynamic cells, frame selections, streaming."""
import tracemalloc
import numpy as np
import pytest
from mdexafs_testing import bcc_sites, gaussian_frames, write_extxyz, write_lammps


def _bcc_frames(n_frames=5, sd=0.05):
    sites, box = bcc_sites(3)
    return np.array(["W"] * len(sites)), gaussian_frames(sites, n_frames, sd, seed=2), box


def test_extxyz_static_and_dynamic_cells_and_selection(tmp_path):
    from feffgpu.io.trajectory import TrajectoryReader
    sym, frames, box = _bcc_frames(7)
    cells = [np.eye(3) * box * (1 + 0.01 * i) for i in range(7)]
    write_extxyz(tmp_path / "t.extxyz", sym, frames, np.array(cells))
    dyn = TrajectoryReader(tmp_path / "t.extxyz", dynamic=True)
    assert dyn.format == "extxyz" and dyn.count() == 7
    sel = dyn.selection(1, None, 2)
    got = list(dyn.iter_frames(sel))
    assert [g.index for g in got] == [1, 3, 5] == sel
    assert all(np.allclose(g.positions, frames[g.index], atol=1e-7) for g in got)
    assert all(np.allclose(g.cell, cells[g.index]) for g in got)                                  # the cell of every frame (NPT)
    static = TrajectoryReader(tmp_path / "t.extxyz", cell=np.eye(3) * 20.0)
    assert np.allclose(static.read_frame(4).cell, np.eye(3) * 20.0)                               # the static cell wins
    odd = list(dyn.iter_frames([0, 2, 3, 6]))                                                      # not an arithmetic progression
    assert [g.index for g in odd] == [0, 2, 3, 6]
    assert dyn.selection(0, 100, 3) == [0, 3, 6]                                                   # a stop beyond the end is clipped
    with pytest.raises(ValueError):
        dyn.selection(9)


def test_lammps_dump_with_type_map_matches_the_extxyz(tmp_path):
    from feffgpu.io.trajectory import TrajectoryReader
    sites, box = bcc_sites(3)
    frames = gaussian_frames(sites, 4, 0.05, seed=3)
    types = np.where(np.arange(len(sites)) % 2 == 0, 1, 2)                                         # two numeric types
    write_lammps(tmp_path / "d.lammpstrj", types, frames, box)
    r = TrajectoryReader(tmp_path / "d.lammpstrj", type_map={1: "W", "2": "Mo"}, cell=np.eye(3) * box)
    assert r.format == "lammps-dump-text" and r.count() == 4
    got = list(r.iter_frames(r.selection(0, None, 1)))
    want = np.where(types == 1, "W", "Mo")
    assert all(np.array_equal(g.symbols, want) for g in got)
    assert all(np.allclose(g.positions, frames[g.index], atol=1e-6) for g in got)
    dyn = TrajectoryReader(tmp_path / "d.lammpstrj", type_map={1: "W", 2: "Mo"}, dynamic=True)        # the dump's box
    assert np.allclose(dyn.read_frame(2).cell, np.eye(3) * box, atol=1e-8)
    with pytest.raises(ValueError, match="not in the type map"):
        TrajectoryReader(tmp_path / "d.lammpstrj", type_map={1: "W"}, cell=np.eye(3) * box).read_frame(0)
    with pytest.raises(ValueError, match="not an element symbol"):
        TrajectoryReader(tmp_path / "d.lammpstrj", type_map={1: "W", 2: "Qq"}, cell=np.eye(3) * box)


def test_xdatcar_with_a_variable_cell_is_streamed(tmp_path):
    from feffgpu.io.trajectory import TrajectoryReader
    rng = np.random.default_rng(4)
    frames, cells = [], []
    with open(tmp_path / "XDATCAR", "w") as f:
        for i in range(5):
            c = np.array([[6.0 + 0.02 * i, 0, 0], [0.3, 6.0, 0], [0, 0, 6.1]])
            frac = rng.uniform(0, 1, (4, 3))
            frames.append(frac @ c), cells.append(c)
            f.write("variable cell\n1.0\n" + "\n".join(" ".join(f"{v:.8f}" for v in row) for row in c) + "\nZn C\n2 2\n")
            f.write(f"Direct configuration= {i + 1}\n" + "\n".join(" ".join(f"{v:.10f}" for v in row) for row in frac) + "\n")
    r = TrajectoryReader(tmp_path / "XDATCAR", dynamic=True)
    assert r.format == "vasp-xdatcar" and r.count() == 5
    got = list(r.iter_frames([1, 4]))
    assert [g.index for g in got] == [1, 4] and list(got[0].symbols) == ["Zn", "Zn", "C", "C"]
    assert all(np.allclose(g.positions, frames[g.index], atol=1e-7) and np.allclose(g.cell, cells[g.index]) for g in got)
    c = cells[0]                                                                                  # a constant cell: one header
    with open(tmp_path / "const", "w") as f:
        f.write("constant cell\n2.0\n" + "\n".join(" ".join(f"{v / 2:.8f}" for v in row) for row in c) + "\nZn C\n2 2\n")
        for i in range(3):
            frac = frames[i] @ np.linalg.inv(c)
            f.write(f"Direct configuration= {i + 1}\n" + "\n".join(" ".join(f"{v:.10f}" for v in row) for row in frac) + "\n")
    rc = TrajectoryReader(tmp_path / "const", format="vasp-xdatcar", dynamic=True)
    assert rc.count() == 3
    assert all(np.allclose(g.positions, frames[g.index], atol=1e-7) and np.allclose(g.cell, c) for g in rc.iter_frames([0, 2]))


def test_cp2k_xyz_with_a_static_cell(tmp_path):
    from feffgpu.io.trajectory import TrajectoryReader
    sym, frames, box = _bcc_frames(3)
    with open(tmp_path / "run-pos-1.xyz", "w") as f:                                             # CP2K's coordinate output
        for i, p in enumerate(frames):
            f.write(f"{len(p):8d}\n i =     {10 * i}, time =      {5.0 * i:.3f}, E =     -1234.5678901234\n")
            f.write("".join(f"  {s} {x:20.10f} {y:20.10f} {z:20.10f}\n" for s, (x, y, z) in zip(sym, p)))
    r = TrajectoryReader(tmp_path / "run-pos-1.xyz", cell=np.eye(3) * box)
    assert r.format == "extxyz" and r.count() == 3
    got = list(r.iter_frames([0, 2]))
    assert all(np.allclose(g.positions, frames[g.index], atol=1e-8) and np.allclose(g.cell, np.eye(3) * box) for g in got)


def test_ase_traj_files(tmp_path):
    from ase import Atoms
    from ase.io.trajectory import Trajectory
    from feffgpu.io.trajectory import TrajectoryReader
    sym, frames, box = _bcc_frames(4)
    with Trajectory(str(tmp_path / "md.traj"), "w") as t:
        for p in frames:
            t.write(Atoms(list(sym), positions=p, cell=np.eye(3) * box, pbc=True))
    r = TrajectoryReader(tmp_path / "md.traj", dynamic=True)
    assert r.format == "traj" and r.count() == 4
    got = list(r.iter_frames([1, 3]))
    assert [g.index for g in got] == [1, 3] and all(np.allclose(g.positions, frames[g.index]) for g in got)
    assert np.allclose(got[0].cell, np.eye(3) * box)


def test_errors_name_the_problem(tmp_path):
    from feffgpu.io.trajectory import TrajectoryReader
    sym, frames, box = _bcc_frames(2)
    write_extxyz(tmp_path / "nocell.extxyz", sym, frames, None)
    with pytest.raises(ValueError, match="no periodic cell"):
        TrajectoryReader(tmp_path / "nocell.extxyz", dynamic=True).read_frame(0)
    cluster = TrajectoryReader(tmp_path / "nocell.extxyz", periodic=False)
    assert cluster.read_frame(1).cell is None
    with open(tmp_path / "changes.extxyz", "w") as f:                                             # the composition changes
        from feffgpu.io.trajectory import write_extxyz_frame
        write_extxyz_frame(f, sym, frames[0], np.eye(3) * box)
        write_extxyz_frame(f, sym[:-1], frames[1][:-1], np.eye(3) * box)
    with pytest.raises(ValueError, match="same atoms in the same order"):
        list(TrajectoryReader(tmp_path / "changes.extxyz", dynamic=True).iter_frames([0, 1]))
    with pytest.raises(FileNotFoundError):
        TrajectoryReader(tmp_path / "missing.extxyz", dynamic=True)
    with pytest.raises(ValueError, match="exclude each other"):
        TrajectoryReader(tmp_path / "nocell.extxyz", cell=np.eye(3) * box, dynamic=True)


def test_mdanalysis_reader_is_optional_with_a_clear_message(tmp_path):
    from feffgpu.io.trajectory import TrajectoryReader
    pytest.importorskip("ase")
    try:
        import MDAnalysis  # noqa: F401
        pytest.skip("MDAnalysis is installed; this test checks the message when it is missing")
    except ImportError:
        pass
    sym, frames, box = _bcc_frames(1)
    write_extxyz(tmp_path / "t.extxyz", sym, frames, np.eye(3) * box)
    r = TrajectoryReader(tmp_path / "t.extxyz", reader="mdanalysis", dynamic=True)
    with pytest.raises(ImportError, match=r"pip install 'feffgpu\[mda\]'"):
        r.count()


def _write_many_frames(path, n_frames, n_atoms=8, lammps=False, tail=""):
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 6, (n_atoms, 3))
    sym = ["Zn"] * (n_atoms // 2) + ["C"] * (n_atoms - n_atoms // 2)
    with open(path, "w") as f:
        for i in range(n_frames):
            p = base + 0.01 * rng.normal(size=(n_atoms, 3))
            if lammps:
                rows = [f"{j + 1} {1 if s == 'Zn' else 2} {x:.5f} {y:.5f} {z:.5f}" for j, (s, (x, y, z)) in enumerate(zip(sym, p))]
                f.write(f"ITEM: TIMESTEP\n{i}\nITEM: NUMBER OF ATOMS\n{n_atoms}\nITEM: BOX BOUNDS pp pp pp\n0 6\n0 6\n0 6\n"
                        "ITEM: ATOMS id type x y z\n" + "\n".join(rows) + "\n")
            else:
                f.write(f'{n_atoms}\nLattice="6 0 0 0 6 0 0 0 6" Properties=species:S:1:pos:R:3 pbc="T T T" frame={i}\n'
                        + "\n".join(f"{s} {x:.5f} {y:.5f} {z:.5f}" for s, (x, y, z) in zip(sym, p)) + "\n")
        f.write(tail)


def test_a_10000_frame_trajectory_streams_without_holding_the_frames(tmp_path):
    """Iterating 10,000 frames keeps the traced memory far below what the frames themselves take, and a LAMMPS dump
    yields its first frames before the reader reaches a damaged end of the file (frames are read lazily)."""
    from feffgpu.io.trajectory import TrajectoryReader
    path = tmp_path / "long.extxyz"
    _write_many_frames(path, 10000)
    r = TrajectoryReader(path, dynamic=True)
    assert r.count() == 10000
    sel = r.selection()
    tracemalloc.start()
    n = 0
    for fr in r.iter_frames(sel):
        n += 1
    _cur, peak_stream = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    held = list(r.iter_frames(sel[:2000]))                                                        # a fifth of the frames, held
    held_bytes, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert n == 10000 and len(held) == 2000
    assert peak_stream < 6e6, peak_stream                                                         # a few MB: one frame and the offsets
    assert peak_stream < 2.5 * held_bytes, (peak_stream, held_bytes)                              # less than half of all 10,000 frames held
    dump = tmp_path / "damaged.lammpstrj"
    _write_many_frames(dump, 10000, lammps=True, tail="ITEM: TIMESTEP\nnot a number\nITEM: ATOMS id type x y z\ngarbage\n")
    lr = TrajectoryReader(dump, type_map={1: "Zn", 2: "C"}, cell=np.eye(3) * 6.0)
    first = list(lr.iter_frames([0, 1, 2]))                                                       # the damaged end is never read
    assert [f.index for f in first] == [0, 1, 2]
    tracemalloc.start()
    for fr in lr.iter_frames(list(range(0, 9999, 3))):
        pass
    _c, peak_dump = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_dump < 6e6, peak_dump
