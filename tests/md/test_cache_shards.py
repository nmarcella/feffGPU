"""The HDF5 cache: shards merge to the single run, a stopped run continues, the averages come from the cache."""
import json
import h5py
import numpy as np
import pytest
from mdexafs_testing import gpu, rock_salt_run

ROW_DATASETS = ("frame_index", "n_absorbers", "chi", "chi_ss", "chi_ms", "chi_types", "ss_histogram", "ms_binned")


def _rows(path, edge="C"):
    with h5py.File(path, "r") as f:
        g = f["edges"][edge]
        order = np.argsort(g["frame_index"][:])
        return {ds: g[ds][:][order] for ds in ROW_DATASETS}


@gpu
def test_three_shards_merge_to_the_single_run(tmp_path):
    from feffgpu.md import average_cache, load_config, merge_shards, run
    one = rock_salt_run(tmp_path, n_frames=7, output="single")
    three = rock_salt_run(tmp_path, n_frames=7, output="sharded")
    run(load_config(one), log=lambda m: None)
    for i in (1, 2, 3):
        s = run(load_config(three), shard=f"{i}/3", log=lambda m: None)
        assert (tmp_path / f"sharded/chi.shard-{i}-of-3.h5").is_file() and s["frames_in_shard"] == len(range(i - 1, 7, 3))
        assert (tmp_path / f"sharded/report.shard-{i}-of-3.json").is_file()
    res = merge_shards(tmp_path / "sharded", log=lambda m: None)
    assert res["frames"] == 7 and res["shards"] == 3
    a, b = _rows(tmp_path / "single/chi.h5"), _rows(tmp_path / "sharded/chi.h5")
    assert np.array_equal(a["frame_index"], np.arange(7)) and np.array_equal(b["frame_index"], np.arange(7))
    for ds in ROW_DATASETS[1:]:
        scale = float(np.abs(a[ds]).max()) or 1.0
        assert np.abs(a[ds].astype(float) - b[ds].astype(float)).max() <= 1e-5 * scale, ds
    ra, rb = average_cache(tmp_path / "single/chi.h5")["C"], average_cache(tmp_path / "sharded/chi.h5")["C"]
    assert np.allclose(ra["mean"], rb["mean"], rtol=0, atol=1e-6 * np.abs(ra["mean"]).max())
    assert np.allclose(ra["stderr"], rb["stderr"], rtol=0, atol=1e-6 * np.abs(ra["mean"]).max())
    with h5py.File(tmp_path / "sharded/chi.h5") as f:
        assert f.attrs["merged"] and f.attrs["n_rows"] == 7


@gpu
def test_a_stopped_run_continues_from_its_cache(tmp_path):
    from feffgpu.md import load_config, run
    ref = rock_salt_run(tmp_path, n_frames=6, output="ref")
    run(load_config(ref), log=lambda m: None)
    cfg_path = rock_salt_run(tmp_path, n_frames=6, output="stopped")

    class Stop(Exception):
        pass

    def stop_after_first_block(line):
        if "frames (frame" in line:                                                    # the progress line of a written block
            raise Stop()
    with pytest.raises(Stop):
        run(load_config(cfg_path), log=stop_after_first_block)
    with h5py.File(tmp_path / "stopped/chi.h5") as f:
        assert f.attrs["n_rows"] == 2                                                   # one block of two frames was kept
    lines = []
    s = run(load_config(cfg_path), log=lines.append)
    assert s["frames_computed_now"] == 4 and any("continuing" in ln for ln in lines)
    a, b = _rows(tmp_path / "ref/chi.h5"), _rows(tmp_path / "stopped/chi.h5")
    for ds in ROW_DATASETS:
        scale = float(np.abs(a[ds]).max()) or 1.0
        assert np.abs(a[ds].astype(float) - b[ds].astype(float)).max() <= 1e-5 * scale, ds
    with h5py.File(tmp_path / "stopped/chi.h5", "r+") as f:                            # a block cut short: rows beyond n_rows are dropped
        f.attrs["n_rows"] = 5
    s2 = run(load_config(cfg_path), log=lambda m: None)
    assert s2["frames_computed_now"] == 1 and s2["frames_in_cache"] == 6
    assert np.array_equal(_rows(tmp_path / "stopped/chi.h5")["frame_index"], np.arange(6))


@gpu
def test_other_settings_are_not_mixed_into_a_cache(tmp_path):
    from feffgpu.md import load_config, run
    p = rock_salt_run(tmp_path, n_frames=2, output="mix")
    run(load_config(p), log=lambda m: None)
    changed = rock_salt_run(tmp_path, n_frames=2, output="mix", rpath=4.5)
    with pytest.raises(ValueError, match="other settings.*rpath"):
        run(load_config(changed), log=lambda m: None)


@gpu
def test_absorber_subsets_are_reproducible_and_independent_of_shards(tmp_path):
    from feffgpu.md import load_config, run
    p = rock_salt_run(tmp_path, n_frames=4, output="a", absorbers=5)
    run(load_config(p), log=lambda m: None)
    q = rock_salt_run(tmp_path, n_frames=4, output="b", absorbers=5)
    for i in (1, 2):
        run(load_config(q), shard=f"{i}/2", log=lambda m: None)
    from feffgpu.md import merge_shards
    merge_shards(tmp_path / "b", log=lambda m: None)
    a, b = _rows(tmp_path / "a/chi.h5"), _rows(tmp_path / "b/chi.h5")
    assert np.all(a["n_absorbers"] == 5)
    assert np.abs(a["chi"] - b["chi"]).max() <= 1e-5 * np.abs(a["chi"]).max()           # the same absorbers in every frame
    rep = json.loads((tmp_path / "a/report.json").read_text())
    assert rep["edges"]["C"]["absorbers_per_frame"] == "5"


@gpu
def test_extxyz_and_lammps_dump_of_the_same_frames_give_the_same_spectra(tmp_path):
    """A bcc W block with Gaussian displacements, written as extended xyz (elements, cell in every frame) and as a
    LAMMPS dump (numeric type 1, static cell from the configuration): the cached spectra agree."""
    from feffgpu.md import load_config, run
    from mdexafs_testing import PHASE_W, bcc_sites, gaussian_frames, write_config, write_extxyz, write_lammps
    sites, box = bcc_sites(4)
    frames = gaussian_frames(sites, 3, 0.05, seed=9)
    write_extxyz(tmp_path / "w.extxyz", ["W"] * len(sites), frames, np.eye(3) * box)
    write_lammps(tmp_path / "w.lammpstrj", [1] * len(sites), frames, box)
    edge = [{"element": "W", "potentials": "file", "phase": str(PHASE_W), "rpath": 4.5, "nleg": 3, "criteria": 1.0, "kmax": 16.0}]
    write_config(tmp_path / "x.toml", {"system": {"trajectory": "w.extxyz", "output": "x"}, "lattice": {"dynamic": True},
                                       "frames": {"absorbers_per_frame": 10}, "edge": edge})
    write_config(tmp_path / "l.toml", {"system": {"trajectory": "w.lammpstrj", "output": "l"}, "lattice": {"a": box, "b": box, "c": box},
                                       "atoms": {"1": "W"}, "frames": {"absorbers_per_frame": 10}, "edge": edge})
    for name in ("x", "l"):
        run(load_config(tmp_path / f"{name}.toml"), log=lambda m: None)
    a, b = _rows(tmp_path / "x/chi.h5", "W"), _rows(tmp_path / "l/chi.h5", "W")
    for ds in ("chi", "chi_types", "ss_histogram"):
        assert np.abs(a[ds] - b[ds]).max() <= 1e-4 * np.abs(a[ds]).max(), ds


def test_path_type_selection_rules(tmp_path):
    """Either spelling of a type, limits of multiple-scattering types on bin boundaries only, and one list for several
    edges (each edge takes the types of its own absorber)."""
    from feffgpu.md.average import resolve_selection
    with h5py.File(tmp_path / "g.h5", "w") as f:
        g = f.create_group("edges/C")
        g.attrs.update({"path_types": json.dumps(["C-Zn", "C-C", "C-C-C", "C-C-Zn", "C-Zn-Zn"]), "ss_types": json.dumps(["C-Zn", "C-C"]),
                        "ms_types": json.dumps(["C-C-C", "C-C-Zn", "C-Zn-Zn"]), "rpath": 5.0, "element": "C"})
        g.create_dataset("ms_bin_edges", data=0.5 * np.arange(11))
        items = resolve_selection(g, ["C-Zn", "C-Zn-C", "Zn-C"], [2.6, 4.0, 3.0])               # Zn-C belongs to another edge
        assert [(i["name"], i["kind"], i["max_distance"]) for i in items] == [("C-Zn", "ss", 2.6), ("C-C-Zn", "ms", 4.0)]
        assert resolve_selection(g, ["Zn-C"], [3.0]) == []                                       # nothing for this edge: the total
        assert resolve_selection(g, ["C-C-C"], 6.0)[0]["max_distance"] is None                   # beyond rpath: the whole type
        with pytest.raises(ValueError, match="not a boundary of the stored length bins"):
            resolve_selection(g, ["C-C-C"], 4.2)
        with pytest.raises(ValueError, match="not a type of edge"):
            resolve_selection(g, ["C-O"], None)


def test_merge_refuses_incomplete_shards(tmp_path):
    from feffgpu.md import merge_shards
    (tmp_path / "chi.shard-1-of-3.h5").write_bytes(b"")
    with pytest.raises(ValueError, match=r"shards \[2, 3\] of 3 are missing"):
        merge_shards(tmp_path, log=lambda m: None)
    with pytest.raises(FileNotFoundError):
        merge_shards(tmp_path / "nowhere", log=lambda m: None)
