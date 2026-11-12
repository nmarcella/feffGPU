"""The optional dynamics runner, with ASE's EMT calculator (no extra package, CPU): it writes a trajectory that the
reader streams and a log that the convergence tools read."""
import numpy as np
import pytest
from mdexafs_testing import write_config


def test_emt_dynamics_writes_a_readable_trajectory_and_log(tmp_path):
    import ase.io
    from ase.build import bulk
    from feffgpu.analysis.convergence import read_log, thermo_summary
    from feffgpu.io.trajectory import TrajectoryReader
    from feffgpu.md import load_config
    from feffgpu.md.dynamics import run_dynamics
    ase.io.write(tmp_path / "Cu.cif", bulk("Cu", "fcc", a=3.61, cubic=True))
    p = write_config(tmp_path / "dyn.toml", {
        "system": {"trajectory": "traj.extxyz", "output": "out"}, "lattice": {"dynamic": True},
        "edge": [{"element": "Cu", "potentials": "reference"}],
        "dynamics": {"structure": "Cu.cif", "supercell": [2, 2, 2], "calculator": "emt", "temperature": 300.0, "timestep": 2.0,
                     "equilibration": 10, "steps": 30, "save_every": 10, "log_every": 5, "seed": 3},
        "run": {"devices": "cpu"}})
    cfg = load_config(p)
    res = run_dynamics(cfg, log=lambda m: None, device="cpu")
    assert res["frames"] == 3 and res["atoms"] == 32 and res["temperature_mean"] > 0
    r = TrajectoryReader(tmp_path / "traj.extxyz", dynamic=True)
    frames = list(r.iter_frames(r.selection()))
    assert len(frames) == 3 and np.allclose(frames[0].cell, np.eye(3) * 2 * 3.61) and set(frames[0].symbols) == {"Cu"}
    log = read_log(tmp_path / "md.log")
    assert log["format"] == "ase" and len(log["time"]) == 9                           # steps 0, 5, ..., 40
    assert thermo_summary(log)["temperature"]["mean"] > 0
    assert (tmp_path / "out/dynamics.json").is_file()


def test_calculator_names():
    from feffgpu.md.dynamics import make_calculator
    assert type(make_calculator("ase.calculators.emt:EMT")).__name__ == "EMT"
    with pytest.raises(ValueError, match="unknown calculator"):
        make_calculator("nonsense")
    with pytest.raises(ImportError):
        make_calculator("no_such_module_xyz:Calc")
