"""An optional, small runner that makes a trajectory for users who do not have one.

NVT Langevin dynamics with ASE (``ase.md.langevin.Langevin``) and any ASE calculator:

- ``calculator = "mace_mp"``: a MACE foundation model through ``mace.calculators.mace_mp`` (the optional extra
  ``pip install 'feffgpu[mace]'``; the model files are downloaded by MACE the first time);
- ``calculator = "emt"``: ASE's effective-medium potential (a few fcc metals; for tests and demonstrations only);
- ``calculator = "module:function"``: any function that returns an ASE calculator, called with
  ``calculator_options``.

Frames are written every ``save_every`` steps after ``equilibration`` steps, as extended xyz, to the trajectory file
of ``[system]``; the thermodynamic log (ASE's ``MDLogger`` format) goes to ``[dynamics] log``. The trajectory is then
read by ``md-exafs run`` like any other. This runner is a convenience: production dynamics usually come from a
dedicated program (LAMMPS, CP2K, ...), whose trajectories this package reads directly.
"""
from __future__ import annotations
import importlib
import json
import time
from pathlib import Path
import numpy as np
from ..io.trajectory import write_extxyz_frame

__all__ = ["make_calculator", "run_dynamics"]


def make_calculator(name, model="medium", options=None, device="cpu"):
    """An ASE calculator from a short name.

    Parameters
    ----------
    name : str
        ``"mace_mp"``, ``"emt"`` or ``"module:function"``.
    model : str
        The MACE model (``"small"``, ``"medium"``, ``"large"`` or a file) for ``"mace_mp"``.
    options : dict, optional
        Extra keyword arguments of the constructor.
    device : str
        ``"cuda"`` or ``"cpu"`` (MACE only).
    """
    options = dict(options or {})
    if name == "emt":
        from ase.calculators.emt import EMT
        return EMT(**options)
    if name == "mace_mp":
        try:
            from mace.calculators import mace_mp
        except ImportError as err:
            raise ImportError("calculator = \"mace_mp\" needs MACE, which is not installed: pip install 'feffgpu[mace]'") from err
        dev = "cuda" if str(device).startswith("cuda") else "cpu"
        options.setdefault("default_dtype", "float32")
        return mace_mp(model=model, device=dev, **options)
    if ":" in name:
        mod, func = name.split(":", 1)
        try:
            factory = getattr(importlib.import_module(mod), func)
        except (ImportError, AttributeError) as err:
            raise ImportError(f"calculator {name!r}: {err}") from err
        return factory(**options)
    raise ValueError(f"unknown calculator {name!r}: use \"mace_mp\", \"emt\" or \"module:function\"")


def run_dynamics(cfg, log=None, device=None):
    """Run the ``[dynamics]`` of a configuration and write the trajectory and the log.

    Parameters
    ----------
    cfg : MDExafsConfig
    log : callable, optional
    device : str, optional
        Default: the first of ``[run] devices``.

    Returns
    -------
    dict
        ``trajectory``, ``log``, ``frames``, ``atoms``, ``steps``, ``temperature_mean`` (K, production), ``seconds``;
        also written to ``<output>/dynamics.json``.
    """
    import ase.io
    from ase import units
    from ase.constraints import FixCom
    from ase.md import MDLogger
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import Stationary
    try:                                                                       # ASE 3.29 renamed the velocity initializer
        from ase.md.velocitydistribution import thermalize_momenta as _thermalize
    except ImportError:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as _thermalize
    from ..parallel import primary_device
    say = log or (lambda s: print(s, flush=True))
    d = cfg.dynamics
    if d is None:
        raise ValueError("the configuration has no [dynamics] section")
    dev = device or primary_device(cfg.run.devices, quiet=True)
    atoms = ase.io.read(d.structure)
    atoms = atoms.repeat(tuple(d.supercell))
    if not atoms.pbc.all() or abs(np.linalg.det(atoms.cell.array)) < 1e-9:
        raise ValueError(f"{Path(d.structure).name}: the starting structure must be periodic (a cell in all three directions)")
    atoms.calc = make_calculator(d.calculator, d.model, d.calculator_options, dev)
    rng = np.random.default_rng(d.seed)
    _thermalize(atoms, temperature_K=d.temperature, rng=rng)
    Stationary(atoms)
    atoms.set_constraint(FixCom())                                             # the center of mass stays fixed (no drift)
    traj = Path(cfg.system.trajectory)
    traj.parent.mkdir(parents=True, exist_ok=True)
    logfile = Path(d.log)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    dyn = Langevin(atoms, timestep=d.timestep * units.fs, temperature_K=d.temperature, friction=d.friction / units.fs, fixcm=False,
                   rng=np.random.default_rng(d.seed + 1))
    logger = MDLogger(dyn, atoms, str(logfile), header=True, stress=False, peratom=False, mode="w")
    dyn.attach(logger, interval=d.log_every)
    temps, frames = [], [0]
    t0 = time.time()
    total = d.equilibration + d.steps
    say(f"dynamics: {len(atoms)} atoms, {d.calculator} ({d.model if d.calculator == 'mace_mp' else 'default'}), {d.temperature} K, "
        f"{d.timestep} fs steps: {d.equilibration} equilibration + {d.steps} production; frames every {d.save_every} steps -> {traj.name}")
    with open(traj, "w") as fh:
        def write():
            step = dyn.nsteps
            if step > d.equilibration and (step - d.equilibration) % d.save_every == 0:
                write_extxyz_frame(fh, atoms.get_chemical_symbols(), atoms.get_positions(), atoms.cell.array,
                                   info=f"step={step} time_fs={step * d.timestep:.3f} temperature_K={atoms.get_temperature():.3f}")
                frames[0] += 1
            if step > d.equilibration:
                temps.append(atoms.get_temperature())
            if step % max(1, total // 10) == 0 and step > 0:
                say(f"  step {step}/{total}: {atoms.get_temperature():.1f} K ({time.time() - t0:.0f} s)")
        dyn.attach(write, interval=1)
        dyn.run(total)
    logger.close()
    summary = {"command": "dynamics", "trajectory": str(traj), "log": str(logfile), "frames": frames[0], "atoms": len(atoms),
               "steps": total,
               "temperature_target": d.temperature, "temperature_mean": float(np.mean(temps)) if temps else None,
               "calculator": d.calculator, "seconds": round(time.time() - t0, 2)}
    out = Path(cfg.output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dynamics.json").write_text(json.dumps(summary, indent=1))
    say(f"dynamics done: {frames[0]} frames written to {traj} ({summary['seconds']:.0f} s)")
    return summary
