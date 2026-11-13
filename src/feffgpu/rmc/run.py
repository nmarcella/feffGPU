"""From a configuration to a run: building the model cells and the edges, running the search or the refinement.

``run_search(config)`` runs ``rmc-exafs run``; ``run_refine(config)`` runs ``rmc-exafs refine``. Both accept a path
to a TOML file or an ``RmcConfig``.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from ..exafs import k_grid, read_chi, to_grid, check_window_inside_data
from ..exafs.edge import EdgeModel
from ..pad import load_phase
from ..parallel import primary_device
from .config import RmcConfig, load_config
from .structure import ModelCell, fluorite_sites, read_site_file, sublattice_sites
from .search import RmcSample, EvaxSearch
from .outputs import write_json

__all__ = ["build_sample", "build_samples", "run_search", "run_refine", "KMAX_GRID"]

KMAX_GRID = 24.0          # the model's k grid runs from 0 to 24 1/A in steps of 0.05 1/A


def _value(v, name):
    return float(v[name]) if isinstance(v, dict) else float(v)


def _start_value(table, name, default):
    if isinstance(table, dict):
        return float(table.get(name, default))
    return float(table)


def build_sample(s, cfg: RmcConfig, device="cuda", phases=None):
    """The ``RmcSample`` of one ``[[sample]]`` block: the model cell (with the hop sublattice, if any) and its edges.

    Parameters
    ----------
    s : SampleSettings
    cfg : RmcConfig
        For the moves (the hop sublattice) and the start values of S0^2 and the E0 shift.
    device : str
    phases : dict, optional
        A cache ``{path: PhaseData}`` shared between samples.
    """
    phases = {} if phases is None else phases
    model = ModelCell.from_file(s.structure, supercell=s.supercell, format=s.format)
    model.name = s.label
    hops = cfg.moves.hops
    spec = s.hop_sites or (hops.sites if hops is not None else None)
    if hops is not None and spec is not None and (model.symbols == hops.element).any():
        if spec == "fluorite":
            model.sites = fluorite_sites(model.symbols, model.positions, model.cell, hops.element)
        else:
            model.sites = sublattice_sites(model.symbols, model.positions, model.cell, hops.element, read_site_file(spec), radius=hops.radius)
    k = k_grid(KMAX_GRID)
    edges = []
    for e in s.edge:
        kd, cd = read_chi(e.chi, *e.chi_columns)
        check_window_inside_data(kd, e.window.kmin, e.window.kmax, e.window.dk, name=f"{s.label}, spectrum {e.name} ({e.chi})")
        if e.window.kmax + e.window.dk / 2 > KMAX_GRID:
            raise ValueError(f"{s.label}, spectrum {e.name}: the window reaches beyond the model's k grid ({KMAX_GRID} 1/A)")
        data = to_grid(k, kd, cd)
        if e.phase not in phases:
            phases[e.phase] = load_phase(e.phase)
        edges.append(EdgeModel(e.name, k, data, e.window.fit_window(), phases[e.phase], model.symbols, model.cell, e.absorber,
                               potentials=e.potentials, e0=_start_value(cfg.amplitude.de0, e.name, 0.0), s02=_start_value(cfg.amplitude.s02, e.name, 1.0),
                               rpath=e.rpath, nleg=e.nleg, criteria=e.criteria, multiple_scattering=e.multiple_scattering,
                               pair_windows=e.pair_windows, pair_min=e.pair_min, ignore=e.ignore, objective=e.objective, wavelet_r0=e.wavelet_r0,
                               device=device, criteria_rule=e.criteria_rule))
    return RmcSample(s.label, model, edges)


def build_samples(cfg: RmcConfig, device="cuda"):
    """Every sample of a configuration (the phase files are read once)."""
    phases = {}
    return [build_sample(s, cfg, device=device, phases=phases) for s in cfg.sample]


def _config(config):
    return load_config(config) if isinstance(config, (str, Path)) else config


def run_search(config, log=print):
    """Run the population search of a configuration (``rmc-exafs run``).

    Parameters
    ----------
    config : str, Path or RmcConfig
        A TOML file (relative paths relative to it) or a validated configuration with absolute paths.
    log : callable
        Receives the progress lines.

    Returns
    -------
    dict
        The summary written to ``run.json``.
    """
    cfg = _config(config)
    if cfg.search is None:
        raise ValueError("the configuration has no [search] section (needed by `rmc-exafs run`)")
    device = primary_device(cfg.run.devices)
    out = Path(cfg.run.output)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_samples(cfg, device=device)
    search = EvaxSearch(samples, cfg.search, amplitude=cfg.amplitude, moves=cfg.moves, constraints=cfg.constraints, seed=cfg.run.seed, output=out,
                        save_every=cfg.run.save_every, log=log, resolved=cfg.model_dump(mode="json"))
    return search.run()


def run_refine(config, log=print):
    """Run the gradient refinement of a configuration (``rmc-exafs refine``): its first sample, whose first spectrum's
    absorber is the only element.

    Writes ``<output>/<label>/refine.json`` (settings, history, misfits) and ``refine.npz`` (``start``, ``positions``,
    ``k``, ``target``, ``model_start``, ``model_final``). Returns the summary."""
    cfg = _config(config)
    if cfg.refine is None:
        raise ValueError("the configuration has no [refine] section (needed by `rmc-exafs refine`)")
    from .refine import GpuRmc
    rs = cfg.refine
    device = primary_device(cfg.run.devices)
    s = cfg.sample[0]
    e = s.edge[0]
    model = ModelCell.from_file(s.structure, supercell=s.supercell, format=s.format)
    if set(model.symbols.tolist()) != {e.absorber}:
        raise ValueError(f"the refinement takes a structure of one element, the absorber {e.absorber} (got {sorted(set(model.symbols.tolist()))}); "
                         "use `rmc-exafs run` for other structures")
    if model.cell is None:
        raise ValueError("the refinement needs a periodic cell")
    k = k_grid(KMAX_GRID)
    kd, cd = read_chi(e.chi, *e.chi_columns)
    check_window_inside_data(kd, e.window.kmin, e.window.kmax, e.window.dk, name=f"{s.label}, spectrum {e.name}")
    s02 = _start_value(cfg.amplitude.s02, e.name, 1.0)
    de0 = _start_value(cfg.amplitude.de0, e.name, 0.0)
    target = to_grid(k, kd, cd) / s02
    phase = load_phase(e.phase)
    t0 = time.time()
    rmc = GpuRmc(phase, k, target, de0, model.cell, window=(e.window.kmin, e.window.kmax), r_range=(e.window.rmin, e.window.rmax), rmax=e.rpath,
                 critpw=e.criteria, r_min=rs.core_distance, core=rs.core_stiffness, ms=e.multiple_scattering and e.nleg >= 3,
                 kweight=e.window.kweight, dk=e.window.dk, device=device, criteria_rule=e.criteria_rule, sites=model.positions)
    path_types = [] if rmc.ms is None else rmc.ms.types
    if rmc.ms is not None and e.criteria_rule == "evax":
        kept = [t for t in path_types if t["kept"]]
        log(f"[{s.label}] paths chosen on the start lattice by EvAX's rule ({e.criteria} % of the size of the lattice's spectrum): {len(kept)} "
            f"types, degeneracy per atom " + ", ".join(f"{n} legs {sum(t['degeneracy'] for t in kept if t['legs'] == n):.0f}" for n in (3, 4)))
    rng = np.random.default_rng(cfg.run.seed)
    start = np.repeat(model.positions[None], rs.copies, 0) + (rng.normal(0.0, rs.start_kick, (rs.copies, model.n, 3)) if rs.start_kick > 0 else 0.0)
    chi0, M0 = rmc.spectrum_only(start)
    log(f"[{s.label}] refinement of {rs.copies} copies of {model.n} {e.absorber} atoms against {e.chi} (S0^2 {s02}, E0 shift {de0} eV); "
        f"start misfit {M0:.5f}")
    final, history = rmc.refine(start, rs.steps, lr=rs.step_size, ms_absorbers=rs.ms_absorbers, ms_alpha=rs.ms_memory, value_every=rs.value_every,
                                refresh_every=rs.refresh_every, optimizer=rs.optimizer, lr_final=rs.final_fraction, seed=cfg.run.seed,
                                log_every=rs.log_every, log=log)
    chi1, M1 = rmc.spectrum_only(final)
    out = Path(cfg.run.output) / s.label
    out.mkdir(parents=True, exist_ok=True)
    summary = {"method": "feffgpu.rmc: gradient refinement", "label": s.label, "edge": e.name, "copies": rs.copies, "atoms": model.n,
               "misfit_start": M0, "misfit_final": M1, "settings": rs.model_dump(), "s02": s02, "de0": de0, "steps_info": rmc.steps_info,
               "criteria_rule": e.criteria_rule, "path_types": path_types,
               "timing": rmc.timing, "history": history, "seconds": time.time() - t0}
    write_json(out / "refine.json", summary)
    np.savez_compressed(out / "refine.npz", start=start, positions=final, sites=model.positions, k=k, target=target,
                        model_start=chi0.cpu().numpy(), model_final=chi1.cpu().numpy(), cell=model.cell, symbols=model.symbols,
                        window=np.array([e.window.kmin, e.window.kmax, e.window.dk, e.window.kweight, e.window.rmin, e.window.rmax], float))
    from ..config_tools import to_toml
    (Path(cfg.run.output) / "config.resolved.toml").write_text(to_toml(cfg.model_dump(mode="json")))
    log(f"[{s.label}] misfit {M0:.5f} -> {M1:.5f}; written {out}/refine.json, refine.npz")
    return summary
