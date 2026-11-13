"""Running MD-EXAFS: from a configuration to the cache of per-frame spectra.

``run`` reads the trajectory frame by frame, computes the spectrum of every selected frame for every edge, and writes
the results into the cache in blocks. With a shard (``Shard(3, 16)``) it computes only that shard's frames (every
16th frame of the selection, starting with the third) and writes ``chi.shard-3-of-16.h5``. A run that stopped
continues from its cache. The loop over frames goes through ``feffgpu.parallel.run_items``, the access point for
several GPUs.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np
from ..exafs import k_grid
from ..io.structure import supercell
from ..io.trajectory import TrajectoryReader
from ..parallel import Shard, primary_device, run_items
from .cache import CacheWriter, cache_name, run_settings
from .config import MDExafsConfig, load_config, resolved_dict, to_toml
from .potentials import PhaseProvider
from .spectrum import EdgeSpectra, choose_absorbers

__all__ = ["load_run", "make_reader", "write_resolved_config", "run", "FrameJob"]


def load_run(path):
    """A run from a configuration file or from an output folder (which holds ``config.resolved.toml``).

    Returns
    -------
    cfg : MDExafsConfig
    output : Path
    """
    p = Path(path)
    if p.is_dir():
        res = p / "config.resolved.toml"
        if not res.is_file():
            raise FileNotFoundError(f"{p} holds no config.resolved.toml (is it the output folder of a run?)")
        cfg = load_config(res)
        return cfg, p.resolve()
    cfg = load_config(p)
    return cfg, cfg.output_path


def make_reader(cfg):
    """The trajectory reader of a configuration."""
    lat = cfg.lattice
    periodic = lat.periodic
    static = lat.cell()
    dynamic = periodic and static is None
    return TrajectoryReader(cfg.system.trajectory, format=cfg.system.format, reader=cfg.system.reader, topology=cfg.system.topology,
                            type_map=cfg.atoms, cell=static, dynamic=dynamic, periodic=periodic)


def _atomic_text(path, text):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_resolved_config(cfg, folder):
    """Write ``config.resolved.toml`` (every default filled in, absolute paths) into ``folder``; returns the text."""
    from .. import __version__
    text = to_toml(resolved_dict(cfg), header=f"MD-EXAFS configuration with every default filled in (feffgpu {__version__}).\n"
                                              "Paths are absolute. `md-exafs run` on this file repeats the run.")
    Path(folder).mkdir(parents=True, exist_ok=True)
    _atomic_text(Path(folder) / "config.resolved.toml", text)
    return text


def _replicated(symbols, positions, cell, n):
    if n == [1, 1, 1]:
        return symbols, positions, cell
    if cell is None:
        raise ValueError("[lattice] replicate needs a periodic cell")
    s, p, c = supercell(np.asarray(symbols), np.asarray(positions, float), np.asarray(cell, float), n)
    return s, p, c


class FrameJob:
    """The work of one frame: the spectrum of every edge. Called by ``run_items`` as ``job(frame, device)``.

    Parameters
    ----------
    cfg : MDExafsConfig
    providers : dict
        ``{edge name: PhaseProvider}``.
    candidates : dict
        ``{edge name: indices of the atoms of the absorbing element}`` (in the original cell).
    elements : list of str
        Every element of the system.
    log : callable
    """

    def __init__(self, cfg, providers, candidates, elements, log=print):
        self.cfg, self.providers, self.candidates, self.elements, self.log = cfg, providers, candidates, elements, log
        self.calcs = {}                                                       # edge name -> (phase key, EdgeSpectra)

    def calculator(self, edge, frame_index, device):
        """The ``EdgeSpectra`` of an edge for a frame (built when the frame's phase file changes)."""
        phase, ipot_of, key = self.providers[edge.name].phase_for(frame_index)
        cur = self.calcs.get(edge.name)
        if cur is None or cur[0] != key or str(cur[1].device) != str(device):
            run = self.cfg.run
            calc = EdgeSpectra(phase, edge.element, ipot_of, self.elements, k_grid(edge.kmax), rpath=edge.rpath, nleg=edge.nleg,
                               criteria=edge.criteria, pair_rmin=edge.pair_rmin, pair_dr=edge.pair_dr, ms_bin_width=edge.ms_bin_width,
                               device=device, absorber_batch=run.absorber_batch, path_chunk=run.path_chunk, ms_group=run.ms_group)
            self.calcs[edge.name] = (key, calc)
        return self.calcs[edge.name]

    def __call__(self, frame, device):
        t0 = time.time()
        sym, pos, cell = _replicated(frame.symbols, frame.positions, frame.cell, self.cfg.lattice.replicate)
        out = {"frame_index": int(frame.index), "edges": {}, "keys": {}}
        fr = self.cfg.frames
        for edge in self.cfg.edge:
            key, calc = self.calculator(edge, frame.index, device)
            ab = choose_absorbers(self.candidates[edge.name], fr.absorbers_per_frame, fr.seed, frame.index, edge.element)
            spec = calc.spectrum(pos, cell, ab, sym)
            out["edges"][edge.name] = spec
            out["keys"][edge.name] = key
        out["seconds"] = time.time() - t0
        return out


def _edge_description(edge, calc, cfg):
    return {"element": edge.element, "edge": edge.edge, "k": calc.k, "r_table": calc.r_table, "ms_bin_edges": calc.bin_edges,
            "types": calc.types,
            "attrs": {"potentials": edge.potentials, "rpath": edge.rpath, "nleg": edge.nleg, "criteria": edge.criteria, "kmax": edge.kmax,
                      "pair_rmin": calc.r0, "pair_dr": calc.dr, "ms_bin_width": edge.ms_bin_width, "ipot_of": calc.ipot_of,
                      "absorbers_per_frame": str(cfg.frames.absorbers_per_frame), "seed": cfg.frames.seed,
                      "replicate": list(cfg.lattice.replicate)}}


def run(cfg, shard=None, log=None, potentials_only=False):
    """Compute the spectra of the selected frames (of one shard) and write them to the cache.

    Parameters
    ----------
    cfg : MDExafsConfig or str or Path
        The configuration (a file is loaded with ``load_config``).
    shard : feffgpu.parallel.Shard or str, optional
        The part of the frames to compute (``"3/16"``); default all.
    log : callable, optional
        Receives progress lines (default: print with flush).
    potentials_only : bool
        Compute (or load) the phase files of every edge for the whole selection and stop.

    Returns
    -------
    dict
        The summary also written to ``report.json`` (``report.shard-<i>-of-<n>.json`` for a shard).
    """
    log = log or (lambda s: print(s, flush=True))
    if not isinstance(cfg, MDExafsConfig):
        cfg = load_config(cfg)
    shard = shard if isinstance(shard, Shard) else Shard.parse(shard)
    t_start = time.time()
    out = cfg.output_path
    out.mkdir(parents=True, exist_ok=True)
    text = write_resolved_config(cfg, out)
    device = primary_device(cfg.run.devices)
    reader = make_reader(cfg)
    selection = reader.selection(cfg.frames.start, cfg.frames.stop, cfg.frames.step)
    mine = shard.select(selection)
    first = reader.read_frame(selection[0])
    sym0 = first.symbols
    sym_calc, _p, _c = _replicated(sym0, first.positions, first.cell, cfg.lattice.replicate)
    elements = sorted(set(str(s) for s in sym0))
    candidates = {}
    for e in cfg.edge:
        cand = np.nonzero(sym0 == e.element)[0]
        if len(cand) == 0:
            raise ValueError(f"edge {e.name}: the trajectory has no {e.element} atom (elements: {elements})")
        candidates[e.name] = cand
    counts = ", ".join(f"{el} {int((sym0 == el).sum())}" for el in elements)
    log(f"MD-EXAFS {cfg.system.name}: {reader.path.name} ({reader.format}), {reader.count()} frames in the file, {len(selection)} "
        f"selected, {len(mine)} in shard {shard}; {len(sym0)} atoms ({counts}); device {device}")
    providers = {e.name: PhaseProvider(e, out, selection, sym_calc, reader, device=device, log=log) for e in cfg.edge}
    if potentials_only:
        keys = {e.name: providers[e.name].prepare() for e in cfg.edge}
        summary = {"command": "run --potentials-only", "output": str(out), "phase_files": keys, "seconds": round(time.time() - t_start, 2)}
        _atomic_text(out / "report.potentials.json", json.dumps(summary, indent=1))
        return summary
    job = FrameJob(cfg, providers, candidates, elements, log=log)
    first_mine = mine[0] if mine else selection[0]
    descriptions = {e.name: _edge_description(e, job.calculator(e, first_mine, device)[1], cfg) for e in cfg.edge}
    for name, d in descriptions.items():
        log(f"[{name}] {len(d['types'].names)} path types: {', '.join(d['types'].names)}")
    path = out / cache_name(shard)
    writer = CacheWriter(path, run_settings(cfg, reader.path.stat().st_size), text, selection, shard, descriptions)
    done = writer.done
    todo = [f for f in mine if f not in done]
    if done:
        log(f"continuing {path.name}: {len(done)} frames already there, {len(todo)} to compute")
    block = int(cfg.run.frames_per_block)
    computed, seconds, paths, short = 0, [], {}, 0
    t_loop = time.time()
    try:
        frames = reader.iter_frames(todo)
        while True:
            chunk = []
            for fr in frames:
                chunk.append(fr)
                if len(chunk) == block:
                    break
            if not chunk:
                break
            results = run_items(job, chunk, devices=device, quiet=True)
            rows = []
            for res in results:
                edges = {}
                for e in cfg.edge:
                    key = res["keys"][e.name]
                    calc = job.calcs[e.name][1]
                    group = writer.phase_group(e.name, key, calc.tables)
                    edges[e.name] = (res["edges"][e.name], group)
                    for nleg, count in res["edges"][e.name].n_paths.items():
                        paths[int(nleg)] = paths.get(int(nleg), 0) + int(count)
                    short += res["edges"][e.name].n_short_pairs
                rows.append({"frame_index": res["frame_index"], "seconds": res["seconds"], "edges": edges})
                seconds.append(res["seconds"])
            writer.write_block(rows)
            computed += len(rows)
            per = (time.time() - t_loop) / computed
            log(f"  {computed + len(done)}/{len(mine)} frames (frame {rows[-1]['frame_index']}): {per:.2f} s per frame, about "
                f"{per * (len(todo) - computed) / 60:.1f} min left")
    finally:
        n_total = int(writer.f.attrs["n_rows"]) if writer.f is not None else None
        writer.close()
    if short:
        log(f"warning: {short} absorber-neighbor pairs were closer than pair_rmin and were placed on the table's first point; "
            "lower [[edge]] pair_rmin")
    summary = {"command": "run", "system": cfg.system.name, "output": str(out), "cache": str(path), "shard": str(shard),
               "frames_selected": len(selection), "frames_in_shard": len(mine), "frames_computed_now": computed, "frames_in_cache": n_total,
               "seconds": round(time.time() - t_start, 2), "seconds_per_frame": float(np.mean(seconds)) if seconds else None,
               "edges": {e.name: {"element": e.element, "edge": e.edge, "potentials": e.potentials,
                                  "path_types": descriptions[e.name]["types"].names, "absorber_atoms": int(len(candidates[e.name])),
                                  "absorbers_per_frame": str(cfg.frames.absorbers_per_frame)} for e in cfg.edge},
               "mean_paths_per_frame": {str(n): paths.get(n, 0) / max(computed * len(cfg.edge), 1) for n in sorted(set(paths) | {3, 4})},
               "short_pairs": int(short), "device": device}
    name = "report.json" if shard.count == 1 else f"report.shard-{shard.index}-of-{shard.count}.json"
    _atomic_text(out / name, json.dumps(summary, indent=1))
    log(f"done: {computed} frames computed now, {n_total} in {path.name} ({summary['seconds']:.0f} s)")
    return summary
