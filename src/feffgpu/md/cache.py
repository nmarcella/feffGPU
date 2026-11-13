"""The cache of an MD-EXAFS run: one HDF5 file with the spectrum of every frame, its parts, and what is needed to
select other parts later without a new forward calculation.

A run writes ``chi.h5`` in its output folder; a shard ``i`` of ``n`` writes ``chi.shard-i-of-n.h5``, and
``merge_shards`` combines the shard files into ``chi.h5``. Rows are appended in the order in which frames are
computed and carry the frame's position in the trajectory (``frame_index``); every reader sorts rows by that index,
so a merged file and a single run give the same averages. A run that stops can continue: rows already in the file
are kept (``n_rows`` is updated after each completed block of frames) and their frames are skipped.

Every dataset and attribute is described in ``docs/md/outputs.md``.
"""
from __future__ import annotations
import hashlib
import json
import platform
import re
from pathlib import Path
import numpy as np

__all__ = ["FORMAT", "FORMAT_VERSION", "cache_name", "shard_files", "run_settings", "settings_key", "CacheWriter", "merge_shards",
           "read_rows", "edge_names", "open_cache"]

FORMAT = "feffgpu md-exafs cache"
FORMAT_VERSION = 1
_ROW_DATASETS = ("frame_index", "phase_group", "n_absorbers", "n_paths", "n_short_pairs", "seconds", "chi", "chi_ss", "chi_ms",
                 "chi_types", "ss_histogram", "ms_binned")


def _h5py():
    import h5py
    return h5py


def cache_name(shard):
    """``chi.h5`` for a whole run (shard 1/1), else ``chi.shard-<i>-of-<n>.h5``."""
    return "chi.h5" if shard.count == 1 else f"chi.shard-{shard.index}-of-{shard.count}.h5"


def shard_files(folder):
    """``{count: {index: path}}`` of the shard files in an output folder."""
    out = {}
    for p in sorted(Path(folder).glob("chi.shard-*-of-*.h5")):
        m = re.fullmatch(r"chi\.shard-(\d+)-of-(\d+)\.h5", p.name)
        if m:
            out.setdefault(int(m.group(2)), {})[int(m.group(1))] = p
    return out


def run_settings(cfg, trajectory_size=None):
    """The settings that determine the spectrum of every frame (not batch sizes, averages or comparisons), as a
    dictionary. A cache is continued or merged only with the same settings."""
    edges = []
    for e in cfg.edge:
        edges.append({k: getattr(e, k) for k in ("name", "element", "edge", "potentials", "phase", "ipots", "reference_frame",
                                                 "reference_atom", "cluster_radius", "scf_radius", "cards", "potentials_every",
                                                 "rpath", "nleg", "criteria", "kmax", "pair_rmin", "pair_dr", "ms_bin_width")}
                     | {"phase_kmax": e.phase_kmax_resolved})
    lat = cfg.lattice
    return {"trajectory": str(cfg.system.trajectory), "trajectory_bytes": trajectory_size, "format": cfg.system.format,
            "reader": cfg.system.reader, "topology": cfg.system.topology, "atoms": dict(cfg.atoms),
            "lattice": {"dynamic": lat.dynamic, "a": lat.a, "b": lat.b, "c": lat.c, "alpha": lat.alpha, "beta": lat.beta,
                        "gamma": lat.gamma, "matrix": lat.matrix, "pbc": lat.pbc, "replicate": lat.replicate},
            "frames": {"start": cfg.frames.start, "stop": cfg.frames.stop, "step": cfg.frames.step,
                       "absorbers_per_frame": cfg.frames.absorbers_per_frame, "seed": cfg.frames.seed},
            "edges": edges}


def settings_key(settings):
    """A short hash of ``run_settings``."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _differences(a, b, prefix=""):
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _differences(va, vb, f"{prefix}{k}.")
        elif isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb) and all(isinstance(x, dict) for x in va + vb):
            for i, (x, y) in enumerate(zip(va, vb)):
                out += _differences(x, y, f"{prefix}{k}.{i}.")
        elif va != vb:
            out.append(f"{prefix}{k}: {va!r} -> {vb!r}")
    return out


def _versions():
    import h5py
    import torch
    from .. import __version__
    return {"feffgpu_version": str(__version__), "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "h5py_version": str(h5py.__version__), "python_version": str(platform.python_version())}


def open_cache(path, mode="r"):
    """Open a cache file with h5py, with a clear message if it is not one."""
    h5py = _h5py()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no cache file {p} (run `md-exafs run` first, or `md-exafs merge` after a sharded run)")
    try:
        f = h5py.File(p, mode)
    except OSError as err:
        raise ValueError(f"{p} cannot be opened as HDF5 ({err}); if a run was killed while writing it, delete the file and run "
                         "again") from err
    if f.attrs.get("format") != FORMAT:
        f.close()
        raise ValueError(f"{p} is not an MD-EXAFS cache")
    return f


def edge_names(f):
    """The edge names of an open cache, in the order of the configuration."""
    return list(json.loads(f.attrs["edge_names"]))


class CacheWriter:
    """Writes the frames of a run (or of one shard) into its cache file.

    Parameters
    ----------
    path : Path
        The cache file; created if missing, continued if it holds rows of the same settings.
    settings : dict
        ``run_settings`` of the run.
    config_text : str
        The resolved configuration (TOML), stored as an attribute.
    selection : list of int
        Positions of every frame of the run (all shards).
    shard : feffgpu.parallel.Shard
    edges : dict
        ``{name: description}`` with, per edge: ``element``, ``edge``, ``k`` (NK,), ``r_table`` (NR,), ``ms_bin_edges``
        (NB + 1,), ``types`` (``PathTypes``), ``attrs`` (dict of scalar settings).
    """

    def __init__(self, path, settings, config_text, selection, shard, edges):
        h5py = _h5py()
        self.path = Path(path)
        self.key = settings_key(settings)
        exists = self.path.is_file()
        if exists:
            f = open_cache(self.path, "r+")
            old = json.loads(f.attrs["settings"])
            if f.attrs["settings_key"] != self.key:
                f.close()
                diff = _differences(old, settings)
                what = "; ".join(diff[:6]) or "unknown difference"
                raise ValueError(f"{self.path.name} was made with other settings ({what}); use a new output folder, or delete the file "
                                 "to start again")
            if f.attrs.get("merged", False):
                f.close()
                raise ValueError(f"{self.path.name} is a merged file; it is not continued (run the shards, or delete it to start again)")
            self.f = f
            n = int(f.attrs["n_rows"])
            for name in edges:                                               # drop rows of a block that was not completed
                g = f["edges"][name]
                for ds in _ROW_DATASETS:
                    if g[ds].shape[0] != n:
                        g[ds].resize(n, axis=0)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            f = h5py.File(self.path, "w")
            f.attrs.update({"format": FORMAT, "format_version": FORMAT_VERSION, "settings": json.dumps(settings, default=str),
                            "settings_key": self.key, "config": config_text, "shard": str(shard), "shard_index": shard.index,
                            "shard_count": shard.count, "merged": False, "n_rows": 0, "edge_names": json.dumps(list(edges)), **_versions()})
            f.create_dataset("selection", data=np.asarray(selection, dtype=np.int64))
            ge = f.create_group("edges")
            for name, d in edges.items():
                g = ge.create_group(name)
                types = d["types"]
                NK, NR, NB = len(d["k"]), len(d["r_table"]), len(d["ms_bin_edges"]) - 1
                P, M, T = len(types.ss), len(types.ms), len(types.names)
                g.attrs.update({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in d["attrs"].items()})
                g.attrs.update({"element": d["element"], "edge": d["edge"], "path_types": json.dumps(types.names),
                                "ss_types": json.dumps(types.ss), "ms_types": json.dumps(types.ms),
                                "ss_elements": json.dumps(types.ss_elements)})
                g.create_dataset("k", data=np.asarray(d["k"], float))
                g.create_dataset("r_table", data=np.asarray(d["r_table"], float))
                g.create_dataset("ms_bin_edges", data=np.asarray(d["ms_bin_edges"], float))
                g.create_dataset("phase_keys", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
                g.create_dataset("ss_tables", shape=(0, P, NR, NK), maxshape=(None, P, NR, NK), dtype="f4", chunks=(1, P, NR, NK),
                                 compression="gzip", compression_opts=4)
                z = dict(compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype="i8", chunks=(1024,))
                g.create_dataset("phase_group", shape=(0,), maxshape=(None,), dtype="i4", chunks=(1024,))
                g.create_dataset("n_absorbers", shape=(0,), maxshape=(None,), dtype="i4", chunks=(1024,))
                legs = max(2, int(d["attrs"].get("nleg", 4)) - 2)                  # one column per number of legs from three on (at least two)
                g.create_dataset("n_paths", shape=(0, legs), maxshape=(None, legs), dtype="i8", chunks=(1024, legs))
                g.create_dataset("n_short_pairs", shape=(0,), maxshape=(None,), dtype="i8", chunks=(1024,))
                g.create_dataset("seconds", shape=(0,), maxshape=(None,), dtype="f4", chunks=(1024,))
                g.create_dataset("chi", shape=(0, NK), maxshape=(None, NK), dtype="f4", chunks=(64, NK), **z)
                g.create_dataset("chi_ss", shape=(0, NK), maxshape=(None, NK), dtype="f4", chunks=(64, NK), **z)
                g.create_dataset("chi_ms", shape=(0, NK), maxshape=(None, NK), dtype="f4", chunks=(64, NK), **z)
                g.create_dataset("chi_types", shape=(0, T, NK), maxshape=(None, T, NK), dtype="f4", chunks=(8, T, NK), **z)
                g.create_dataset("ss_histogram", shape=(0, P, NR), maxshape=(None, P, NR), dtype="f4", chunks=(8, P, NR), **z)
                g.create_dataset("ms_binned", shape=(0, M, NB, NK), maxshape=(None, M if M else None, NB, NK), dtype="f4",
                                 chunks=(1, max(M, 1), NB, NK), **z)              # an empty type axis (nleg 2) must be unlimited for HDF5
            f.flush()
            self.f = f
        self.edges = list(edges)

    @property
    def done(self):
        """Positions of the frames already in the file."""
        n = int(self.f.attrs["n_rows"])
        return set(int(i) for i in self.f["edges"][self.edges[0]]["frame_index"][:n])

    def phase_group(self, edge, key, tables):
        """The index of the phase file ``key`` of an edge in its ``phase_keys``; its tables (P, NR, NK) are stored the
        first time it is seen."""
        g = self.f["edges"][edge]
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in g["phase_keys"][:]]
        if key in keys:
            return keys.index(key)
        n = len(keys)
        g["phase_keys"].resize(n + 1, axis=0)
        g["phase_keys"][n] = key
        g["ss_tables"].resize(n + 1, axis=0)
        g["ss_tables"][n] = np.asarray(tables, np.float32)
        return n

    def write_block(self, rows):
        """Append a completed block of frames.

        Parameters
        ----------
        rows : list of dict
            One per frame: ``{"frame_index": int, "seconds": float, "edges": {name: (FrameSpectrum, phase group)}}``.
        """
        if not rows:
            return
        n0 = int(self.f.attrs["n_rows"])
        n1 = n0 + len(rows)
        for name in self.edges:
            g = self.f["edges"][name]
            for ds in _ROW_DATASETS:
                g[ds].resize(n1, axis=0)
            spectra = [r["edges"][name][0] for r in rows]
            g["frame_index"][n0:n1] = [r["frame_index"] for r in rows]
            g["phase_group"][n0:n1] = [r["edges"][name][1] for r in rows]
            g["seconds"][n0:n1] = [r["seconds"] for r in rows]
            g["n_absorbers"][n0:n1] = [s.n_absorbers for s in spectra]
            g["n_paths"][n0:n1] = [[s.n_paths.get(3 + j, 0) for j in range(g["n_paths"].shape[1])] for s in spectra]
            g["n_short_pairs"][n0:n1] = [s.n_short_pairs for s in spectra]
            g["chi"][n0:n1] = np.stack([s.chi for s in spectra]).astype(np.float32)
            g["chi_ss"][n0:n1] = np.stack([s.chi_ss for s in spectra]).astype(np.float32)
            g["chi_ms"][n0:n1] = np.stack([s.chi_ms for s in spectra]).astype(np.float32)
            g["chi_types"][n0:n1] = np.stack([s.chi_types for s in spectra]).astype(np.float32)
            g["ss_histogram"][n0:n1] = np.stack([s.ss_histogram for s in spectra]).astype(np.float32)
            g["ms_binned"][n0:n1] = np.stack([s.ms_binned for s in spectra]).astype(np.float32)
        self.f.attrs["n_rows"] = n1
        self.f.flush()

    def close(self):
        """Close the file."""
        if getattr(self, "f", None) is not None:
            self.f.close()
            self.f = None


def read_rows(g, n_rows=None):
    """Row order of an edge group sorted by frame index: ``(order, frame_index)`` (numpy)."""
    n = g["frame_index"].shape[0] if n_rows is None else int(n_rows)
    fi = g["frame_index"][:n]
    order = np.argsort(fi, kind="stable")
    return order, fi[order]


def merge_shards(folder, force=False, log=print):
    """Combine the shard files of an output folder into ``chi.h5``.

    The shard files must all come from the same settings and the same number of shards, every shard must be present,
    and together they must hold every frame of the selection exactly once. Rows are written in order of frame index.

    Parameters
    ----------
    folder : str or Path
        The run's output folder.
    force : bool
        Overwrite an existing ``chi.h5`` that is not itself a merged file.
    log : callable

    Returns
    -------
    dict
        ``output``, ``shards``, ``frames``, ``edges``.
    """
    h5py = _h5py()
    folder = Path(folder)
    found = shard_files(folder)
    if not found:
        raise FileNotFoundError(f"no shard files (chi.shard-<i>-of-<n>.h5) in {folder}")
    if len(found) > 1:
        raise ValueError(f"{folder} holds shard files of different shard counts {sorted(found)}; keep one set")
    count, files = next(iter(found.items()))
    missing = [i for i in range(1, count + 1) if i not in files]
    if missing:
        raise ValueError(f"shards {missing} of {count} are missing in {folder}; run them first")
    target = folder / "chi.h5"
    if target.exists():
        with h5py.File(target, "r") as t:
            merged_before = bool(t.attrs.get("merged", False))
        if not merged_before and not force:
            raise ValueError(f"{target} exists and is not a merged file (a single run?); move it away or use --force")
    srcs = [open_cache(files[i], "r") for i in range(1, count + 1)]
    try:
        keys = {s.attrs["settings_key"] for s in srcs}
        if len(keys) != 1:
            raise ValueError("the shard files were made with different settings; they cannot be merged")
        selection = srcs[0]["selection"][:]
        names = edge_names(srcs[0])
        frames, owner = [], []
        for j, s in enumerate(srcs):
            n = int(s.attrs["n_rows"])
            fi = s["edges"][names[0]]["frame_index"][:n]
            frames.append(fi)
            owner.append(np.full(n, j))
        allf = np.concatenate(frames)
        if len(np.unique(allf)) != len(allf):
            raise ValueError("a frame appears in more than one shard file")
        expected = set(int(x) for x in selection)
        have = set(int(x) for x in allf)
        if have != expected:
            lack = sorted(expected - have)
            raise ValueError(f"the shards hold {len(have)} of the {len(expected)} frames; missing for example {lack[:10]} (finish the "
                             "shards first)")
        order = np.argsort(allf, kind="stable")
        src_of = np.concatenate(owner)[order]
        row_of = np.concatenate([np.arange(len(fr)) for fr in frames])[order]
        tmp = folder / "chi.h5.merging"
        with h5py.File(tmp, "w") as out:
            for k, v in srcs[0].attrs.items():
                out.attrs[k] = v
            out.attrs.update({"merged": True, "shard": f"merged from {count} shards", "shard_index": 1, "shard_count": 1,
                              "n_rows": len(allf), **_versions()})
            out.create_dataset("selection", data=selection)
            ge = out.create_group("edges")
            for name in names:
                g0 = srcs[0]["edges"][name]
                g = ge.create_group(name)
                for k, v in g0.attrs.items():
                    g.attrs[k] = v
                for ds in ("k", "r_table", "ms_bin_edges"):
                    g.create_dataset(ds, data=g0[ds][:])
                # phase files: the union of every shard's keys; the first shard that has a key provides its tables
                keys_out, tables_out, remap = [], [], []
                for s in srcs:
                    gs = s["edges"][name]
                    ks = [k.decode() if isinstance(k, bytes) else str(k) for k in gs["phase_keys"][:]]
                    m = []
                    for i, k in enumerate(ks):
                        if k not in keys_out:
                            keys_out.append(k)
                            tables_out.append(gs["ss_tables"][i])
                        else:
                            ref = tables_out[keys_out.index(k)]
                            scale = float(np.abs(ref).max()) or 1.0
                            if float(np.abs(gs["ss_tables"][i] - ref).max()) > 1e-4 * scale:
                                log(f"[{name}] note: the tables of phase file {k} differ between shards by more than 1e-4 (the phase "
                                    "file was computed in more than one shard); the first is kept")
                        m.append(keys_out.index(k))
                    remap.append(np.asarray(m, dtype=np.int32))
                g.create_dataset("phase_keys", data=np.asarray(keys_out, dtype=object), dtype=h5py.string_dtype(), maxshape=(None,))
                P, NR, NK = g0["ss_tables"].shape[1:]
                tables = np.stack(tables_out).astype(np.float32) if tables_out else np.zeros((0, P, NR, NK), np.float32)
                g.create_dataset("ss_tables", data=tables, maxshape=(None, P, NR, NK), chunks=(1, P, NR, NK), compression="gzip",
                                 compression_opts=4)
                for ds in _ROW_DATASETS:
                    d0 = g0[ds]
                    shape = (len(allf),) + d0.shape[1:]
                    kw = {"compression": "gzip", "compression_opts": 4, "shuffle": True} if d0.compression else {}
                    dout = g.create_dataset(ds, shape=shape, maxshape=d0.maxshape, dtype=d0.dtype, chunks=d0.chunks, **kw)
                    for j, s in enumerate(srcs):                                   # copy shard by shard, rows to their sorted places
                        sel = np.nonzero(src_of == j)[0]                           # target rows (increasing)
                        src_rows = row_of[sel]
                        src = s["edges"][name][ds]
                        for c in range(0, len(sel), 256):                          # bounded memory for long runs
                            t, r = sel[c:c + 256], src_rows[c:c + 256]
                            o = np.argsort(r, kind="stable")
                            data = np.empty((len(r),) + src.shape[1:], dtype=src.dtype)
                            data[o] = src[r[o]]
                            if ds == "phase_group":
                                data = remap[j][data]
                            dout[t] = data
        tmp.replace(target)
    finally:
        for s in srcs:
            s.close()
    log(f"merged {count} shards ({len(allf)} frames) into {target}")
    return {"output": str(target), "shards": count, "frames": int(len(allf)), "edges": names}
