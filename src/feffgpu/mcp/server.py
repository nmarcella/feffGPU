"""The MCP server of feffgpu: tools through which an agent validates a configuration, starts a run, follows it and
reads the results.

Start it with ``feffgpu-mcp --workspace /path/to/project`` (standard input/output transport, which is what desktop
and command-line MCP clients use) or ``feffgpu-mcp --workspace ... --transport streamable-http`` on a cluster login
node. MCP is the Model Context Protocol, the interface through which agents call tools.

Rules of the server:

- It reads and writes only inside its workspace directory.
- It cannot run shell commands. It starts only ``feffgpu`` commands from a fixed list.
- Long runs are jobs: a start tool returns an identifier at once; ``job_status``, ``job_log`` and ``job_result``
  follow it. Only one job runs on a GPU at a time; later jobs wait in a queue.
- Results come back as short summaries and file paths, never as large arrays.

The tool functions are plain methods of ``Tools`` and can be called from Python without any MCP client.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

from .jobs import JobManager, Workspace, WorkspaceError

KINDS = ("md", "rmc")
DOCS_URL = "https://feffgpu.readthedocs.io"
INSTRUCTIONS = (
    "feffgpu calculates EXAFS spectra with FEFF 8.5L's theory on NVIDIA GPUs and has two tools built on it: MD-EXAFS (the "
    "spectrum of a molecular dynamics trajectory) and RMC-EXAFS (reverse Monte Carlo fitting of a structure to measured "
    "spectra). Typical order: `info` (is the installation complete?), `list_docs` / `read_doc` (how a task is done), "
    "`example_config` and `config_schema` (write a TOML file into the workspace), `validate_config`, then `md_exafs_start` or "
    "`rmc_start`, then `job_status` / `job_log` until the job is done, then `job_result` and the analysis tools. "
    "All paths are inside the workspace. Runs are jobs and can take hours: poll, do not wait in one call."
)


def _error(message):
    return {"error": str(message)}


def _config_module(kind):
    if kind not in KINDS:
        raise WorkspaceError(f"kind must be one of {KINDS} (md = MD-EXAFS, rmc = RMC-EXAFS), got {kind!r}")
    return importlib.import_module(f"feffgpu.{kind}.config")


def _docs_dir():
    """The documentation sources: ``$FEFFGPU_DOCS``, or ``docs/`` of a source checkout (an editable install)."""
    env = os.environ.get("FEFFGPU_DOCS")
    cands = [Path(env)] if env else []
    cands.append(Path(__file__).resolve().parents[3] / "docs")
    for c in cands:
        if (c / "index.md").exists():
            return c
    return None


class Tools:
    """The tools of the server as plain methods. Every method returns a JSON-serializable dict; a user error comes
    back as ``{"error": message}`` instead of an exception.

    Parameters
    ----------
    workspace : str or Path
        The only directory the tools read and write.
    """

    def __init__(self, workspace):
        self.ws = Workspace(workspace)
        self.jobs = JobManager(self.ws)

    # -- installation -------------------------------------------------------------------------------------------
    def info(self):
        """Versions, GPUs with their free memory, the free-atom table, optional parts, and a list of problems."""
        from ..cli import environment_report
        rep = environment_report()
        rep["workspace"] = str(self.ws.root)
        rep["documentation"] = DOCS_URL
        return rep

    # -- configurations -------------------------------------------------------------------------------------------
    def config_schema(self, kind):
        """The JSON Schema of the configuration file of MD-EXAFS (kind "md") or RMC-EXAFS (kind "rmc"): every key
        with its type, default and description."""
        try:
            return {"kind": kind, "schema": _config_module(kind).json_schema()}
        except (WorkspaceError, ImportError) as err:
            return _error(err)

    def example_config(self, kind, name=""):
        """An example configuration file as TOML text, to copy and edit. Without ``name``: the list of examples."""
        try:
            mod = _config_module(kind)
            examples = mod.example_configs()
        except (WorkspaceError, ImportError) as err:
            return _error(err)
        if not name:
            return {"kind": kind, "examples": sorted(examples)}
        if name not in examples:
            return _error(f"no example {name!r} for {kind}; available: {sorted(examples)}")
        return {"kind": kind, "name": name, "toml": examples[name]}

    def validate_config(self, kind, path):
        """Check a configuration file without running it. Returns ``valid`` and, if not, one message per problem
        with the path of the field (for example ``edge.0.window.kmax``)."""
        try:
            p = self.ws.resolve(path, must_exist=True)
            problems = _config_module(kind).validate_file(p)
        except (WorkspaceError, ImportError) as err:
            return _error(err)
        return {"kind": kind, "path": self.ws.relative(p), "valid": not problems, "problems": problems}

    # -- runs (jobs) ----------------------------------------------------------------------------------------------
    def feff_run(self, feff_inp, out="chi_gpu.dat", device="cuda"):
        """Start a FEFF calculation: a ``feff.inp`` to chi(k) (S0^2 = 1, no Debye-Waller factors). Returns a job."""
        try:
            inp, outp = self.ws.resolve(feff_inp, must_exist=True), self.ws.resolve(out)
            return self.jobs.start(["run", str(inp), "--out", str(outp), "--device", "cuda" if device.startswith("cuda") else device, "--json"],
                                   device=device, label=f"FEFF {self.ws.relative(inp)}", outputs=[outp])
        except WorkspaceError as err:
            return _error(err)

    def phase_build(self, feff_inp, out="phase.pkl", xkmax=24.5, device="cuda"):
        """Start the calculation of a phase file (potentials and phase shifts) from a ``feff.inp`` of a representative
        cluster. MD-EXAFS and RMC-EXAFS take the phase file as input. Returns a job."""
        try:
            inp, outp = self.ws.resolve(feff_inp, must_exist=True), self.ws.resolve(out)
            return self.jobs.start(["phase", str(inp), str(outp), "--xkmax", str(float(xkmax)), "--device", "cuda" if device.startswith("cuda") else device,
                                    "--json"], device=device, label=f"phase file from {self.ws.relative(inp)}", outputs=[outp])
        except WorkspaceError as err:
            return _error(err)

    def _start_config_run(self, kind, config, extra, device, label):
        try:
            p = self.ws.resolve(config, must_exist=True)
            problems = _config_module(kind).validate_file(p)
            if problems:
                return {"error": "the configuration is not valid; nothing was started", "problems": problems}
            out = _config_module(kind).output_directory(p)
            outp = self.ws.resolve(out)
            return self.jobs.start([kind, "run", str(p), *extra], device=device, label=f"{label} {self.ws.relative(p)}", outputs=[outp], cwd=p.parent)
        except (WorkspaceError, ImportError) as err:
            return _error(err)

    def md_exafs_start(self, config, shard="", device="cuda"):
        """Start MD-EXAFS from a TOML configuration: the spectrum of a trajectory averaged over frames and absorbing
        atoms. ``shard`` ("3/16") computes one part of the frames, for several GPUs; merge the parts afterwards.
        Returns a job."""
        return self._start_config_run("md", config, ["--shard", shard] if shard else [], device, "MD-EXAFS")

    def rmc_start(self, config, device="cuda"):
        """Start RMC-EXAFS from a TOML configuration: a population of structures is moved until their spectra fit the
        measured ones. Returns a job."""
        return self._start_config_run("rmc", config, [], device, "RMC-EXAFS")

    def command_start(self, tool, command, arguments, device="cuda"):
        """Start another ``md-exafs`` or ``rmc-exafs`` subcommand as a job (for example tool "md", command "merge",
        arguments ["mdexafs_out"]). Only the documented subcommands are allowed; every argument that names an
        existing file or directory must be inside the workspace."""
        try:
            args = [str(a) for a in arguments]
            for a in args:
                value = a.split("=", 1)[1] if a.startswith("-") and "=" in a else ("" if a.startswith("-") else a)
                if value and ("/" in value or value in (".", "..") or (self.ws.root / value).exists()):
                    self.ws.resolve(value)
            return self.jobs.start([tool, command, *args], device=device, label=f"{tool} {command}")
        except WorkspaceError as err:
            return _error(err)

    def job_status(self, job_id):
        """The state of a job (queued, running, done, failed, canceled), its run time and its last output line."""
        try:
            return self.jobs.status(job_id)
        except WorkspaceError as err:
            return _error(err)

    def job_list(self):
        """All jobs of the workspace, oldest first."""
        return {"jobs": self.jobs.list()}

    def job_log(self, job_id, lines=40):
        """The last lines of a job's output (at most 400)."""
        try:
            return self.jobs.log(job_id, lines)
        except WorkspaceError as err:
            return _error(err)

    def job_cancel(self, job_id):
        """Stop a queued or running job."""
        try:
            return self.jobs.cancel(job_id)
        except WorkspaceError as err:
            return _error(err)

    def job_result(self, job_id):
        """The state of a job and the files it wrote (paths and sizes). Read numbers from ``report.json`` and the
        other documented output files; arrays are not returned."""
        try:
            return self.jobs.result(job_id)
        except WorkspaceError as err:
            return _error(err)

    # -- quick analysis (no job: seconds on the CPU) ---------------------------------------------------------------
    def _structure(self, path):
        from ..io import read_structure
        return read_structure(self.ws.resolve(path, must_exist=True))

    def shell_table(self, structure, shells):
        """Coordination number N, mean distance R, MSRD (variance) and third cumulant of pair shells of a structure
        file, by integration of the pair histogram (nothing fitted). ``shells``: a list of
        [center element, neighbor element, r_min, r_max] in A. The structure is any file ASE reads (POSCAR, CIF,
        extended xyz). For the states of an RMC run or the frames of a trajectory use `rmc-exafs shells` or
        `md-exafs rdf`."""
        try:
            from ..analysis import shell_table
            sym, pos, cell = self._structure(structure)
            rows = shell_table(pos, sym, cell, [(str(a), str(b), float(lo), float(hi)) for a, b, lo, hi in shells])
            return {"structure": structure, "atoms": int(len(sym)), "shells": rows}
        except (WorkspaceError, ValueError, OSError) as err:
            return _error(err)

    def short_pairs(self, structure, thresholds):
        """Pairs closer than a threshold: per pair kind the number per center atom, the share of centers that have
        one, and the closest distance. ``thresholds``: {"A-B": distance in A}."""
        try:
            from ..analysis import short_pairs
            sym, pos, cell = self._structure(structure)
            return {"structure": structure, "pairs": short_pairs(pos, sym, cell, {str(k): float(v) for k, v in thresholds.items()})}
        except (WorkspaceError, ValueError, OSError) as err:
            return _error(err)

    def compare_spectrum(self, model, data, kmin=3.0, kmax=14.0, dk=1.0, kweight=2, rmin=1.0, rmax=6.0, fit_s02=False, fit_e0=False,
                         model_s02=1.0):
        """The R-factor between a calculated chi(k) file and a measured one (two columns: k, chi), in xraylarch's
        conventions. With ``fit_s02`` / ``fit_e0`` the amplitude factor S0^2 and the energy shift are fitted on the
        model and returned with 90 % intervals."""
        try:
            import numpy as np
            import torch
            from ..exafs import FitWindow, SpectrumComparison, check_window_inside_data, fit_shared, k_grid, read_chi, shared_intervals, to_grid
            km, cm = read_chi(self.ws.resolve(model, must_exist=True))
            kd, cd = read_chi(self.ws.resolve(data, must_exist=True))
            win = FitWindow(kmin=float(kmin), kmax=float(kmax), dk=float(dk), kweight=int(kweight), rmin=float(rmin), rmax=float(rmax))
            check_window_inside_data(kd, win.kmin, win.kmax, win.dk, "the measured spectrum")
            check_window_inside_data(km, win.kmin, win.kmax, win.dk, "the calculated spectrum")
            k = k_grid(float(np.ceil(max(kd[-1], km[-1]) / 0.05) * 0.05))
            cmp = SpectrumComparison("spectrum", k, to_grid(k, kd, cd), win, s02=float(model_s02), fit_s02=bool(fit_s02), fit_e0=bool(fit_e0), device="cpu")
            chi = torch.tensor(to_grid(k, km, cm), dtype=torch.float32)
            out = {"model": model, "data": data, "window": win.as_dict(), "r_factor_before_fit": cmp.r_factor(chi)}
            if fit_s02 or fit_e0:
                groups = [[(cmp, chi, 1.0)]]
                fit = fit_shared(groups)
                iv = shared_intervals(groups)
                out.update(r_factor=fit["ratios"][0][0], s02=cmp.s02, de0_eV=cmp.de0, at_bound=fit["at_bound"],
                           s02_interval_90=[iv["s02_lo"].get("spectrum"), iv["s02_hi"].get("spectrum")] if fit_s02 else None,
                           de0_interval_90=[iv["de0_lo"].get("spectrum"), iv["de0_hi"].get("spectrum")] if fit_e0 else None,
                           uncertainty=iv["method"])
            else:
                out["r_factor"] = out["r_factor_before_fit"]
            return out
        except (WorkspaceError, ValueError, OSError) as err:
            return _error(err)

    # -- documentation --------------------------------------------------------------------------------------------
    def list_docs(self):
        """The documentation pages that can be read with ``read_doc`` (name and title)."""
        d = _docs_dir()
        if d is None:
            return {"pages": [], "note": f"the documentation sources are not installed with the package; read {DOCS_URL} "
                    f"(plain text for agents: {DOCS_URL}/llms.txt), or set FEFFGPU_DOCS to the docs folder of a source checkout"}
        pages = []
        for f in sorted(d.rglob("*.md")):
            if "_build" in f.parts:
                continue
            title = next((ln.lstrip("# ").strip() for ln in f.read_text(errors="replace").splitlines() if ln.startswith("# ")), f.stem)
            pages.append({"name": str(f.relative_to(d).with_suffix("")), "title": title})
        return {"pages": pages}

    def read_doc(self, name, start_line=1, max_lines=400):
        """One documentation page as Markdown text (``name`` as given by ``list_docs``, for example
        "md/quickstart"). Long pages are returned in parts: ``start_line`` and ``max_lines``."""
        d = _docs_dir()
        if d is None:
            return _error(f"the documentation sources are not installed; read {DOCS_URL}")
        f = (d / (str(name) + ".md")).resolve()
        if d.resolve() not in f.parents or not f.exists():
            return _error(f"no documentation page {name!r}; see list_docs")
        lines = f.read_text(errors="replace").splitlines()
        a = max(1, int(start_line))
        part = lines[a - 1: a - 1 + max(1, min(int(max_lines), 1000))]
        return {"name": name, "lines": f"{a}-{a - 1 + len(part)} of {len(lines)}", "text": "\n".join(part)}

    def search_docs(self, query, max_results=12):
        """Search the documentation pages for words (all must occur in a line or its heading). Returns page names,
        the heading of the section and the matching line."""
        d = _docs_dir()
        if d is None:
            return _error(f"the documentation sources are not installed; read {DOCS_URL}")
        words = [w.lower() for w in re.findall(r"[\w^-]+", str(query)) if len(w) > 1]
        if not words:
            return _error("give at least one search word")
        hits = []
        for f in sorted(d.rglob("*.md")):
            if "_build" in f.parts:
                continue
            heading = ""
            for i, ln in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if ln.startswith("#"):
                    heading = ln.lstrip("# ").strip()
                low = (heading + " " + ln).lower()
                if all(w in low for w in words):
                    hits.append({"page": str(f.relative_to(d).with_suffix("")), "section": heading, "line": i, "text": ln.strip()[:240]})
                    if len(hits) >= int(max_results):
                        return {"query": query, "hits": hits, "truncated": True}
        return {"query": query, "hits": hits, "truncated": False}


TOOL_NAMES = ("info", "config_schema", "example_config", "validate_config", "feff_run", "phase_build", "md_exafs_start", "rmc_start",
              "command_start", "job_status", "job_list", "job_log", "job_cancel", "job_result", "shell_table", "short_pairs", "compare_spectrum",
              "list_docs", "read_doc", "search_docs")


def _server_class():
    """The server class of the installed MCP SDK (version 2: ``MCPServer``; version 1: ``FastMCP``)."""
    try:
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError as err:
        raise SystemExit("feffgpu: the MCP server needs the `mcp` package: pip install 'feffgpu[mcp]'") from err


def build_server(workspace, host="127.0.0.1", port=8765):
    """An MCP server with the tools of ``Tools`` bound to ``workspace``, plus the configuration schemas and the
    documentation index as resources. Returns ``(server, tools)``."""
    cls = _server_class()
    kwargs = {"instructions": INSTRUCTIONS}
    try:
        server = cls("feffgpu", host=host, port=port, **kwargs)              # SDK version 1 takes the address here
    except TypeError:
        server = cls("feffgpu", **kwargs)
    tools = Tools(workspace)
    for name in TOOL_NAMES:
        method = getattr(tools, name)
        server.tool(name=name, description=" ".join((method.__doc__ or name).split()))(method)

    @server.resource("feffgpu://schema/md", name="md-exafs configuration schema", mime_type="application/json")
    def md_schema() -> str:
        return json.dumps(tools.config_schema("md"))

    @server.resource("feffgpu://schema/rmc", name="rmc-exafs configuration schema", mime_type="application/json")
    def rmc_schema() -> str:
        return json.dumps(tools.config_schema("rmc"))

    @server.resource("feffgpu://docs/index", name="documentation pages", mime_type="application/json")
    def docs_index() -> str:
        return json.dumps(tools.list_docs())

    return server, tools


def main(argv=None):
    """Entry point of ``feffgpu-mcp`` (and ``feffgpu mcp``)."""
    ap = argparse.ArgumentParser(prog="feffgpu-mcp", description="The MCP server of feffgpu: lets an agent validate configurations, start "
                                 "FEFF, MD-EXAFS and RMC-EXAFS runs as jobs, follow them and read results. It reads and writes only inside the "
                                 "workspace and cannot run shell commands.")
    ap.add_argument("--workspace", default=".", help="the directory the server may read and write (default: the current directory)")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"],
                    help="stdio (default; for desktop and command-line clients) or streamable-http (for a remote machine)")
    ap.add_argument("--host", default="127.0.0.1", help="address for the HTTP transports (default 127.0.0.1: reachable from this machine only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--list-tools", action="store_true", help="print the tool names and descriptions as JSON and exit")
    a = ap.parse_args(argv)
    if a.list_tools:
        t = Tools(a.workspace)
        print(json.dumps([{"name": n, "description": " ".join((getattr(t, n).__doc__ or "").split())} for n in TOOL_NAMES], indent=1))
        return 0
    server, _tools = build_server(a.workspace, host=a.host, port=a.port)
    if a.transport == "stdio":
        server.run(transport="stdio")
    else:
        try:
            server.run(transport=a.transport, host=a.host, port=a.port)       # SDK version 2
        except TypeError:
            server.run(transport=a.transport)                                 # SDK version 1: the address was given to the constructor
    return 0


if __name__ == "__main__":
    sys.exit(main())
