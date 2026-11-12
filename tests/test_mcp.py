"""The agent interface: the workspace boundary, jobs (start, queue, follow, cancel), the tools as plain methods, and the
MCP server built from them."""
import asyncio
import json
import time
from pathlib import Path
import numpy as np
import pytest

from feffgpu.mcp.jobs import JobManager, Workspace, WorkspaceError
from feffgpu.mcp.server import TOOL_NAMES, Tools

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_keeps_paths_inside(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.resolve("a/b.txt") == tmp_path.resolve() / "a/b.txt"
    for bad in ("../x", "/etc/passwd", "a/../../x"):
        with pytest.raises(WorkspaceError):
            ws.resolve(bad)
    (tmp_path / "link").symlink_to("/etc")
    with pytest.raises(WorkspaceError):                       # a symbolic link that leaves the workspace
        ws.resolve("link/passwd")
    with pytest.raises(WorkspaceError):
        ws.resolve("missing.txt", must_exist=True)


def test_jobs_run_queue_and_refuse_other_commands(tmp_path):
    jm = JobManager(Workspace(tmp_path))
    with pytest.raises(WorkspaceError):
        jm.start(["fetch-atoms"])                             # not in the list of commands a server may start
    with pytest.raises(WorkspaceError):
        jm.start(["rmc", "run-anything"])
    a = jm.start(["doctor", "--json"], device="cpu", label="check")
    b = jm.start(["doctor", "--json"], device="cpu")
    assert a["state"] in ("running", "done", "failed") and b["state"] in ("queued", "running", "done", "failed")
    sa, sb = jm.wait(a["id"], timeout=120), jm.wait(b["id"], timeout=120)
    assert sa["state"] in ("done", "failed") and sb["state"] in ("done", "failed")      # doctor exits 1 on a machine without a GPU or table
    log = "\n".join(jm.log(a["id"], lines=400)["lines"])
    assert '"feffgpu"' in log and json.loads(log)["python"]
    assert [j["id"] for j in jm.list()] == [a["id"], b["id"]]
    with pytest.raises(WorkspaceError):
        jm.status("../../etc")
    assert JobManager(Workspace(tmp_path)).status(a["id"])["state"] == sa["state"]         # the records are on disk


def test_a_running_job_can_be_canceled(tmp_path):
    jm = JobManager(Workspace(tmp_path))
    inp = tmp_path / "feff.inp"
    inp.write_text((ROOT / "tests/data/W/feff.inp").read_text())
    job = jm.start(["run", str(inp), "--out", str(tmp_path / "chi.dat"), "--device", "cpu"], device="cpu", outputs=[tmp_path / "chi.dat"])
    t0 = time.time()
    while jm.status(job["id"])["state"] == "queued" and time.time() - t0 < 30:
        time.sleep(0.1)
    st = jm.cancel(job["id"])
    assert st["state"] in ("canceled", "failed", "done")
    assert jm.result(job["id"])["outputs"][0]["kind"] in ("missing", "file")


def _poscar(path):
    from feffgpu.io import bcc_block
    sites, box = bcc_block(3, 3.0)
    lines = ["bcc block", "1.0", f"{box} 0 0", f"0 {box} 0", f"0 0 {box}", "W", str(len(sites)), "Cartesian"]
    lines += [" ".join(f"{v:.6f}" for v in p) for p in sites]
    path.write_text("\n".join(lines) + "\n")


def test_tools_as_plain_methods(tmp_path):
    t = Tools(tmp_path)
    info = t.info()
    assert info["feffgpu"] and "gpus" in info and info["workspace"] == str(tmp_path.resolve())
    _poscar(tmp_path / "POSCAR")
    tab = t.shell_table("POSCAR", [["W", "W", 2.0, 2.8], ["W", "W", 2.8, 3.2]])
    assert tab["shells"][0]["N"] == pytest.approx(8.0) and tab["shells"][1]["N"] == pytest.approx(6.0)
    assert t.short_pairs("POSCAR", {"W-W": 2.7})["pairs"]["W-W"]["per_center"] == pytest.approx(8.0)
    assert "outside the workspace" in t.shell_table("/etc/hostname", [])["error"]
    k = np.arange(0, 18.0001, 0.05)
    chi = np.sin(5 * k + 0.3) * np.exp(-0.006 * k ** 2) / np.maximum(k, 0.5)
    np.savetxt(tmp_path / "model.chi", np.c_[k, chi], header="k chi")
    np.savetxt(tmp_path / "data.chi", np.c_[k, 0.8 * chi], header="k chi")
    plain = t.compare_spectrum("model.chi", "data.chi", kmin=3, kmax=14, rmin=1, rmax=4)
    assert plain["r_factor"] == pytest.approx(0.0625, rel=1e-3)                          # (1 - 0.8)^2 / 0.8^2
    fitted = t.compare_spectrum("model.chi", "data.chi", kmin=3, kmax=14, rmin=1, rmax=4, fit_s02=True)
    assert fitted["s02"] == pytest.approx(0.8, rel=1e-3) and fitted["r_factor"] < 1e-6 and "standard deviations" in fitted["uncertainty"]
    assert "window" in t.compare_spectrum("model.chi", "data.chi", kmin=3, kmax=30)["error"]
    assert "not a job identifier" in t.job_status("nope")["error"] or "no job" in t.job_status("nope")["error"]
    assert "kind must be one of" in t.config_schema("xyz")["error"]
    docs = t.list_docs()
    assert any(p["name"] == "index" for p in docs["pages"])
    assert t.read_doc("index", max_lines=5)["text"] and "error" in t.read_doc("../README")
    assert t.search_docs("free-atom table")["hits"]


def test_the_mcp_server_lists_and_calls_the_tools(tmp_path):
    pytest.importorskip("mcp")
    from feffgpu.mcp.server import build_server
    server, _tools = build_server(tmp_path)
    listed = asyncio.run(server.list_tools())
    names = sorted(tool.name for tool in listed)
    assert names == sorted(TOOL_NAMES)
    assert all(tool.description for tool in listed)
    out = asyncio.run(server.call_tool("job_list", {}))
    assert "jobs" in json.dumps(out, default=lambda o: getattr(o, "text", str(o)))


def test_command_start_checks_every_path_argument(tmp_path):
    t = Tools(tmp_path)
    for bad in (["/etc"], ["--out=/etc/x"], [".."], ["sub/../../x"]):
        assert "outside the workspace" in t.command_start("md", "merge", bad)["error"], bad
    assert "not a command" in t.command_start("md", "anything", [])["error"]
    assert "not a command" in t.command_start("rm", "-rf", [])["error"]
