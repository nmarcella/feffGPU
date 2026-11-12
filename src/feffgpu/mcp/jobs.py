"""Long runs as jobs: start, follow, cancel, read the result. Used by the MCP server, usable on its own.

A job is one ``feffgpu`` command run as a separate process. ``JobManager.start`` returns an identifier at once; the
run itself can take hours. Every job has a folder ``<workspace>/.feffgpu_jobs/<id>/`` with ``job.json`` (the command,
the state, times, the exit code) and ``log.txt`` (everything the command printed), so jobs survive a restart of the
server that started them.

Only one job runs on a device at a time; a job started while its device is busy waits in a queue. The commands that
can be started are a fixed list of ``feffgpu`` subcommands: there is no way to run an arbitrary shell command.
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ALLOWED = {
    ("run",), ("phase",), ("doctor",),
    ("md", "run"), ("md", "merge"), ("md", "average"), ("md", "compare"), ("md", "rdf"), ("md", "convergence"), ("md", "plot"), ("md", "validate"), ("md", "dynamics"),
    ("rmc", "run"), ("rmc", "refine"), ("rmc", "validate"), ("rmc", "shells"), ("rmc", "short-pairs"), ("rmc", "rdf"), ("rmc", "angles"),
    ("rmc", "plot"), ("rmc", "signal-map"),
}
STATES = ("queued", "running", "done", "failed", "canceled")


class WorkspaceError(ValueError):
    """A path outside the workspace, or a command that is not allowed."""


class Workspace:
    """The one directory that the server may read and write.

    Parameters
    ----------
    root : str or Path
        Every path given to a tool is resolved (symbolic links followed) and must lie inside this directory.
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"the workspace {self.root} is not a directory")

    def resolve(self, path, must_exist=False):
        """An absolute path inside the workspace for ``path`` (relative paths are relative to the workspace)."""
        p = Path(path).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()
        if p != self.root and self.root not in p.parents:
            raise WorkspaceError(f"{path} is outside the workspace {self.root}; the server only reads and writes inside its workspace")
        if must_exist and not p.exists():
            raise WorkspaceError(f"{path} does not exist (looked for {p})")
        return p

    def relative(self, path):
        """``path`` relative to the workspace, as text (for short results)."""
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, TypeError):
        return False
    try:                                                      # a finished child that was not waited for still answers the signal
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class JobManager:
    """Start and follow ``feffgpu`` commands inside a workspace, one at a time per device.

    Parameters
    ----------
    workspace : Workspace
    python : str
        Interpreter that runs the commands (default: the one that runs the server).
    """

    def __init__(self, workspace, python=None):
        self.ws = workspace
        self.dir = workspace.root / ".feffgpu_jobs"
        self.dir.mkdir(exist_ok=True)
        self.python = python or sys.executable
        self._lock = threading.RLock()
        self._procs = {}
        self._watcher = None

    # -- records ------------------------------------------------------------------------------------------------
    def _path(self, job_id):
        if not job_id or any(c not in "0123456789abcdef-" for c in job_id):
            raise WorkspaceError(f"{job_id!r} is not a job identifier")
        return self.dir / job_id

    def _read(self, job_id):
        f = self._path(job_id) / "job.json"
        if not f.exists():
            raise WorkspaceError(f"no job {job_id} in {self.ws.root}")
        return json.loads(f.read_text())

    def _write(self, rec):
        d = self._path(rec["id"])
        d.mkdir(exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(d / "job.json")

    # -- starting -------------------------------------------------------------------------------------------------
    def start(self, args, device="cuda", label="", outputs=None, cwd=None):
        """Queue the command ``feffgpu <args>`` and return its record at once.

        Parameters
        ----------
        args : list of str
            Arguments after ``feffgpu``; the leading words must be one of the allowed subcommands.
        device : str
            The device the job will use; jobs on the same device run one after the other.
        label : str
            A short description for listings.
        outputs : list of str, optional
            Paths (inside the workspace) where the results will be, for ``result``.
        cwd : str, optional
            Working directory of the command (inside the workspace; default the workspace).
        """
        args = [str(a) for a in args]
        if not any(tuple(args[:n]) in ALLOWED for n in (1, 2)):
            raise WorkspaceError(f"`feffgpu {' '.join(args[:2])}` is not a command the server may start")
        cwd = self.ws.resolve(cwd or ".", must_exist=True)
        rec = {"id": uuid.uuid4().hex[:12], "label": label or " ".join(args[:2]), "args": args, "device": device, "state": "queued",
               "created": time.time(), "started": None, "ended": None, "returncode": None, "pid": None, "cwd": str(cwd),
               "outputs": [str(self.ws.resolve(o)) for o in (outputs or [])]}
        with self._lock:
            self._write(rec)
            self._pump()
            self._ensure_watcher()
        return self.status(rec["id"])

    def _launch(self, rec):
        d = self._path(rec["id"])
        log = open(d / "log.txt", "ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if str(rec["device"]).startswith("cuda:"):                           # bind the job to its GPU; inside the job that GPU is cuda:0
            env["CUDA_VISIBLE_DEVICES"] = str(rec["device"]).split(":", 1)[1]
        proc = subprocess.Popen([self.python, "-m", "feffgpu.cli", *rec["args"]], cwd=rec["cwd"], stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        log.close()
        self._procs[rec["id"]] = proc
        rec.update(state="running", started=time.time(), pid=proc.pid)
        self._write(rec)

    def _refresh(self, rec):
        """Bring a running job's record up to date with its process."""
        if rec["state"] != "running":
            return rec
        proc = self._procs.get(rec["id"])
        code = proc.poll() if proc is not None else (None if _alive(rec["pid"]) else -1)
        if code is not None:
            rec.update(state="done" if code == 0 else "failed", ended=time.time(), returncode=code)
            if proc is None and code == -1:
                rec["note"] = "the process ended while no server was watching it; its exit code is not known: read the log"
            self._procs.pop(rec["id"], None)
            self._write(rec)
        return rec

    def _all(self):
        out = []
        for d in sorted(self.dir.iterdir()):
            f = d / "job.json"
            if f.exists():
                out.append(json.loads(f.read_text()))
        return sorted(out, key=lambda r: r["created"])

    def _pump(self):
        """Start queued jobs whose device is free (oldest first)."""
        recs = [self._refresh(r) for r in self._all()]
        busy = {r["device"] for r in recs if r["state"] == "running"}
        for r in recs:
            if r["state"] == "queued" and r["device"] not in busy:
                self._launch(r)
                busy.add(r["device"])

    def _ensure_watcher(self):
        if self._watcher is not None and self._watcher.is_alive():
            return

        def watch():
            while True:
                time.sleep(2.0)
                with self._lock:
                    self._pump()
                    if not any(r["state"] in ("queued", "running") for r in self._all()):
                        return
        self._watcher = threading.Thread(target=watch, daemon=True, name="feffgpu-jobs")
        self._watcher.start()

    # -- following ------------------------------------------------------------------------------------------------
    def status(self, job_id):
        """The record of a job: ``id``, ``label``, ``state`` (queued, running, done, failed, canceled), times, the
        exit code, ``seconds`` (run time so far), and the last line of its log."""
        with self._lock:
            self._pump()
            rec = self._refresh(self._read(job_id))
        out = {k: rec.get(k) for k in ("id", "label", "state", "device", "returncode", "note")}
        out["command"] = "feffgpu " + " ".join(rec["args"])
        if rec["started"]:
            out["seconds"] = round((rec["ended"] or time.time()) - rec["started"], 1)
        tail = self.log(job_id, lines=1)["lines"]
        out["last_line"] = tail[-1] if tail else ""
        return {k: v for k, v in out.items() if v is not None}

    def list(self):
        """All jobs of the workspace, oldest first (short records)."""
        with self._lock:
            self._pump()
            return [self.status(r["id"]) for r in self._all()]

    def log(self, job_id, lines=40):
        """The last ``lines`` lines of a job's output (at most 400)."""
        f = self._path(job_id) / "log.txt"
        self._read(job_id)
        if not f.exists():
            return {"id": job_id, "lines": []}
        n = max(1, min(int(lines), 400))
        with open(f, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            text = fh.read().decode(errors="replace")
        return {"id": job_id, "lines": [ln[:500] for ln in text.splitlines()[-n:]]}

    def cancel(self, job_id):
        """Stop a queued or running job (the process group gets SIGTERM, then SIGKILL after 5 s)."""
        with self._lock:
            rec = self._refresh(self._read(job_id))
            if rec["state"] == "queued":
                rec.update(state="canceled", ended=time.time())
                self._write(rec)
            elif rec["state"] == "running":
                try:
                    os.killpg(rec["pid"], signal.SIGTERM)
                    for _ in range(50):
                        if not _alive(rec["pid"]):
                            break
                        time.sleep(0.1)
                    else:
                        os.killpg(rec["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc = self._procs.pop(job_id, None)
                if proc is not None:
                    proc.wait(timeout=10)
                rec.update(state="canceled", ended=time.time())
                self._write(rec)
            self._pump()
        return self.status(job_id)

    def result(self, job_id):
        """The state of a job and where its results are: the declared output paths that exist (with sizes), and, for
        a directory, the names of the files in it. Arrays are never returned; read the files."""
        st = self.status(job_id)
        rec = self._read(job_id)
        files = []
        for o in rec["outputs"]:
            p = Path(o)
            if p.is_dir():
                names = sorted(q.name for q in p.iterdir())
                files.append({"path": self.ws.relative(p), "kind": "directory", "files": names[:60], "more": max(0, len(names) - 60)})
            elif p.exists():
                files.append({"path": self.ws.relative(p), "kind": "file", "bytes": p.stat().st_size})
            else:
                files.append({"path": self.ws.relative(p), "kind": "missing"})
        st["outputs"] = files
        return st

    def wait(self, job_id, timeout=None, poll=0.5):
        """Block until the job has ended (for scripts and tests; an agent should poll ``status`` instead)."""
        t0 = time.time()
        while True:
            st = self.status(job_id)
            if st["state"] in ("done", "failed", "canceled"):
                return st
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(f"job {job_id} still {st['state']} after {timeout} s")
            time.sleep(poll)
