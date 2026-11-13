"""Run the tutorial notebooks and store them with their outputs.

    python tools/run_tutorials.py                              # every notebook of tutorials/
    python tools/run_tutorials.py 01_md_exafs_iron.ipynb       # one of them
    python tools/run_tutorials.py --check                      # run only; the files are not changed

Each notebook is executed with nbclient in a temporary copy of the tutorials folder, so nothing is written into the
checkout. In the stored outputs the paths of this machine are shortened: "..." stands for the temporary folder and
for the checkout. A notebook whose outputs still name a home directory is not written.

A notebook can state two things in its metadata, under "feffgpu": "needs_env", the name of an environment variable
that must point to files the repository does not hold (the notebook is skipped without it), and
"store_outputs": false, for a notebook that is kept without outputs in the repository. ``--out FOLDER`` writes the
executed notebooks into a folder instead (with their outputs, whatever the metadata says).

Needs the notebook tools: ``pip install 'feffgpu[notebooks]'``. The notebooks run their calculations on the GPU.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"


def shorten(text, replacements):
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def clean_outputs(nb, replacements):
    """Shorten machine paths in every text output of the notebook, and join the pieces of printed text (in place)."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        joined = []
        for out in cell.get("outputs", []):                      # a command's output arrives in many small pieces
            if joined and out.get("output_type") == "stream" and joined[-1].get("output_type") == "stream" \
                    and joined[-1].get("name") == out.get("name"):
                joined[-1]["text"] += out["text"]
            else:
                joined.append(out)
        cell["outputs"] = joined
        for out in cell["outputs"]:
            if "text" in out:
                out["text"] = shorten(out["text"], replacements)
            if "traceback" in out:
                out["traceback"] = [shorten(t, replacements) for t in out["traceback"]]
            for kind, value in out.get("data", {}).items():
                if kind.startswith("text/") and isinstance(value, str):
                    out["data"][kind] = shorten(value, replacements)


def run_notebook(path, check=False, timeout=14400, out=None):
    """Execute one notebook of tutorials/ in a temporary copy of the folder; store it with its outputs unless
    ``check`` (or its metadata says that it is kept without outputs), or write it into the folder ``out``. Returns the
    executed notebook, or None for a notebook whose required environment variable is not set."""
    import nbformat
    from nbclient import NotebookClient
    wishes = nbformat.read(path, as_version=4).metadata.get("feffgpu", {})
    if wishes.get("needs_env") and not os.environ.get(wishes["needs_env"]):
        print(f"{path.name}: skipped (it needs the environment variable {wishes['needs_env']})", flush=True)
        return None
    with tempfile.TemporaryDirectory(prefix="feffgpu_tutorials_") as tmp:
        tmp = Path(tmp).resolve()
        work = tmp / "repo" / "tutorials"
        shutil.copytree(TUTORIALS, work, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
        nb = nbformat.read(work / path.name, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell["outputs"], cell["execution_count"] = [], None
        # the notebook's own environment: the commands of this installation first on the path, and no optional
        # FEFF library of this machine (its path would appear in the report of `feffgpu doctor`)
        env = {name: value for name, value in os.environ.items() if name not in ("FEFF_HOME", "FEFFGPU_POTPH_LIB")}
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
        t0 = time.time()
        NotebookClient(nb, timeout=timeout, kernel_name="python3", record_timing=False,
                       resources={"metadata": {"path": str(work)}}).execute(env=env)
        seconds = time.time() - t0
        # the longest paths first, so that a folder inside another one is shortened by its own rule
        replacements = sorted([(str(tmp / "repo"), "..."), (str(tmp), "..."), (str(ROOT), "..."),
                               (str(Path(sys.prefix)), "...")], key=lambda r: -len(r[0]))
        clean_outputs(nb, replacements)
    text = json.dumps(nb)
    writes = out is not None or (not check and wishes.get("store_outputs", True))
    if writes and (str(Path.home()) in text or "/home/" in text):
        raise SystemExit(f"{path.name}: a home directory is still named in the outputs; not written")
    nb.metadata["language_info"] = {"name": "python"}
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, Path(out) / path.name)
        done = f", written to {out}"
    elif not check and wishes.get("store_outputs", True):
        nbformat.write(nb, path)
        done = ", stored with outputs"
    else:
        done = ""
    print(f"{path.name}: ran in {seconds:.0f} s{done}", flush=True)
    return nb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="*", help="file names in tutorials/ (default: every notebook)")
    ap.add_argument("--check", action="store_true", help="run the notebooks without changing the files")
    ap.add_argument("--out", default=None, help="write the executed notebooks into this folder instead")
    args = ap.parse_args()
    paths = [TUTORIALS / n for n in args.notebooks] or sorted(TUTORIALS.glob("*.ipynb"))
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"no notebook {p}")
        run_notebook(p, check=args.check, out=args.out)


if __name__ == "__main__":
    main()
