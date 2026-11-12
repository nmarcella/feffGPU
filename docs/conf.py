"""Sphinx configuration of the feffgpu documentation (built on Read the Docs without a GPU: the API reference imports
the package with a CPU-only PyTorch; nothing is executed during the build)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))
import feffgpu  # noqa: E402

project = "feffgpu"
author = "N. Marcella and the feffgpu contributors"
copyright = "2026, " + author
release = feffgpu.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "**/_make_*.py", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header"]

autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
python_use_unqualified_type_names = True

html_theme = "furo"
html_title = f"feffgpu {release}"
html_static_path = ["_static"]
html_extra_path = []
copybutton_prompt_text = r"\$ |>>> "
copybutton_prompt_is_regexp = True


def _write_llms_files(app, exception):
    """After an HTML build: ``llms.txt`` (an index of the pages for agents) and ``llms-full.txt`` (all pages as plain
    Markdown in reading order), next to the HTML."""
    if exception is not None or app.builder.name not in ("html", "dirhtml"):
        return
    src, out = Path(app.srcdir), Path(app.outdir)
    base = os.environ.get("READTHEDOCS_CANONICAL_URL", "").rstrip("/")
    order = [d for d in app.env.found_docs]
    toc = list(app.env.toctree_includes.get(app.config.root_doc, []))
    seen, ordered = set(), []

    def walk(name):
        if name in seen or name not in app.env.found_docs:
            return
        seen.add(name)
        ordered.append(name)
        for child in app.env.toctree_includes.get(name, []):
            walk(child)
    walk(app.config.root_doc)
    for name in toc + sorted(order):
        walk(name)
    index = ["# feffgpu", "", "> FEFF 8.5L's EXAFS pipeline on NVIDIA GPUs in PyTorch, with MD-EXAFS and RMC-EXAFS built on it.", "",
             "Every page of the documentation as Markdown is in llms-full.txt.", "", "## Pages", ""]
    full = []
    for name in ordered:
        f = src / (name + ".md")
        if not f.exists() or name.startswith("api/_generated"):
            continue
        text = f.read_text()
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), name)
        url = f"{base}/{name}.html" if base else f"{name}.html"
        index.append(f"- [{title}]({url})")
        full.append(f"<!-- page: {name} -->\n\n{text.strip()}\n")
    (out / "llms.txt").write_text("\n".join(index) + "\n")
    (out / "llms-full.txt").write_text("\n\n".join(full) + "\n")


# -- the API reference: one page per part of the package, written here from the modules that exist ---------------------
API_PARTS = [
    ("pipeline", "The FEFF pipeline", ["feffgpu.chain", "feffgpu.inp", "feffgpu.pot", "feffgpu.scf", "feffgpu.fms", "feffgpu.xsph", "feffgpu.kernels",
                                       "feffgpu.pathfinder", "feffgpu.stage", "feffgpu.genfmt", "feffgpu.pad", "feffgpu.cell", "feffgpu.compare",
                                       "feffgpu.atoms_table", "feffgpu.cli", "feffgpu.config_tools"]),
    ("exafs", "Conventions and the forward model (feffgpu.exafs)", "feffgpu.exafs"),
    ("io", "Structures and trajectories (feffgpu.io)", "feffgpu.io"),
    ("analysis", "Analysis of configurations (feffgpu.analysis)", "feffgpu.analysis"),
    ("parallel", "Devices, shards and random streams (feffgpu.parallel)", "feffgpu.parallel"),
    ("md", "MD-EXAFS (feffgpu.md)", "feffgpu.md"),
    ("rmc", "RMC-EXAFS (feffgpu.rmc)", "feffgpu.rmc"),
    ("mcp", "Agent access (feffgpu.mcp)", "feffgpu.mcp"),
]
# Modules that are ports of FEFF's routines keep their working notes as docstrings; they are shown verbatim.
VERBATIM_MODULES = ("feffgpu.chain", "feffgpu.inp", "feffgpu.pot", "feffgpu.scf", "feffgpu.fms", "feffgpu.xsph", "feffgpu.kernels",
                    "feffgpu.pathfinder", "feffgpu.stage", "feffgpu.genfmt", "feffgpu.pad", "feffgpu.cell", "feffgpu.compare")


def _modules_of(package):
    import importlib
    import pkgutil
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return []
    names = [package]
    for info in pkgutil.walk_packages(pkg.__path__, package + "."):
        if not info.name.rsplit(".", 1)[-1].startswith("_"):
            names.append(info.name)
    return names


def _write_api_pages():
    import importlib
    out = Path(__file__).parent / "api" / "_generated"
    out.mkdir(parents=True, exist_ok=True)
    for slug, title, spec in API_PARTS:
        modules = spec if isinstance(spec, list) else _modules_of(spec)
        lines = [f"# {title}", ""]
        if not modules:
            lines += ["This part of the package is not installed in the environment that built the documentation.", ""]
        for name in modules:
            mod = importlib.import_module(name)
            is_package = hasattr(mod, "__path__")
            lines += [f"## `{name}`", "", "```{eval-rst}", f".. automodule:: {name}"]
            lines += ["   :no-members:"] if is_package else ["   :members:", "   :show-inheritance:"]
            lines += ["```", ""]
        (out / f"{slug}.md").write_text("\n".join(lines))


_write_api_pages()


def _verbatim_docstrings(app, what, name, obj, options, lines):
    """Show the docstrings of the ported FEFF modules as they are written (they use FEFF's notation, not markup)."""
    if lines and name.startswith(VERBATIM_MODULES):
        text = list(lines)
        lines[:] = ["::", ""] + ["    " + ln for ln in text] + [""]


def setup(app):
    app.connect("autodoc-process-docstring", _verbatim_docstrings, priority=400)
    app.connect("build-finished", _write_llms_files)
