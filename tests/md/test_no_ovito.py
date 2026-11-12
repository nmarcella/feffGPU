"""MD-EXAFS does not need OVITO: no source, test, documentation page or example imports it."""
from mdexafs_testing import ROOT

NAME = "OVITO".lower()                                              # the package that md_exafs 2.0 needed
PATTERNS = (f"import {NAME}", f"from {NAME}")


def test_nothing_imports_OVITO():
    hits = []
    for folder in ("src", "tests", "docs", "examples"):
        for p in (ROOT / folder).rglob("*"):
            text_file = p.suffix in (".py", ".md", ".toml", ".txt", ".sh", ".sbatch", ".cfg", ".ini", ".rst")
            if not p.is_file() or "_build" in p.parts or not text_file:
                continue
            text = p.read_text(errors="ignore")
            hits += [f"{p.relative_to(ROOT)}: {pat}" for pat in PATTERNS if pat in text]
    assert not hits, hits
