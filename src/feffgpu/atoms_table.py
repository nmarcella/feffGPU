"""The free-atom table: FEFF's single-configuration Dirac-Fock results for every element (Z = 1..99) as a neighbor
and as an absorber with each core hole.

The potentials stage reads the table in place of running FEFF's atomic code. It ships inside the package, one file
per element in ``feffgpu/data/atoms``: ``Z001.npz.xz`` to ``Z099.npz.xz`` (76 MB in all). Each file is a plain numpy
``.npz`` archive compressed as a whole with xz; the arrays are float64 and are stored without any loss.

``FEFFGPU_ATOMS`` points the package at another directory, for example a table that you built yourself with
``tools/build_atoms_table.py`` (needs the optional FEFF 8.5L library, ``fortran/README.md``). Such a directory may
hold plain ``Z001.npz`` files; they are used as they are. ``python -m feffgpu.atoms_table pack SOURCE TARGET`` packs
a directory of plain files into the form that ships.
"""
from __future__ import annotations
import functools
import io
import lzma
import os
import sys
import zipfile
import numpy as np

N_ELEMENTS = 99


def table_dir():
    """The directory of the table: ``$FEFFGPU_ATOMS`` if set, else ``data/atoms`` inside the installed package."""
    return os.environ.get("FEFFGPU_ATOMS", os.path.join(os.path.dirname(__file__), "data", "atoms"))


def element_file(z, directory=None):
    """The file of element ``z`` in the table directory: the plain ``Z074.npz`` if it is there, else the packed
    ``Z074.npz.xz``; None if neither exists."""
    d = table_dir() if directory is None else directory
    for name in (f"Z{int(z):03d}.npz", f"Z{int(z):03d}.npz.xz"):
        f = os.path.join(d, name)
        if os.path.isfile(f):
            return f
    return None


@functools.lru_cache(maxsize=8)
def _load(path, mtime):
    if path.endswith(".xz"):
        with open(path, "rb") as f:
            return np.load(io.BytesIO(lzma.decompress(f.read())))
    return np.load(path)


def load_element(z, directory=None):
    """The table of one element as a numpy archive (``arrays = load_element(74); arrays["holes"]``), or None if the
    element is not in the table. The last few elements read stay in memory."""
    f = element_file(z, directory)
    return None if f is None else _load(f, os.path.getmtime(f))


def table_status(directory=None):
    """``{"files": number of elements found, "complete": all 99 present}`` for the table directory."""
    n = sum(element_file(z, directory) is not None for z in range(1, N_ELEMENTS + 1))
    return {"files": n, "complete": n >= N_ELEMENTS}


def pack_element(npz_path, out_path):
    """Write the packed form of one plain ``.npz`` file: the same arrays in an uncompressed ``.npz`` archive, compressed
    as a whole with xz. The arrays read back from the packed file are checked to be identical, byte for byte."""
    src = np.load(npz_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in src.files:
            item = io.BytesIO()
            np.lib.format.write_array(item, np.asanyarray(src[key]), allow_pickle=False)
            archive.writestr(key + ".npy", item.getvalue())
    packed = lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME)
    back = np.load(io.BytesIO(lzma.decompress(packed)))
    if set(back.files) != set(src.files):
        raise RuntimeError(f"{npz_path}: the packed file does not hold the same arrays")
    for key in src.files:
        a, b = src[key], back[key]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise RuntimeError(f"{npz_path}: array {key} changed in packing")
    with open(out_path, "wb") as f:
        f.write(packed)
    return len(packed)


def pack_table(source, target, quiet=False):
    """Pack every plain ``Z*.npz`` of ``source`` into ``target`` as ``Z*.npz.xz``. Returns the total size in bytes."""
    os.makedirs(target, exist_ok=True)
    total = 0
    names = sorted(n for n in os.listdir(source) if n.startswith("Z") and n.endswith(".npz"))
    if not names:
        raise SystemExit(f"feffgpu: no Z*.npz files in {source}")
    for name in names:
        total += pack_element(os.path.join(source, name), os.path.join(target, name + ".xz"))
        if not quiet:
            print(f"{name} -> {name}.xz", flush=True)
    if not quiet:
        print(f"{len(names)} elements, {total / 1e6:.1f} MB in {target}")
    return total


def main(argv=None):
    """``python -m feffgpu.atoms_table status`` or ``python -m feffgpu.atoms_table pack SOURCE TARGET``."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m feffgpu.atoms_table", description="The free-atom table that ships with feffgpu.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="where the table is and how many elements it holds")
    p = sub.add_parser("pack", help="pack a directory of plain Z*.npz files (from tools/build_atoms_table.py) into Z*.npz.xz")
    p.add_argument("source")
    p.add_argument("target")
    a = ap.parse_args(argv)
    if a.command == "status":
        st = table_status()
        print(f"{st['files']} of {N_ELEMENTS} elements in {table_dir()}" + ("" if st["complete"] else " (incomplete)"))
        return 0 if st["complete"] else 1
    pack_table(a.source, a.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
