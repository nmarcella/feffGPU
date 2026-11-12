"""The free-atom table: FEFF's single-configuration Dirac-Fock results for every element (Z = 1..99) as a neighbor
and as an absorber with each core hole, one compressed file per element (100 MB in all).

The potentials stage reads it in place of running FEFF's atomic code. It is not part of the installed package
(too large for a wheel). Get it once with::

    feffgpu fetch-atoms                      # download into the package directory
    feffgpu fetch-atoms /data/feff_atoms     # or into a directory of your choice; then set FEFFGPU_ATOMS to it
    feffgpu fetch-atoms --from-file feffgpu_atoms_v1.tar.gz     # offline: from a file copied by hand

The download is checked against a SHA-256 checksum. The table can also be rebuilt with ``tools/build_atoms_table.py``
(needs the optional FEFF 8.5L library, ``fortran/README.md``).
"""
from __future__ import annotations
import hashlib
import os
import sys
import tarfile
import urllib.error
import urllib.request

REPO, TAG, ASSET = "nmarcella/feffGPU", "atoms-v1", "feffgpu_atoms_v1.tar.gz"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
SHA256 = "57341c76b5527420b42a01593e008345a118d87286b205e2857c3339bae9f01f"
N_ELEMENTS = 99


def table_dir():
    """The directory of the table: ``$FEFFGPU_ATOMS`` if set, else ``data/atoms`` inside the installed package."""
    return os.environ.get("FEFFGPU_ATOMS", os.path.join(os.path.dirname(__file__), "data", "atoms"))


TABLE_DIR = table_dir()


def table_status(directory=None):
    """``{"files": number of element files found, "complete": all 99 present}`` for the table directory."""
    d = table_dir() if directory is None else directory
    n = len([f for f in os.listdir(d) if f.startswith("Z") and f.endswith(".npz")]) if os.path.isdir(d) else 0
    return {"files": n, "complete": n >= N_ELEMENTS}


def sha256_of(path, chunk=1 << 20):
    """The SHA-256 checksum of a file, as hexadecimal text."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download(url, tmp, token=None):
    """Plain download; with a token (GH_TOKEN or GITHUB_TOKEN in the environment, needed only for a private fork)
    through the GitHub API's asset endpoint."""
    import json
    if token is None:
        urllib.request.urlretrieve(url, tmp)
        return
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "feffgpu"}
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", headers=headers)
    rel = json.load(urllib.request.urlopen(req))
    asset = next(a for a in rel["assets"] if a["name"] == ASSET)
    req = urllib.request.Request(asset["url"], headers={**headers, "Accept": "application/octet-stream"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def install_archive(archive, dest, checksum=SHA256):
    """Check an archive of the table against its checksum and extract the element files into ``dest``.

    Only members named ``Z*.npz`` are extracted, each under its base name, so an archive cannot write outside
    ``dest``. Returns the number of element files in ``dest``."""
    if checksum:
        found = sha256_of(archive)
        if found != checksum:
            raise SystemExit(f"feffgpu: {archive} does not have the expected SHA-256 checksum\n  expected {checksum}\n  found    {found}\n"
                             "The file is damaged or is not the free-atom table of this version. Download it again.")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name).startswith("Z") and m.name.endswith(".npz")]
        for m in members:
            m.name = os.path.basename(m.name)
        tar.extractall(dest, members=members)
    return table_status(dest)["files"]


def fetch(url=URL, dest=None, quiet=False, from_file=None, checksum=SHA256):
    """Download (or take from ``from_file``) the free-atom table and install it into ``dest``.

    Parameters
    ----------
    url : str
        Where the archive is (default: the release asset of this package's repository).
    dest : str, optional
        Target directory (default ``table_dir()``). If it is not the default, set ``FEFFGPU_ATOMS`` to it.
    quiet : bool
    from_file : str, optional
        An archive that is already on disk (offline installation).
    checksum : str or None
        Expected SHA-256 of the archive; None skips the check (for a table you built yourself).

    Returns
    -------
    int
        Number of element files installed.
    """
    dest = table_dir() if dest is None else dest
    os.makedirs(dest, exist_ok=True)
    if from_file is not None:
        if not os.path.isfile(from_file):
            raise SystemExit(f"feffgpu: no such file: {from_file}")
        n = install_archive(from_file, dest, checksum)
    else:
        tmp = os.path.join(dest, "atoms.tar.gz")
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not quiet:
            print(f"downloading {url} -> {dest}", flush=True)
        try:
            try:
                _download(url, tmp, None)
            except urllib.error.HTTPError as err:
                if err.code in (401, 403, 404) and token is not None and url == URL:
                    _download(url, tmp, token)                       # a private fork: through the API with the token
                else:
                    raise
        except (urllib.error.URLError, OSError) as err:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise SystemExit(f"feffgpu: the download failed ({err}).\nWithout network access on this machine: download\n  {url}\n"
                             f"elsewhere, copy it here, and run `feffgpu fetch-atoms --from-file {ASSET}`.")
        try:
            n = install_archive(tmp, dest, checksum if url == URL else None)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if not quiet:
        print(f"{n} element files in {dest}")
        if os.path.abspath(dest) != os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "atoms")) and os.environ.get("FEFFGPU_ATOMS") != dest:
            print(f"set the environment variable for later runs:  export FEFFGPU_ATOMS={dest}")
    return n


def main(argv=None):
    """``feffgpu fetch-atoms`` (also installed as ``feffgpu-fetch-atoms``)."""
    import argparse
    ap = argparse.ArgumentParser(prog="feffgpu fetch-atoms", description="Download the free-atom table (100 MB) that the potentials stage needs, "
                                 "check it against its SHA-256 checksum, and install it.")
    ap.add_argument("dest", nargs="?", default=None, help=f"directory to put it in (default: inside the package, {table_dir()}); "
                    "elsewhere, set FEFFGPU_ATOMS to that directory")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--from-file", default=None, help="install from an archive that is already on disk (no network)")
    ap.add_argument("--no-checksum", action="store_true", help="skip the checksum (for a table that you built yourself)")
    a = ap.parse_args(argv)
    fetch(url=a.url, dest=a.dest, from_file=a.from_file, checksum=None if a.no_checksum else SHA256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
