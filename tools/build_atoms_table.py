"""Build the free-atom table: FEFF's single-configuration Dirac-Fock results (ATOM/scfdat.f, through the bundled
library) for every element Z = 1..99 as a neighbour (no hole) and as an absorber with each occupied core hole
1..16 (K to N7), plus the initial-state orbital of each hole. Neutral atoms, f electrons frozen (FEFF's defaults).
Output: atoms_table_build/Z0NN.npz (or the directory in FEFFGPU_ATOMS), one file per element. To use the new table, either
set FEFFGPU_ATOMS to that directory, or pack it into the form that ships with the package:

    python tools/build_atoms_table.py [zmin zmax]
    python -m feffgpu.atoms_table pack atoms_table_build src/feffgpu/data/atoms
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feffgpu.potph_lib import scfdat
from feffgpu.xsph import getorb

OUT = os.environ.get("FEFFGPU_ATOMS", os.path.join(os.path.dirname(__file__), "..", "atoms_table_build"))
os.makedirs(OUT, exist_ok=True)
KEYS = ("vcoul", "rho", "dmag", "rhoval", "dgc", "dpc", "adgc", "adpc", "xnvmu", "xnval", "iorb", "eorb", "kappa")
t0 = time.time()


def worker(Z, holes, path):
    """One element: all requested holes; writes the file. Runs in a subprocess because FEFF's atom code stops the
    process when it fails to converge."""
    entry, ok = {}, []
    for h in holes:
        print(f"hole {h}", flush=True)
        A = scfdat(Z, h, 0.0, 0, 0)
        for k in KEYS:
            entry[f"h{h}_{k}"] = np.asarray(A[k])
        entry[f"h{h}_scalars"] = np.array([A["s02"], A["efrozn"], A["eatom"], A["norb"]])
        if h > 0:
            B = scfdat(Z, 0, 0.0, 0, h)                           # initial state, spinor of the hole orbital saved
            entry[f"h{h}_dgc0"] = B["dgc0"]
            entry[f"h{h}_dpc0"] = B["dpc0"]
            entry[f"h{h}_initial"] = np.array([B["s02"], B["efrozn"], B["eatom"]])
        ok.append(h)
    np.savez_compressed(path, holes=np.array(ok), **entry)


if len(sys.argv) > 3 and sys.argv[1] == "--worker":
    Z = int(sys.argv[2])
    holes = [int(v) for v in sys.argv[3].split(",")]
    worker(Z, holes, sys.argv[4])
    sys.exit(0)

import subprocess
zmin = int(sys.argv[1]) if len(sys.argv) > 1 else 1
zmax = int(sys.argv[2]) if len(sys.argv) > 2 else 99
for Z in range(zmin, zmax + 1):
    holes = [0]
    for h in range(1, 17):
        try:
            getorb(Z, h, 0.0, 0)
        except Exception:
            continue
        holes.append(h)
    path = os.path.join(OUT, f"Z{Z:03d}.npz")
    remaining, failed = list(holes), []
    while remaining:
        if os.path.exists(path):
            os.remove(path)
        r = subprocess.run([sys.executable, __file__, "--worker", str(Z), ",".join(map(str, remaining)), path],
                           capture_output=True, text=True)
        if os.path.exists(path):                 # FEFF's atom code stops the process with exit code 0 when it fails
            break
        done = [int(l.split()[1]) for l in r.stdout.splitlines() if l.startswith("hole ")]
        bad = done[-1] if done else remaining[0]
        failed.append(bad)
        remaining = [h for h in remaining if h != bad]
        print(f"Z {Z}: hole {bad} failed in FEFF's atom code, retrying without it", flush=True)
    print(f"Z {Z}: holes {[h for h in holes if h not in failed]} failed {failed} ({time.time() - t0:.0f} s)", flush=True)
