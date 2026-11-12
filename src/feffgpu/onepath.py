"""In-process FEFF path stage: effective amplitude and phase of one path for any geometry.

Binds `calc_onepath` from a rebuilt FEFF (fortran/lib/libfeff8lpath.so or FEFF_HOME, 80 partial
waves) through Larch's ctypes wrapper, pointed at the rebuilt library instead of Larch's own 24-wave build. The
phase shifts come from a `phase.pad` written by the potentials + phase-shift stages (for example tests/data/W/phase.pad);
only the path geometry and the potential index of each atom are supplied here, so no FEFF process is started and
nothing is written to disk. FEFF's reader opens the fixed name `phase.pad` in the working directory (rdxsph.f), so
every call runs with the working directory temporarily set to the phase file's folder.

Usage:
    op = OnePath(phase_pad)                      # once
    out = op.compute(coords, ipots)              # coords (nleg, 3) A, absorber first; ipots (nleg,), absorber 0
    out["k"], out["mag"], out["phase"], out["real_phc"], out["red_fact"], out["lam"], out["rep"], out["reff"]
"""
from __future__ import annotations
import ctypes, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEFF_HOME = Path(os.environ.get("FEFF_HOME", ROOT / "fortran"))


def _load_library():
    import larch.xafs.feff8lpath as f8
    if f8.F8LIB is None or getattr(f8.F8LIB, "_rebuilt", False) is False:
        lib_dir = FEFF_HOME / "lib"
        for dep in ("libonepath.so",):                       # dependencies first, global so symbols resolve
            p = lib_dir / dep
            if p.exists():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(str(lib_dir / "libfeff8lpath.so"), mode=ctypes.RTLD_GLOBAL)
        lib._rebuilt = True
        f8.F8LIB = lib
    return f8


class OnePath:
    def __init__(self, phase_pad):
        self.f8 = _load_library()
        self.phase_pad = str(phase_pad)
        if not Path(self.phase_pad).exists():
            raise FileNotFoundError(self.phase_pad)
        self.phase_dir = str(Path(self.phase_pad).resolve().parent)
        if Path(self.phase_pad).name != "phase.pad":
            raise ValueError("FEFF's reader only opens a file named phase.pad; copy or link it under that name")
        self._path = self.f8.Feff8L_XAFSPath(phase_file=self.phase_pad)

    def compute(self, coords, ipots, degen=1.0, index=9999):
        coords = np.asarray(coords, float)
        ipots = [int(i) for i in ipots]
        if len(coords) != len(ipots) or len(coords) < 2 or len(coords) > 9:
            raise ValueError("coords/ipots: 2 to 9 atoms (absorber first), one potential index per atom")
        if np.linalg.norm(coords[0]) > 1e-9 or ipots[0] != 0:
            raise ValueError("the absorber must be first, at the origin, with potential index 0")
        p = self._path
        p.reset(phase_file=self.phase_pad)
        p.degen, p.index = float(degen), int(index)
        p.set_absorber(0.0, 0.0, 0.0)
        for (x, y, z), ip in zip(coords[1:], ipots[1:]):
            p.add_scatterer(float(x), float(y), float(z), ipot=ip)
        cwd = os.getcwd()
        os.chdir(self.phase_dir)
        try:
            p.calculate_xafs()
        finally:
            os.chdir(cwd)
        n = int(p.nepts)
        return {"k": np.array(p.k[:n]), "mag": np.array(p.mag_feff[:n]), "phase": np.array(p.pha_feff[:n]),
                "real_phc": np.array(p.real_phc[:n]), "red_fact": np.array(p.red_fact[:n]),
                "lam": np.array(p.lam[:n]), "rep": np.array(p.rep[:n]), "reff": float(p.reff),
                "nleg": int(p.nleg), "degen": float(p.degen)}
