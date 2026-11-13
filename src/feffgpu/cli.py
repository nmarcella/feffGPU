"""The ``feffgpu`` command.

::

    feffgpu feff.inp --out chi.dat        # FEFF input -> chi(k), every stage on the GPU (the same as `feffgpu run`)
    feffgpu phase feff.inp phase.pkl      # potentials and phase shifts only, for MD-EXAFS and RMC-EXAFS
    feffgpu doctor                        # versions, GPUs, the free-atom table, optional parts
    feffgpu md ...                        # MD-EXAFS  (the same as the command `md-exafs`)
    feffgpu rmc ...                       # RMC-EXAFS (the same as the command `rmc-exafs`)
    feffgpu mcp ...                       # the MCP server for agents (the same as `feffgpu-mcp`)

``feffgpu-phase`` remains as a separate command.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import pickle
import sys
import time

SUBCOMMANDS = ("run", "phase", "doctor", "md", "rmc", "mcp")


def _fail(message):
    """Exit with one clear message (no traceback) on a user error."""
    raise SystemExit(f"feffgpu: {message}")


def check_device(device):
    """Return ``device`` if it can be used; exit with the fix if CUDA is asked for and missing."""
    import torch
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        built = torch.version.cuda
        if built is None:
            _fail("this PyTorch is a CPU-only build. Install the CUDA build that matches your NVIDIA driver "
                  "(https://pytorch.org/get-started/locally/), or pass --device cpu for a small input (slow).")
        _fail(f"PyTorch was built for CUDA {built} but no NVIDIA GPU is usable (check `nvidia-smi` and CUDA_VISIBLE_DEVICES), "
              "or pass --device cpu for a small input (slow).")
    return device


def _run_guarded(function, *args, **kwargs):
    """Call a pipeline function and turn the known user errors into one clear message."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError as err:                       # an unsupported FEFF card names itself
        _fail(f"{err}. feffgpu calculates EXAFS only; see the documentation page 'Limits'.")
    except FileNotFoundError as err:
        _fail(str(err))
    except RuntimeError as err:
        if type(err).__name__ != "KernelCheckError":         # the GPU kernels failed their check: one message, no traceback
            raise
        _fail(str(err))


def main(argv=None):
    """Entry point of ``feffgpu``: dispatch to a subcommand; a first argument that is not a subcommand is a FEFF
    input file (the form ``feffgpu feff.inp --out chi.dat``)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        name, rest = argv[0], argv[1:]
        if name == "run":
            return run_main(rest)
        if name == "phase":
            return phase_main(rest)
        if name == "doctor":
            return doctor_main(rest)
        module = {"md": "feffgpu.md.cli", "rmc": "feffgpu.rmc.cli", "mcp": "feffgpu.mcp.server"}[name]
        try:
            target = importlib.import_module(module)
        except ImportError as err:
            extra = {"mcp": " (install the extra: pip install 'feffgpu[mcp]')"}.get(name, "")
            _fail(f"`feffgpu {name}` could not be loaded: {err}{extra}")
        return target.main(rest)
    if argv and argv[0] == "fetch-atoms":                    # a command of version 0.1: the table is part of the package now
        from .atoms_table import table_dir, table_status
        st = table_status()
        print(f"The free-atom table ships with the package since version 0.2: nothing to download ({st['files']} elements in {table_dir()}).")
        return 0 if st["complete"] else 1
    if argv and argv[0] in ("-V", "--version"):
        from . import __version__
        print(f"feffgpu {__version__}")
        return 0
    return run_main(argv)


def run_main(argv=None):
    """``feffgpu [run] feff.inp``: every stage, chi(k) written like FEFF's chi.dat."""
    ap = argparse.ArgumentParser(prog="feffgpu", description="FEFF 8.5L's EXAFS pipeline on the GPU: feff.inp -> chi(k) (S0^2 = 1, no Debye-Waller "
                                 "factors, like FEFF's chi.dat) with every stage in PyTorch. Other tasks: feffgpu phase | doctor | md | rmc | mcp.")
    ap.add_argument("inp", help="a FEFF input file (feff.inp)")
    ap.add_argument("--out", default="chi_gpu.dat", help="output file (default chi_gpu.dat)")
    ap.add_argument("--device", default="cuda", help="torch device (default cuda; cuda:1, cpu)")
    ap.add_argument("--lmax", type=int, default=None, help="override FEFF's partial-wave rule (default: FEFF's own, from the muffin-tin radius and kmax)")
    ap.add_argument("--json", action="store_true", help="print a JSON summary (path counts, Fermi level, timings) instead of the text line")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.inp):
        _fail(f"no such input file: {a.inp}")
    check_device(a.device)
    from .chain import run
    t0 = time.time()
    out = _run_guarded(run, a.inp, device=a.device, verbose=a.verbose, lmax_override=a.lmax)
    P = out["input"]
    with open(a.out, "w") as f:
        for line in P["title"]:
            f.write(f"# {line}\n")
        f.write(f"# feffgpu: {sum(out['counts'].values())} paths ({out['counts']}), Fermi level {out['pot'].xmu * 27.211386:.4f} eV, "
                f"muffin-tin zero {out['pot'].vint * 27.211386:.4f} eV\n#   k          chi\n")
        for k, c in zip(out["k"], out["chi"]):
            f.write(f"{k:10.4f} {c:16.8e}\n")
    if a.json:
        print(json.dumps({"output": a.out, "paths": {str(k): v for k, v in out["counts"].items()}, "fermi_level_eV": out["pot"].xmu * 27.211386,
                          "muffin_tin_zero_eV": out["pot"].vint * 27.211386, "seconds": {k: round(v, 3) for k, v in out["timing"].items()},
                          "total_seconds": round(time.time() - t0, 3)}))
    else:
        print(f"wrote {a.out}; stages (s): " + ", ".join(f"{k} {v:.1f}" for k, v in out["timing"].items()) + f"; total {time.time() - t0:.1f}")
    return 0


def build_phase(feff_inp, xkmax=24.5, lmax=None, device="cuda"):
    """Potentials (free atoms and the self-consistency loop) and phase shifts of a FEFF input.

    Parameters
    ----------
    feff_inp : str or Path
        A FEFF input of a representative cluster. Its ``POTENTIALS`` block defines the potential index of every
        element (0 the absorber, 1.. the scatterers); MD-EXAFS and RMC-EXAFS use the same indices.
    xkmax : float
        Largest k of the phase grid (1/A).
    lmax : int, optional
        Override FEFF's partial-wave rule.
    device : str

    Returns
    -------
    phase : feffgpu.PhaseData
    info : dict
        ``potentials`` (labels), ``Z``, ``seconds_potentials``, ``seconds_phases``.
    """
    os.environ.pop("FEFF_LMAX", None)
    from .inp import read_feff_inp
    from .pad import BOHR
    from .pot import potentials
    from .xsph import xsph
    P = read_feff_inp(feff_inp)
    t0 = time.time()
    pot = potentials(P["iz"], P["ihole"], P["xion"], P["iunf"], P["nph"], P["nat"], P["iphat"], P["rat"], P["iatph"], P["xnatph"],
                     P["folp"], P["iafolp"], P["ixc"], nohole=P["nohole"], inters=P["inters"], totvol=P["totvol_bohr"], ecv0=P["ecv_ha"],
                     title=P["title"], nscmt=P["nscmt"], rfms1=P["rfms1_bohr"], lfms1=P["lfms1"], ca1=P["ca1"], lmaxsc=P["lmaxsc"],
                     icoul=P["icoul"], nmix=P["nmix"], rgrd=P["rgrd"], jumprm=P["jumprm"], verbose=False, device=device)
    t1 = time.time()
    ph, _xsnorm = xsph(pot, gamach=P["gamach_ha"], xkmax=xkmax * BOHR, ixc=P["ixc"], vr0=P["vr0_ha"], vi0=P["vi0_ha"], lreal=P["lreal"],
                       rgrd=P["rgrd"], lmax_override=lmax, ixc0=P["ixc0"], potlbl=P["potlbl"], device=device)
    info = {"potentials": [str(v) for v in P["potlbl"]], "Z": [int(v) for v in P["iz"]], "seconds_potentials": t1 - t0,
            "seconds_phases": time.time() - t1, "ne1": int(ph.ne1), "ltot": int(ph.ltot)}
    return ph, info


def phase_main(argv=None):
    """``feffgpu phase feff.inp out.pkl`` (also installed as ``feffgpu-phase``)."""
    ap = argparse.ArgumentParser(prog="feffgpu phase", description="Phase file for a feff.inp: potentials (free atoms + self-consistency) and phase "
                                 "shifts on the GPU, pickled as feffgpu.PhaseData for the path stage, MD-EXAFS and RMC-EXAFS. "
                                 "A pickle is a Python object file: load only phase files that you made yourself.")
    ap.add_argument("feff_inp")
    ap.add_argument("out", help="output pickle, for example phase.pkl")
    ap.add_argument("--xkmax", type=float, default=24.5, help="largest k of the phase grid (1/A)")
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.feff_inp):
        _fail(f"no such input file: {args.feff_inp}")
    check_device(args.device)
    ph, info = _run_guarded(build_phase, args.feff_inp, xkmax=args.xkmax, lmax=args.lmax, device=args.device)
    with open(args.out, "wb") as f:
        pickle.dump(ph, f)
    if args.json:
        print(json.dumps({"output": args.out, **info}))
    else:
        print(f"potentials in {info['seconds_potentials']:.1f} s, phases in {info['seconds_phases']:.1f} s: potentials {info['potentials']}, "
              f"Z {info['Z']}, ne1 {info['ne1']}, ltot {info['ltot']}")
        print(f"written {args.out}")
    return 0


def environment_report():
    """What ``feffgpu doctor`` prints, as a dict: versions, GPUs, the free-atom table, the optional parts."""
    import platform
    import torch
    from . import __version__
    from .atoms_table import table_dir, table_status
    rep = {"feffgpu": __version__, "python": platform.python_version(), "platform": platform.platform(),
           "torch": torch.__version__, "torch_cuda_build": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()), "gpus": []}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            rep["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i), "memory_total_GB": round(total / 2 ** 30, 2),
                                "memory_free_GB": round(free / 2 ** 30, 2)})
    rep["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:                                                      # the GPU kernels of the radial solver, checked against the torch code
        from .xsph import kernel_status
        rep["gpu_kernels"] = kernel_status("cuda" if rep["cuda_available"] else None, test=True)      # doctor always runs the test
    except Exception as err:
        rep["gpu_kernels"] = {"usable": False, "note": f"could not be checked ({err!r})"[:200]}
    rep["atoms_table"] = {"directory": table_dir(), **table_status()}
    try:
        from .potph_lib import find_library
        rep["feff_library"] = {"found": True, "path": str(find_library())}
    except Exception as err:                                  # optional: only ions, unfrozen f electrons, and the oracle tests need it
        rep["feff_library"] = {"found": False, "note": "optional; needed only for ION / UNFREEZEF cards, rebuilding the free-atom table, and the "
                               "oracle tests (fortran/README.md)", "detail": str(err).splitlines()[0][:200]}
    packages = {}
    for mod, extra in (("numpy", None), ("scipy", None), ("ase", None), ("h5py", None), ("pydantic", None), ("matplotlib", None),
                       ("MDAnalysis", "mda"), ("mace", "mace"), ("larch", "larch"), ("mcp", "mcp")):
        try:
            m = importlib.import_module(mod)
            packages[mod] = getattr(m, "__version__", "installed")
        except Exception:
            packages[mod] = None if extra is None else f"not installed (optional: pip install 'feffgpu[{extra}]')"
    rep["packages"] = packages
    problems = []
    if not rep["cuda_available"]:
        problems.append("no usable NVIDIA GPU: feffgpu runs on the CPU only for small inputs and slowly")
    if rep["cuda_available"] and not rep["gpu_kernels"]["usable"]:
        if rep["gpu_kernels"]["note"].startswith("switched off on purpose"):
            rep["notes"] = ["the GPU kernels of the radial solver are switched off (FEFFGPU_KERNELS=0): results are the same, potentials "
                            "and phase shifts take longer"]
        else:
            problems.append(rep["gpu_kernels"]["note"])
    if not rep["atoms_table"]["complete"]:
        problems.append("the free-atom table is missing or incomplete (it ships with the package; check FEFFGPU_ATOMS if it is set)")
    for mod in ("numpy", "scipy", "ase", "h5py", "pydantic", "matplotlib"):
        if packages[mod] is None:
            problems.append(f"the required package {mod} is not installed")
    rep["problems"] = problems
    return rep


def doctor_main(argv=None):
    """``feffgpu doctor``: check the installation."""
    ap = argparse.ArgumentParser(prog="feffgpu doctor", description="Check the installation: versions, GPUs and their memory, the free-atom table, "
                                 "the optional FEFF library and the optional packages. Exit code 1 if something required is missing.")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = environment_report()
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print(f"feffgpu {rep['feffgpu']}, Python {rep['python']}, PyTorch {rep['torch']} (CUDA build {rep['torch_cuda_build']})")
        if rep["gpus"]:
            for g in rep["gpus"]:
                print(f"  GPU {g['index']}: {g['name']}, {g['memory_free_GB']} of {g['memory_total_GB']} GB free")
        else:
            print("  no usable NVIDIA GPU")
        if rep["gpus"]:
            print("  GPU kernels of the radial solver: " + rep["gpu_kernels"]["note"])
        t = rep["atoms_table"]
        print(f"  free-atom table: {t['files']} element files in {t['directory']}" + ("" if t["complete"] else "  <- incomplete"))
        lib = rep["feff_library"]
        print("  FEFF 8.5L library (optional): " + (lib["path"] if lib["found"] else "not found (" + lib["note"] + ")"))
        print("  packages: " + ", ".join(f"{k} {v}" for k, v in rep["packages"].items() if v and not str(v).startswith("not installed")))
        missing = [f"{k}: {v}" for k, v in rep["packages"].items() if v is None or str(v).startswith("not installed")]
        if missing:
            print("  not installed: " + "; ".join(m.replace(": None", " (required)") for m in missing))
        for note in rep.get("notes", []):
            print("  note: " + note)
        print("OK" if not rep["problems"] else "PROBLEMS:\n  " + "\n  ".join(rep["problems"]))
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
