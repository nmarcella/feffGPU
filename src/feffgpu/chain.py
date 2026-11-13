"""feff.inp to chi(k) with every stage in this package: the input cards (inp), the potentials with FEFF's
free atoms and the self-consistency loop (pot, scf, fms), the phase shifts and matrix elements
(xsph), the path finder and the path stage on the GPU (pathfinder, stage). Returns chi on FEFF's own
fine k grid (the content of chi.dat: S0^2 = 1, no Debye-Waller factors) and the intermediate results."""
from __future__ import annotations
import time
import numpy as np
import torch
from .inp import read_feff_inp
from .pot import potentials
from .xsph import xsph
from .pathfinder import GPUPathFinder, NLEG_MAX
from .stage import GPUPathStage


def run(feff_inp, device="cuda", verbose=False, lmax_override=None, kgrid=None, pot=None, phase=None):
    """Returns dict(k, chi, pot, phase, counts, timing). kgrid: k points in 1/A (default FEFF's 0.05..30.05)."""
    t = {}
    t0 = time.time()
    P = read_feff_inp(feff_inp)
    t["input"] = time.time() - t0
    if pot is None:
        t0 = time.time()
        pot = potentials(P["iz"], P["ihole"], P["xion"], P["iunf"], P["nph"], P["nat"], P["iphat"], P["rat"], P["iatph"], P["xnatph"],
                         P["folp"], P["iafolp"], P["ixc"], nohole=P["nohole"], inters=P["inters"], totvol=P["totvol_bohr"], ecv0=P["ecv_ha"],
                         title=P["title"], nscmt=P["nscmt"], rfms1=P["rfms1_bohr"], lfms1=P["lfms1"], ca1=P["ca1"], lmaxsc=P["lmaxsc"],
                         icoul=P["icoul"], nmix=P["nmix"], rgrd=P["rgrd"], jumprm=P["jumprm"], verbose=verbose, device=device)
        t["potentials"] = time.time() - t0
    if phase is None:
        t0 = time.time()
        phase, xsnorm = xsph(pot, gamach=P["gamach_ha"], xkmax=P["xkmax_bohr"], ixc=P["ixc"], vr0=P["vr0_ha"], vi0=P["vi0_ha"], lreal=P["lreal"],
                             rgrd=P["rgrd"], lmax_override=lmax_override, ixc0=P["ixc0"], potlbl=P["potlbl"], device=device)
        t["phases"] = time.time() - t0
    t0 = time.time()
    finder = GPUPathFinder(phase, device=device, rmax=P["rmax"], nlegmax=max(2, min(P["nlegxx"], NLEG_MAX)), critpw=P["critpw"])
    kept, ref = finder.enumerate(P["rat_ang"], None, 0, np.array(P["iphat"]))
    t["paths"] = time.time() - t0
    t0 = time.time()
    kf = np.arange(0.05, 30.0501, 0.05) if kgrid is None else np.asarray(kgrid, float)
    stage = GPUPathStage(phase, device=device, dtype=torch.complex128)
    S = stage.linear_matrix(kf)
    chi = np.zeros(len(kf))
    counts = {}
    with torch.no_grad():
        for nleg, (coords, ipots, deg, xp, atoms) in kept.items():
            counts[nleg] = int(coords.shape[0])
            for s in range(0, coords.shape[0], 4096):
                cc, rf = stage.cchi(coords[s:s + 4096].to(torch.float64), ipots[s:s + 4096])
                chi += (deg[s:s + 4096, None].to(torch.float64) * stage.chi_fine(cc, rf, kf, S, "feff")).sum(0).cpu().numpy()
    t["chi"] = time.time() - t0
    return dict(k=kf, chi=chi, pot=pot, phase=phase, counts=counts, kept=kept, timing=t, input=P)
