"""feffgpu: FEFF 8.5L's EXAFS pipeline (potentials with self-consistency, phase shifts, the path finder and the path
stage) rewritten for NVIDIA GPUs in PyTorch and validated against FEFF 8.5L itself, with two tools built on it.

Front door::

    from feffgpu import run                      # feff.inp -> chi(k), with every intermediate result
    out = run("feff.inp", device="cuda")        # out["k"], out["chi"], out["pot"], out["phase"], out["kept"], out["timing"]

Lower level: ``read_feff_inp`` (the cards), ``potentials`` (``feffgpu.pot``), ``feffgpu.xsph.xsph`` (phases and matrix
elements), ``GPUPathFinder`` (paths), ``GPUPathStage`` (chi of batches of paths, differentiable in the coordinates),
``PhaseData`` / ``PotData`` (FEFF's pad files).

Subpackages (imported on demand): ``feffgpu.exafs`` (conventions and the forward model of whole configurations),
``feffgpu.io`` (structures and trajectories), ``feffgpu.analysis`` (pair statistics), ``feffgpu.parallel`` (devices and
shards), ``feffgpu.md`` (MD-EXAFS), ``feffgpu.rmc`` (RMC-EXAFS), ``feffgpu.mcp`` (the server for agents).
"""
from .chain import run
from .inp import read_feff_inp
from .pad import PhaseData, PotData, read_phase_pad, read_pot_pad, BOHR, HART
from .pot import potentials
from .pathfinder import GPUPathFinder
from .stage import GPUPathStage
from .cell import as_cell, wrap, min_width
from .compare import r_factor

__version__ = "0.2.0"
__all__ = ["run", "read_feff_inp", "PhaseData", "PotData", "read_phase_pad", "read_pot_pad", "BOHR", "HART", "potentials",
           "GPUPathFinder", "GPUPathStage", "as_cell", "wrap", "min_width", "r_factor"]  # the phase-shift stage: feffgpu.xsph.xsph
