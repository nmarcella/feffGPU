"""The configuration of an MD-EXAFS run: one TOML file, validated with pydantic.

Sections (the names of ``md_exafs`` 2.0 are kept where they still fit; ``docs/md/migration.md`` maps the old keys):

``[system]``       the trajectory, its reader and the output folder;
``[lattice]``      the periodic cell: static (NVT), read from every frame (NPT), or none (clusters);
``[atoms]``        the map from numeric atom types to elements (LAMMPS);
``[frames]``       which frames, how many absorbers per frame, the seed;
``[[edge]]``       one block per absorbing element: potentials, cutoff, number of legs, importance filter, k range;
``[average]``      which frames and path types make the averaged spectrum (computed from the cache);
``[compare]``      the measured spectrum, its window, S0^2 and the E0 shift (one table, or ``[[compare]]`` per edge);
``[rdf]``          radial distribution functions, peak fits, projections and bond angles;
``[convergence]``  the thermodynamic log of the dynamics;
``[dynamics]``     the optional runner that makes a trajectory with an ASE calculator (MACE or any other);
``[run]``          devices and batch sizes.

Relative paths are relative to the TOML file. The functions used by other tools: ``load_config``, ``json_schema``,
``example_configs``, ``validate_file``, ``output_directory``, ``to_toml``.
"""
from __future__ import annotations
import math
from importlib import resources
from pathlib import Path
from typing import Literal
from pydantic import Field, ValidationError, field_validator, model_validator
from ..config_tools import (ConfigError, ConfigModel, WindowModel, absolute_path, drop_none, read_toml, validation_messages,
                            to_toml as _to_toml)
from ..io.structure import Z_OF

__all__ = ["MDExafsConfig", "SystemConfig", "LatticeConfig", "FramesConfig", "EdgeConfig", "AverageConfig", "WindowConfig", "CompareConfig",
           "PeakConfig", "ProjectionConfig", "AngleConfig", "RDFConfig", "ConvergenceConfig", "DynamicsConfig", "RunConfig",
           "ConfigError", "load_config", "parse_config", "json_schema", "example_configs", "validate_file", "output_directory", "to_toml",
           "resolved_dict"]

EDGES = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")


_Model = ConfigModel


def _element(value, field="element"):
    v = str(value).strip()
    if v not in Z_OF or v == "X":
        raise ValueError(f"{v!r} is not an element symbol")
    return v


class SystemConfig(_Model):
    """The trajectory and the output folder."""
    name: str = Field("md_exafs", description="Label of the system, written in output headers and figure titles.")
    trajectory: str = Field(..., description="The trajectory file: any format that ASE reads frame by frame (extended xyz, CP2K xyz, "
                            "LAMMPS text dump, ASE .traj, VASP XDATCAR, ...), or a binary format with reader = \"mdanalysis\". "
                            "Relative to the configuration file.")
    format: str | None = Field(None, description="ASE's name of the file format, for example \"extxyz\", \"lammps-dump-text\", "
                               "\"vasp-xdatcar\" or \"traj\". Guessed from the file name and content when not given. With reader = "
                               "\"mdanalysis\": MDAnalysis's format name (optional).")
    reader: Literal["ase", "mdanalysis"] = Field("ase", description="Library that reads the trajectory: \"ase\" (default) or "
                                                 "\"mdanalysis\" (optional extra feffgpu[mda], for DCD, XTC, TRR, NetCDF and other "
                                                 "binary formats).")
    topology: str | None = Field(None, description="Topology file for reader = \"mdanalysis\" (for example a PDB or LAMMPS data file "
                                 "that names the atoms). Relative to the configuration file.")
    output: str = Field("mdexafs_out", description="Output folder of the run (the cache chi.h5, phase files, averages, figures). "
                        "Relative to the configuration file.")


class LatticeConfig(_Model):
    """The periodic cell. Give a static cell (lengths and angles, three vectors, or a matrix), or read the cell of every
    frame (dynamic), or declare clusters (pbc all false)."""
    dynamic: bool | None = Field(None, description="true: read the cell stored in every frame (NPT runs; extended xyz Lattice, LAMMPS "
                                 "box, XDATCAR header). false: use the static cell given here. Not given: static when a, b, c or matrix "
                                 "are given, otherwise the cell of every frame.")
    a: float | list[float] | None = Field(None, description="First cell length (A) or first lattice vector [x, y, z] (A).")
    b: float | list[float] | None = Field(None, description="Second cell length (A) or second lattice vector (A).")
    c: float | list[float] | None = Field(None, description="Third cell length (A) or third lattice vector (A).")
    alpha: float = Field(90.0, description="Angle between b and c (degrees), when a, b and c are lengths.")
    beta: float = Field(90.0, description="Angle between a and c (degrees), when a, b and c are lengths.")
    gamma: float = Field(90.0, description="Angle between a and b (degrees), when a, b and c are lengths.")
    matrix: list[list[float]] | None = Field(None, description="The three lattice vectors as rows of a 3 x 3 matrix (A); an "
                                             "alternative to a, b and c.")
    pbc: list[bool] = Field([True, True, True], description="Periodic boundaries along a, b, c: all true (a periodic system) or all "
                            "false (clusters, no cell). Mixed periodicity is not supported.")
    replicate: list[int] = Field([1, 1, 1], description="Repeat the cell this many times along a, b, c for the EXAFS calculation, "
                                 "needed when the cell is narrower than twice rpath. The absorbers are taken from the original cell only.")

    @field_validator("a", "b", "c")
    @classmethod
    def _length_or_vector(cls, v):
        if v is None:
            return v
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError("a lattice vector has three components")
            return [float(x) for x in v]
        if float(v) <= 0:
            raise ValueError("a cell length must be positive")
        return float(v)

    @field_validator("matrix")
    @classmethod
    def _matrix(cls, v):
        if v is not None and (len(v) != 3 or any(len(r) != 3 for r in v)):
            raise ValueError("the cell matrix must be 3 x 3 (three rows of three numbers)")
        return v

    @field_validator("pbc")
    @classmethod
    def _pbc(cls, v):
        if len(v) != 3:
            raise ValueError("pbc has three entries")
        if len(set(v)) != 1:
            raise ValueError("mixed periodicity is not supported: use all true (periodic) or all false (clusters)")
        return v

    @field_validator("replicate")
    @classmethod
    def _replicate(cls, v):
        if len(v) != 3 or any(int(x) < 1 for x in v):
            raise ValueError("replicate has three positive integers")
        return [int(x) for x in v]

    @model_validator(mode="after")
    def _consistent(self):
        given = [x is not None for x in (self.a, self.b, self.c)]
        if any(given) and not all(given):
            raise ValueError("give all of a, b and c, or none")
        static = all(given) or self.matrix is not None
        if all(given) and self.matrix is not None:
            raise ValueError("give a, b, c or matrix, not both")
        if all(given) and len({isinstance(x, list) for x in (self.a, self.b, self.c)}) != 1:
            raise ValueError("a, b and c must all be lengths or all be vectors")
        if self.dynamic is True and static:
            raise ValueError("dynamic = true reads the cell of every frame; remove a, b, c / matrix or set dynamic = false")
        if self.dynamic is False and not static and self.periodic:
            raise ValueError("dynamic = false needs a static cell: give a, b, c (lengths or vectors) or matrix")
        return self

    @property
    def periodic(self):
        """True for a periodic system."""
        return bool(self.pbc[0])

    @property
    def is_static(self):
        """True when a static cell is given."""
        return self.matrix is not None or self.a is not None

    def cell(self):
        """The static cell (3 x 3 numpy array) or None."""
        if not self.periodic or not self.is_static:
            return None
        from ..io.trajectory import static_cell
        return static_cell(self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.matrix)


class FramesConfig(_Model):
    """Which frames of the trajectory are used and how many absorbers per frame."""
    start: int = Field(0, ge=0, description="First frame (0 is the first frame of the file).")
    stop: int | None = Field(None, description="End of the frame range (excluded), as in Python's range; not given: the end of the file.")
    step: int = Field(1, ge=1, description="Use every step-th frame.")
    absorbers_per_frame: int | Literal["all"] = Field("all", description="\"all\": every atom of the absorbing element; a number: "
                                                      "a random subset of that size per frame, drawn from the seed and the frame index "
                                                      "only.")
    seed: int = Field(0, description="Seed of the random absorber subsets.")

    @field_validator("absorbers_per_frame")
    @classmethod
    def absolute_path(cls, v):
        if v != "all" and int(v) < 1:
            raise ValueError("absorbers_per_frame is \"all\" or a positive number")
        return v

    @model_validator(mode="after")
    def _range(self):
        if self.stop is not None and self.stop <= self.start:
            raise ValueError(f"stop ({self.stop}) must be larger than start ({self.start})")
        return self


class EdgeConfig(_Model):
    """One absorbing element: its potentials and the settings of its forward calculation."""
    name: str | None = Field(None, description="Name of the edge in outputs and in [average] and [compare]; S0^2 and the E0 shift are "
                             "shared between runs by this name. Default: the element.")
    element: str = Field(..., description="Absorbing element (symbol).")
    edge: Literal["K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"] = Field("K", description="Absorption edge.")
    potentials: Literal["reference", "per_frame", "file"] = Field(
        "reference", description="Where the potentials and phase shifts come from: \"reference\" (computed once from a reference cluster "
                                 "of one frame), \"per_frame\" (computed again every potentials_every frames) or \"file\" (the phase file "
                                 "given in phase).")
    phase: str | None = Field(None, description="Phase file for potentials = \"file\": a phase.pad from FEFF or a pickle from "
                              "`feffgpu phase`. Relative to the configuration file.")
    ipots: dict[str, int] = Field(default_factory=dict, description="Optional {element = potential index} for a phase file whose "
                                  "potentials are not found by atomic number (potentials = \"file\").")
    reference_frame: int | None = Field(None, description="Frame whose cluster defines the potentials (\"reference\"). Default: the "
                                        "first selected frame.")
    reference_atom: int | None = Field(None, description="Index of the absorbing atom at the center of the reference cluster. "
                                       "Default: the first atom of the element.")
    cluster_radius: float = Field(7.0, gt=0, description="Radius of the reference cluster (A); every element must have an atom in it "
                                  "besides the absorber.")
    scf_radius: float = Field(0.0, ge=0, description="Radius of FEFF's self-consistency loop for the potentials (A); 0 for no "
                              "self-consistency (FEFF's default). Self-consistent potentials take tens of seconds per cluster.")
    phase_kmax: float | None = Field(None, gt=0, description="Largest k of the phase grid (1/A). Default: kmax + 2.")
    cards: list[str] = Field(default_factory=list, description="Extra FEFF cards for the reference cluster's input, written as given "
                             "(for example \"EXCHANGE 0 0.0 0.0\"); only cards that feffgpu reads are accepted.")
    potentials_every: int = Field(10, ge=1, description="For potentials = \"per_frame\": new potentials every this many selected frames.")
    rpath: float = Field(6.0, gt=0, description="Path cutoff (A): half the total path length, FEFF's RPATH. Single scattering includes "
                         "every pair up to it.")
    nleg: int = Field(4, ge=2, le=8, description="Largest number of legs of the scattering paths: 2 (single scattering only) up to 8 "
                      "(FEFF's NLEG). Paths of five and more legs are stored as one path type for each number of legs.")
    criteria: float = Field(2.0, ge=0, description="FEFF's plane-wave importance filter for multiple-scattering paths (percent of the "
                            "shortest single-scattering path's importance); 0 keeps every path.")
    kmax: float = Field(20.0, gt=0, description="Largest k of the calculated spectrum (1/A); the grid is 0 to kmax in steps of 0.05.")
    pair_rmin: float = Field(1.0, gt=0, description="Start of the single-scattering distance table (A). Pairs closer than this are "
                             "counted and reported; lower it for systems with shorter bonds.")
    pair_dr: float = Field(0.002, gt=0, description="Step of the single-scattering distance table (A).")
    ms_bin_width: float = Field(0.5, ge=0, description="Width of the effective-length bins in which the multiple-scattering parts are "
                                "stored (A); distance limits of multiple-scattering path types act on these bins. 0: one bin.")

    @field_validator("element")
    @classmethod
    def _el(cls, v):
        return _element(v)

    @field_validator("ipots")
    @classmethod
    def _ipots(cls, v):
        for el, p in v.items():
            _element(el)
            if int(p) < 1:
                raise ValueError(f"{el}: a scatterer potential index is at least 1")
        return v

    @model_validator(mode="after")
    def _consistent(self):
        if self.name is None:
            self.name = self.element
        if self.potentials == "file" and not self.phase:
            raise ValueError("potentials = \"file\" needs the phase file in phase")
        if self.potentials != "file" and self.phase:
            raise ValueError(f"phase is used only with potentials = \"file\" (potentials is \"{self.potentials}\")")
        if self.pair_rmin >= self.rpath:
            raise ValueError("pair_rmin must be smaller than rpath")
        if self.phase_kmax is not None and self.phase_kmax < self.kmax:
            raise ValueError("phase_kmax must be at least kmax")
        return self

    @property
    def phase_kmax_resolved(self):
        """The phase grid's largest k (1/A): phase_kmax, or kmax + 2."""
        return float(self.phase_kmax) if self.phase_kmax is not None else float(self.kmax) + 2.0


class AverageConfig(_Model):
    """The averaged spectrum: which edges, frames and path types. Computed from the cache (md-exafs average)."""
    edges: list[str] = Field(default_factory=list, description="Edges to average (names); default all.")
    path_types: list[str] = Field(default_factory=list, description="Path types to include, for example [\"Zn-C\", \"Zn-C-C\"]; "
                                  "default every path (the total spectrum). Each edge uses the types that start with its absorbing "
                                  "element; an edge with none of them keeps its total spectrum.")
    max_distance: float | list[float] | None = Field(None, description="Largest effective path length (A) of each path type in "
                                                     "path_types (one number for all, or one per type). Single scattering: any value; "
                                                     "multiple scattering: a multiple of the edge's ms_bin_width.")
    start: int | None = Field(None, ge=0, description="First frame of the average (trajectory index); default the first in the cache.")
    stop: int | None = Field(None, description="End of the frames of the average (excluded); default after the last in the cache.")
    step: int = Field(1, ge=1, description="Use every step-th frame of the cache's frames between start and stop.")
    blocks: int = Field(10, ge=2, description="Number of blocks for the block-averaged standard error and the convergence curve.")

    @model_validator(mode="after")
    def _lengths(self):
        if isinstance(self.max_distance, list):
            if not self.path_types:
                raise ValueError("max_distance as a list needs path_types of the same length")
            if len(self.max_distance) != len(self.path_types):
                raise ValueError(f"max_distance has {len(self.max_distance)} entries but path_types has {len(self.path_types)}")
        if self.max_distance is not None:
            vals = self.max_distance if isinstance(self.max_distance, list) else [self.max_distance]
            if any(float(v) <= 0 for v in vals):
                raise ValueError("max_distance values must be positive")
        return self


class WindowConfig(WindowModel):
    """The window of a comparison, in xraylarch's terms (``feffgpu.config_tools.WindowModel`` with default ranges)."""
    kmin: float = Field(3.0, ge=0, description="Center of the low-k sill of the Hanning window (1/A).")
    kmax: float = Field(14.0, gt=0, description="Center of the high-k sill of the Hanning window (1/A).")
    rmin: float = Field(1.0, ge=0, description="Start of the R range of the R-factor (A).")
    rmax: float = Field(6.0, gt=0, description="End of the R range of the R-factor (A).")


class CompareConfig(_Model):
    """A measured spectrum to compare the averaged spectrum with."""
    edge: str | None = Field(None, description="Name of the edge this spectrum belongs to; may be left out when the run has one edge.")
    chi: str = Field(..., description="Measured chi(k): a text file with k (1/A) and chi columns (Athena and xraylarch .chi files are "
                     "read as they are). Relative to the configuration file.")
    k_column: int = Field(0, ge=0, description="Zero-based column of k.")
    chi_column: int = Field(1, ge=0, description="Zero-based column of chi.")
    window: WindowConfig = Field(default_factory=WindowConfig, description="The k window, k weight and R range of the comparison.")
    fit_s02: bool = Field(True, description="Fit S0^2 (on the model).")
    fit_e0: bool = Field(True, description="Fit the E0 shift (on the model).")
    s02: float = Field(1.0, gt=0, description="S0^2: the start value when fitted, else the held value.")
    e0: float = Field(0.0, description="E0 shift (eV): the start value when fitted, else the held value.")
    s02_bounds: list[float] = Field([0.2, 5.0], description="Allowed range of S0^2 (a safety range, not a prior).")
    e0_bounds: list[float] = Field([-15.0, 15.0], description="Allowed range of the E0 shift (eV).")
    label: str | None = Field(None, description="Label of the spectrum in tables and figures; default the file name.")

    @model_validator(mode="after")
    def _bounds(self):
        for name in ("s02_bounds", "e0_bounds"):
            b = getattr(self, name)
            if len(b) != 2 or b[0] >= b[1]:
                raise ValueError(f"{name} is [lower, upper] with lower < upper")
        if not (self.s02_bounds[0] > 0):
            raise ValueError("s02_bounds must be positive")
        return self


class PeakConfig(_Model):
    """A peak of a partial RDF to characterize."""
    center: str = Field(..., description="Element of the center atoms.")
    neighbor: str = Field(..., description="Element of the neighbor atoms.")
    r_min: float = Field(..., ge=0, description="Lower end of the peak's distance range (A).")
    r_max: float = Field(..., gt=0, description="Upper end of the peak's distance range (A).")
    fit: Literal["gaussian", "skewed", "both", "none"] = Field("both", description="Fitted forms besides the plain integration: "
                                                               "\"gaussian\", \"skewed\" (skew-normal), \"both\" or \"none\".")

    @field_validator("center", "neighbor")
    @classmethod
    def _el(cls, v):
        return _element(v)

    @model_validator(mode="after")
    def _order(self):
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be larger than r_min")
        return self


class ProjectionConfig(_Model):
    """An RDF of the bonds along a crystal direction."""
    center: str = Field(..., description="Element of the center atoms.")
    neighbor: str = Field(..., description="Element of the neighbor atoms.")
    direction: list[float] = Field(..., description="Direction [x, y, z] in Cartesian coordinates (for a cubic cell, the crystal "
                                   "direction [h, k, l]).")
    angle: float = Field(30.0, gt=0, le=90, description="A bond counts when its angle to the direction (or to its opposite) is at most "
                         "this (degrees).")
    symmetry: bool = Field(False, description="Also count bonds along every cubic-symmetry equivalent of the direction.")
    r_min: float | None = Field(None, ge=0, description="Lower end of the distance range of the reported numbers (A); default 0.")
    r_max: float | None = Field(None, gt=0, description="Upper end of the distance range of the reported numbers (A); default the RDF's "
                                                         "r_max.")

    @field_validator("center", "neighbor")
    @classmethod
    def _el(cls, v):
        return _element(v)

    @field_validator("direction")
    @classmethod
    def _dir(cls, v):
        if len(v) != 3 or math.sqrt(sum(float(x) ** 2 for x in v)) == 0:
            raise ValueError("direction is a non-zero vector of three numbers")
        return v


class AngleConfig(_Model):
    """A bond-angle distribution B-A-C."""
    center: str = Field(..., description="Element of the center atom A.")
    neighbors: list[str] = Field(..., description="Elements of the two neighbors [B, C] (they may be the same).")
    ranges: list[list[float]] = Field(..., description="Distance ranges [[r_min, r_max], [r_min, r_max]] (A) of A-B and A-C that "
                                      "define a neighbor.")
    bin_width: float = Field(1.0, gt=0, description="Bin width of the angle histogram (degrees).")

    @field_validator("center")
    @classmethod
    def _el(cls, v):
        return _element(v)

    @field_validator("neighbors")
    @classmethod
    def _nb(cls, v):
        if len(v) != 2:
            raise ValueError("neighbors has two elements")
        return [_element(x) for x in v]

    @field_validator("ranges")
    @classmethod
    def _ranges(cls, v):
        if len(v) != 2 or any(len(r) != 2 or r[1] <= r[0] for r in v):
            raise ValueError("ranges is [[r_min, r_max], [r_min, r_max]] with r_min < r_max")
        return v


class RDFConfig(_Model):
    """Radial distribution functions of the trajectory (md-exafs rdf)."""
    r_max: float = Field(8.0, gt=0, description="Largest distance of the RDFs (A). Any value: periodic images are included when it "
                         "exceeds half the cell width.")
    bin_width: float = Field(0.01, gt=0, description="Bin width (A).")
    pairs: list[list[str]] = Field(default_factory=list, description="Center-neighbor element pairs, for example [[\"Zn\", \"C\"]]; "
                                   "default every pair of the system's elements.")
    start: int | None = Field(None, ge=0, description="First frame; default [frames] start.")
    stop: int | None = Field(None, description="End of the frame range (excluded); default [frames] stop.")
    step: int | None = Field(None, ge=1, description="Frame step; default [frames] step.")
    device: str | None = Field(None, description="Torch device of the histogram accumulation; default the first of [run] devices "
                               "(\"cpu\" works too).")
    peak: list[PeakConfig] = Field(default_factory=list, description="Peaks to characterize ([[rdf.peak]] blocks).")
    projection: list[ProjectionConfig] = Field(default_factory=list, description="RDFs along crystal directions ([[rdf.projection]] "
                                                                                 "blocks).")
    angle: list[AngleConfig] = Field(default_factory=list, description="Bond-angle distributions ([[rdf.angle]] blocks).")

    @field_validator("pairs")
    @classmethod
    def _pairs(cls, v):
        for p in v:
            if len(p) != 2:
                raise ValueError("each pair is [center, neighbor]")
            _element(p[0]), _element(p[1])
        return v


class ConvergenceConfig(_Model):
    """The thermodynamic log of the dynamics (md-exafs convergence)."""
    log: str | None = Field(None, description="Log file: a CP2K .ener file, an ASE molecular-dynamics log, or a CSV file with a header "
                            "row. Relative to the configuration file. Default: the log written by [dynamics].")
    format: Literal["auto", "cp2k", "ase", "csv"] = Field("auto", description="Format of the log; \"auto\" guesses it from the content.")
    skip: float = Field(0.0, ge=0, description="Leading part of the log to leave out of the statistics (equilibration), in the log's "
                        "time unit (fs for CP2K, ps for ASE logs, the time column of a CSV).")
    blocks: int = Field(10, ge=2, description="Number of blocks for block averages.")


class DynamicsConfig(_Model):
    """The optional runner: NVT Langevin dynamics with an ASE calculator, written to [system] trajectory."""
    structure: str = Field(..., description="Starting structure: any file that ase.io reads (CIF, POSCAR, extended xyz, ...). Relative "
                           "to the configuration file.")
    supercell: list[int] = Field([1, 1, 1], description="Repeat the starting structure this many times along a, b, c.")
    calculator: str = Field("mace_mp", description="\"mace_mp\" (a MACE foundation model, extra feffgpu[mace]), \"emt\" (ASE's "
                            "effective-medium potential for a few metals, for tests), or \"module:function\" returning any ASE calculator.")
    model: str = Field("medium", description="Model name or file for calculator = \"mace_mp\".")
    calculator_options: dict = Field(default_factory=dict, description="Extra keyword arguments for the calculator's constructor.")
    temperature: float = Field(300.0, gt=0, description="Thermostat temperature (K).")
    timestep: float = Field(2.0, gt=0, description="Time step (fs).")
    friction: float = Field(0.01, gt=0, description="Langevin friction (1/fs).")
    steps: int = Field(10000, ge=1, description="Production steps (after equilibration).")
    equilibration: int = Field(1000, ge=0, description="Steps before frames are written.")
    save_every: int = Field(10, ge=1, description="Write a frame every this many steps.")
    log_every: int = Field(10, ge=1, description="Write a line of the thermodynamic log every this many steps.")
    seed: int = Field(0, description="Seed of the initial velocities and of the thermostat.")
    log: str = Field("md.log", description="ASE molecular-dynamics log written by the runner. Relative to the configuration file.")

    @field_validator("supercell")
    @classmethod
    def _sc(cls, v):
        if len(v) != 3 or any(int(x) < 1 for x in v):
            raise ValueError("supercell has three positive integers")
        return [int(x) for x in v]


class RunConfig(_Model):
    """Devices and batch sizes."""
    devices: str | int | list[int] = Field("cuda", description="Devices: \"cuda\", \"cuda:1\", \"cpu\", \"all\", a count or a list of "
                                           "indices. This version runs on the first; split a run over several GPUs with shards.")
    absorber_batch: int = Field(0, ge=0, description="Absorbers per batch of the distance histogram; 0 chooses it from the free GPU "
                                                     "memory.")
    path_chunk: int = Field(0, ge=0, description="Paths per batch of the path stage; 0 chooses it from the free GPU memory.")
    ms_group: int = Field(64, ge=1, description="Absorbers whose multiple-scattering paths are enumerated together.")
    frames_per_block: int = Field(8, ge=1, description="Frames computed between two writes to the cache.")


class MDExafsConfig(_Model):
    """An MD-EXAFS configuration file."""
    system: SystemConfig = Field(..., description="The trajectory and the output folder.")
    lattice: LatticeConfig = Field(default_factory=LatticeConfig, description="The periodic cell.")
    atoms: dict[str, str] = Field(default_factory=dict, description="Numeric atom type = element, for files with numeric types "
                                  "(LAMMPS): 1 = \"Zn\". The form of md_exafs 2.0 (Zn = 1) is accepted too.")
    frames: FramesConfig = Field(default_factory=FramesConfig, description="Frames and absorbers.")
    edge: list[EdgeConfig] = Field(..., min_length=1, description="One [[edge]] block per absorbing element.")
    average: AverageConfig = Field(default_factory=AverageConfig, description="The averaged spectrum.")
    compare: list[CompareConfig] = Field(default_factory=list, description="Measured spectra ([compare] or [[compare]]).")
    rdf: RDFConfig | None = Field(None, description="Radial distribution functions.")
    convergence: ConvergenceConfig | None = Field(None, description="The thermodynamic log.")
    dynamics: DynamicsConfig | None = Field(None, description="The optional dynamics runner.")
    run: RunConfig = Field(default_factory=RunConfig, description="Devices and batch sizes.")

    @field_validator("atoms", mode="before")
    @classmethod
    def _type_map(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("[atoms] is a table of numeric type = element")
        out = {}
        for key, value in v.items():
            k, val = str(key).strip(), value
            if isinstance(val, int) and not isinstance(val, bool) and k in Z_OF:      # md_exafs 2.0: element = type
                k, val = str(val), k
            if not k.lstrip("-").isdigit():
                raise ValueError(f"type {k!r}: the keys of [atoms] are numeric atom types (1 = \"Zn\")")
            out[k] = _element(val)
        return out

    @field_validator("compare", mode="before")
    @classmethod
    def _one_or_many(cls, v):
        if v is None:
            return []
        return [v] if isinstance(v, dict) else v

    @model_validator(mode="after")
    def _cross(self):
        names = [e.name for e in self.edge]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"edge names must be unique (repeated: {dup}); give each [[edge]] its own name")
        for c in self.compare:
            if c.edge is None:
                if len(names) != 1:
                    raise ValueError("[compare] needs edge = \"<name>\" when the run has several edges")
                c.edge = names[0]
            elif c.edge not in names:
                raise ValueError(f"[compare] edge {c.edge!r} is not an edge of this run ({names})")
        bad = [e for e in self.average.edges if e not in names]
        if bad:
            raise ValueError(f"[average] edges {bad} are not edges of this run ({names})")
        return self

    # -- convenience --------------------------------------------------------------------------------------------
    def edge_by_name(self, name):
        """The [[edge]] block with this name."""
        for e in self.edge:
            if e.name == name:
                return e
        raise KeyError(f"no edge named {name!r}")

    @property
    def output_path(self):
        """The output folder as a Path (absolute after ``load_config``)."""
        return Path(self.system.output)


# -- loading and validation ------------------------------------------------------------------------------------------
def parse_config(data, base=None):
    """Validate a configuration dictionary (as read from TOML) and make its relative paths absolute.

    Parameters
    ----------
    data : dict
    base : str or Path, optional
        Directory that relative paths refer to (the TOML file's folder); None leaves paths as they are.

    Returns
    -------
    MDExafsConfig

    Raises
    ------
    ConfigError
        With one message per problem.
    """
    try:
        cfg = MDExafsConfig.model_validate(data)
    except ValidationError as err:
        raise ConfigError(validation_messages(err)) from None
    if base is not None:
        _resolve_paths(cfg, Path(base))
    return cfg


def _resolve_paths(cfg, base):
    cfg.system.trajectory = absolute_path(cfg.system.trajectory, base)
    cfg.system.topology = absolute_path(cfg.system.topology, base)
    cfg.system.output = absolute_path(cfg.system.output, base)
    for e in cfg.edge:
        e.phase = absolute_path(e.phase, base)
    for c in cfg.compare:
        c.chi = absolute_path(c.chi, base)
    if cfg.convergence is not None:
        cfg.convergence.log = absolute_path(cfg.convergence.log, base)
    if cfg.dynamics is not None:
        cfg.dynamics.structure = absolute_path(cfg.dynamics.structure, base)
        cfg.dynamics.log = absolute_path(cfg.dynamics.log, base)
    return cfg


def load_config(path):
    """Read and validate a TOML configuration file; relative paths become absolute (relative to the file).

    Raises
    ------
    ConfigError
        With one message per problem, each starting with the dotted path of the field.
    """
    p = Path(path).resolve()
    return parse_config(read_toml(p), base=p.parent)


def json_schema():
    """The JSON Schema of an MD-EXAFS configuration (pydantic's ``model_json_schema`` of ``MDExafsConfig``)."""
    return MDExafsConfig.model_json_schema()


def example_configs():
    """The example configurations that ship with the package: ``{name: TOML text}`` (``feffgpu/md/templates``)."""
    out = {}
    folder = resources.files("feffgpu.md") / "templates"
    for item in sorted(folder.iterdir(), key=lambda t: t.name):
        if item.name.endswith(".toml"):
            out[item.name[:-5]] = item.read_text()
    return out


def validate_file(path):
    """Check a configuration file without running anything and without a GPU.

    Returns
    -------
    list of str
        Empty when the file is valid; otherwise one message per problem, each starting with the dotted path of the
        field (for example ``edge.0.rpath: ...``). Input files that the configuration names and that do not exist
        (trajectory, topology, phase files, measured spectra, the starting structure of [dynamics]) are problems too.
        A trajectory or log that [dynamics] will write is not required to exist. Never raises for a user error.
    """
    try:
        cfg = load_config(path)
    except ConfigError as err:
        return list(err.problems)
    except Exception as err:                                                    # anything unexpected is still a message
        return [f"file: {type(err).__name__}: {err}"]
    problems = []

    def need(value, where):
        if value and not Path(value).exists():
            problems.append(f"{where}: no such file: {value}")

    if cfg.dynamics is None:
        need(cfg.system.trajectory, "system.trajectory")
    else:
        need(cfg.dynamics.structure, "dynamics.structure")
    need(cfg.system.topology, "system.topology")
    for i, e in enumerate(cfg.edge):
        need(e.phase, f"edge.{i}.phase")
    for i, c in enumerate(cfg.compare):
        need(c.chi, f"compare.{i}.chi")
    if cfg.convergence is not None and cfg.convergence.log and cfg.dynamics is None:
        need(cfg.convergence.log, "convergence.log")
    if cfg.system.reader == "mdanalysis":
        try:
            import MDAnalysis  # noqa: F401
        except ImportError:
            problems.append("system.reader: \"mdanalysis\" needs MDAnalysis, which is not installed (pip install 'feffgpu[mda]')")
    return problems


def output_directory(path):
    """The absolute output folder that a run of this configuration file writes to (``[system] output``, relative to
    the file; default ``mdexafs_out``)."""
    p = Path(path).resolve()
    try:
        data = read_toml(p)
    except ConfigError:
        data = {}
    out = ((data.get("system") or {}).get("output")) or "mdexafs_out"
    q = Path(str(out)).expanduser()
    return q if q.is_absolute() else (p.parent / q).resolve()


# -- writing ---------------------------------------------------------------------------------------------------------
def resolved_dict(cfg):
    """The configuration as a plain dictionary with every default filled in (None values dropped)."""
    return drop_none(cfg.model_dump(mode="json"))


def to_toml(data, header=""):
    """A configuration dictionary as TOML text (``feffgpu.config_tools.to_toml`` with this tool's inline tables).

    Parameters
    ----------
    data : dict
    header : str
        Comment lines written first (without the leading ``#``).

    Returns
    -------
    str
    """
    return _to_toml(data, header=header, inline_keys=("window", "ipots", "calculator_options"))
