"""The configuration of an RMC-EXAFS run: one TOML file, validated with pydantic.

Every key is described in the models below; ``docs/rmc/configuration.md`` is generated from these descriptions
(``docs/rmc/_make_config_reference.py``). Relative paths in a TOML file are relative to that file. A run writes the
fully resolved configuration (absolute paths, every default filled in) to ``config.resolved.toml`` in its output
directory, which can be run again as it is.

Functions for programs and agents: ``load_config``, ``json_schema``, ``example_configs``, ``validate_file``,
``output_directory``. The fully resolved configuration is written with ``feffgpu.config_tools.to_toml``.
"""
from __future__ import annotations
import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import Literal, Optional, Union
from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator
from ..config_tools import ConfigModel, WindowModel, absolute_path, validation_messages

__all__ = ["RunSettings", "WindowSettings", "EdgeSettings", "SampleSettings", "AmplitudeSettings", "SearchSettings", "HopSettings",
           "MoveSettings", "ConstraintSettings", "RefineSettings", "RmcConfig", "load_config", "json_schema", "example_configs",
           "validate_file", "output_directory"]

LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


class _Model(ConfigModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class RunSettings(_Model):
    """[run]: where the outputs go, the seed and the devices."""
    output: str = Field(..., description="Directory of the outputs. Every sample writes into a subdirectory named by its label; "
                                         "run.json and config.resolved.toml go into the directory itself.")
    seed: int = Field(1, description="Seed of the random numbers. Every sample draws from its own generator, seeded with seed + 11, in a "
                                     "fixed order: the same configuration and seed give the same run.")
    devices: Union[str, int, list[int]] = Field("cuda", description="The device(s): 'cuda', 'cuda:1', 'cpu', 'all', a number of GPUs, or a list of GPU "
                                                                    "indices (feffgpu.parallel.resolve_devices). This version runs on the first of them.")
    save_every: int = Field(25, ge=1, description="Generations between checkpoints (checkpoint.npz and history.json of every sample).")


WindowSettings = WindowModel          # the window of a comparison (kmin, kmax, dk, kweight, rmin, rmax), shared with MD-EXAFS


class EdgeSettings(_Model):
    """[[sample.edge]]: one measured spectrum of a sample and its forward model."""
    name: str = Field(..., min_length=1, description="Label of the spectrum. Spectra with the same name in several samples (the same edge of the "
                                                     "same compound) share S0^2 and the E0 shift.")
    absorber: str = Field(..., min_length=1, description="The absorbing element (its atoms are the absorbers).")
    chi: str = Field(..., description="The measured chi(k): a text file with k (1/A) and chi in columns; comment lines are skipped, so Athena's "
                                      "and xraylarch's .chi files are read as they are. Never divided by S0^2.")
    chi_columns: list[int] = Field([0, 1], min_length=2, max_length=2, description="Zero-based columns of k and chi in the chi file.")
    phase: str = Field(..., description="The phase file of this edge: FEFF 8.5L's phase.pad, or a pickle written by `feffgpu phase` (.pkl). "
                                        "Its potential 0 is the absorber.")
    potentials: dict[str, int] = Field(default_factory=dict, description="{element: scattering potential index} in the phase file (1 or more), "
                                                                         "as in the POTENTIALS block of the feff.inp it was made from. May be left "
                                                                         "out when the phase file has a single scattering potential.")
    window: WindowSettings = Field(..., description="The k window, k weight and R window: { kmin, kmax, dk, kweight, rmin, rmax }.")
    rpath: float = Field(6.0, gt=0, description="Pair and path cutoff (A): FEFF's RPATH, EvAX's R_max_for_FEFF. The periodic cell must be at least "
                                                "twice as wide.")
    nleg: int = Field(4, ge=2, le=8, description="Largest number of legs of the scattering paths: 2 (single scattering only) up to 8 "
                      "[N_legs; EvAX's default is 8].")
    criteria: float = Field(2.5, ge=0, description="Importance filter of the multiple-scattering paths, in percent; what it measures is set "
                                                   "by criteria_rule (FEFF's CRITERIA, EvAX's FEFF_criteria).")
    criteria_rule: Literal["feff", "evax"] = Field("feff", description="'feff': every path of each structure as it is, judged alone by FEFF's "
                                                   "plane-wave importance relative to the shortest single-scattering path (what FEFF's path finder "
                                                   "does on a structure without symmetry); the paths are found again every paths_every generations. "
                                                   "'evax': EvAX's rule: the paths are chosen once on the start lattice; paths of like "
                                                   "geometry form groups as in EvAX's path finder (see group_tolerance), and a group is kept when "
                                                   "the size of its signal on the lattice is more than criteria percent of the size of the "
                                                   "lattice's whole spectrum. Not with exchanges or hops.")
    group_tolerance: float = Field(0.5, gt=0, description="With criteria_rule = 'evax': two paths belong to one group when their atoms, one by "
                                                          "one, lie within this distance (A) of each other in EvAX's path coordinates "
                                                          "[Pathfinder_robustness].")
    multiple_scattering: bool = Field(True, description="False: single scattering only.")
    objective: Literal["r_factor", "wavelet", "cauchy_wavelet"] = Field("r_factor", description="The misfit of this spectrum: xraylarch's R-factor "
                                                                                                "on chi(R), EvAX's wavelet residual, or xraylarch's "
                                                                                                "Cauchy wavelet residual.")
    wavelet_r0: Optional[float] = Field(None, gt=0, description="EvAX's R0_for_WT (A) of the 'wavelet' objective; default the middle of the R window.")
    pair_min: float = Field(1.5, gt=0, description="Shortest pair distance that can enter single scattering (A): the start of the pair tables.")
    pair_windows: dict[str, list[float]] = Field(default_factory=dict, description="Optional {element: [r0, r1, ...]} (A): the pairs with that element are "
                                                                                   "counted from the first to the last boundary (default pair_min to "
                                                                                   "rpath). Inner boundaries split the sum without changing the spectrum.")
    ignore: list[str] = Field(default_factory=list, description="Elements the forward model of this spectrum does not see (they neither absorb "
                                                                "nor scatter). An atom that every spectrum ignores does not move.")


class SampleSettings(_Model):
    """[[sample]]: one sample: a start structure and its measured spectra."""
    label: str = Field(..., description="Name of the sample (letters, digits, '.', '_', '-'): its output subdirectory.")
    structure: str = Field(..., description="The start structure (POSCAR, CIF, extended xyz, or any file ASE reads), with its periodic cell. The "
                                            "position of every atom is its lattice site.")
    format: Optional[str] = Field(None, description="ASE's format name of the structure file, if its name does not tell.")
    supercell: list[int] = Field([1, 1, 1], min_length=3, max_length=3, description="Replication of the structure along its three cell vectors.")
    hop_sites: Optional[str] = Field(None, description="The sites of the hop sublattice of this sample ('fluorite' or a file); default "
                                                       "moves.hops.sites.")
    edge: list[EdgeSettings] = Field(..., min_length=1, description="The measured spectra of this sample, one [[sample.edge]] block each.")

    @field_validator("label")
    @classmethod
    def _label(cls, v):
        if not LABEL.match(v):
            raise ValueError(f"label {v!r} may contain only letters, digits, '.', '_' and '-'")
        return v

    @field_validator("supercell")
    @classmethod
    def _supercell(cls, v):
        if any(int(x) < 1 for x in v):
            raise ValueError(f"supercell must be three positive integers (got {v})")
        return v


class AmplitudeSettings(_Model):
    """[amplitude]: S0^2 and the E0 shift, one value per spectrum name, shared by the samples.

    They are declared and held (``mode = "fixed"``, the default): S0^2 trades with the disorder of the structure, so a
    fit of the structure cannot determine it. ``mode = "fit"`` is for a fit that includes a reference material of known
    structure whose spectra share S0^2 and the E0 shift with the sample's; ``mode = "from_run"`` holds the values that
    an earlier run (of the reference) found."""
    mode: Literal["fixed", "fit", "from_run"] = Field("fixed", description="'fixed': held at the declared s02 and de0 (EvAX's rule). 'from_run': held "
                                                                           "at the shared values of the run named by from_run (for example a fit of a "
                                                                           "reference material). 'fit' (`rmc-exafs run` only): refitted jointly over "
                                                                           "all samples on the best states; meant for a fit that includes a reference "
                                                                           "material of known structure, because S0^2 trades with the disorder.")
    s02: Union[float, dict[str, float]] = Field(1.0, description="S0^2: the held value ('fixed') or the start value ('fit'); one number for every "
                                                                 "spectrum or {name: value}.")
    de0: Union[float, dict[str, float]] = Field(0.0, description="The E0 shift (eV): the held or start value; one number or {name: value}. The "
                                                                 "pair tables and paths are evaluated at it; a fit changes it by interpolation.")
    s02_bounds: list[float] = Field([0.5, 1.5], min_length=2, max_length=2, description="Allowed range of S0^2 in the fits: a safety range, not a "
                                                                                       "prior. A value on a bound is reported.")
    de0_bounds: list[float] = Field([-8.0, 8.0], min_length=2, max_length=2, description="Allowed change of the E0 shift in the fits (eV), relative "
                                                                                        "to the start value.")
    refit_every: int = Field(10, ge=1, description="Generations between refits during the search ('fit').")
    refit_from: int = Field(101, ge=0, description="First generation of a refit during the search ('fit'); 0: no refit during the search. A "
                                                   "final refit on the best states is always made in mode 'fit'.")
    fit_iterations: int = Field(60, ge=1, description="L-BFGS iterations of a refit.")
    from_run: Optional[str] = Field(None, description="A finished run of this command (its output directory). Mode 'from_run': its shared S0^2 "
                                                      "and E0 shifts are held. Mode 'fit': they are the start values.")
    noise_floor: float = Field(0.0, ge=0, description="Least misfit taken as the noise level in the intervals of S0^2 and the E0 shift (the "
                                                      "misfit at the optimum is used when it is larger).")

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "from_run" and not self.from_run:
            raise ValueError("mode 'from_run' needs from_run (the output directory of an earlier run)")
        if self.mode == "fixed" and self.from_run:
            raise ValueError("mode 'fixed' holds s02 and de0; from_run belongs to the modes 'from_run' (held) and 'fit' (start values)")
        lo, hi = self.s02_bounds
        if not 0 < lo < hi:
            raise ValueError(f"s02_bounds must be two increasing positive numbers (got {self.s02_bounds})")
        if not self.de0_bounds[0] < self.de0_bounds[1]:
            raise ValueError(f"de0_bounds must be increasing (got {self.de0_bounds})")
        return self


class SearchSettings(_Model):
    """[search]: the population search under EvAX's rules (EvAX's keyword in brackets)."""
    states: int = Field(32, ge=2, description="Number of states of the population [Number_of_states]; even (EvAX breeds in pairs).")
    generations: int = Field(..., ge=1, description="Number of generations of this run [Stop_after].")
    cells_per_state: int = Field(1, ge=1, description="Copies of the cell in one state, whose spectra are averaged (EvAX: 1).")
    step: float = Field(0.005, gt=0, description="The random move: a uniform step within +/- step on every coordinate of every atom (A) "
                                                 "[Maximal_step_length].")
    acceptance: float = Field(0.8, description="Probability of accepting a worse move of mean size at the start, cooled linearly to zero at "
                                               "froze_in [Acceptance_rate]; 0 or less: worse moves are never accepted.")
    froze_in: int = Field(1500, ge=1, description="Generation count (over all continued runs) at which worse moves stop being accepted [Froze_in].")
    means_from: int = Field(0, ge=0, description="Generation count at which the running mean of the size of the misfit changes (the acceptance) starts again "
                                                 "from zero moves; 0: never (EvAX's rule).")
    weights_means_from: int = Field(0, ge=0, description="Generation count at which the running mean of the spectrum weights starts again; 0: never "
                                                         "(EvAX's rule).")
    adapt_weights: bool = Field(True, description="Adapt the weights of the spectra of a sample during the run [adjust_spectra_weights]; "
                                                  "false: equal weights.")
    gradient_steps: int = Field(0, ge=0, description="FIRE steps on the gradient of the misfit after every random move (an addition to EvAX; "
                                                     "0: EvAX's rule).")
    gradient_force: float = Field(1.0, gt=0, description="Largest force (eV/A) of the misfit on an atom of the start structure: sets the size of "
                                                         "the gradient steps. A continued run keeps the scale of its first run.")
    start_kick: float = Field(0.01, ge=0, description="Random displacement that makes the start states differ (A) [Initial_displacement].")
    start_distribution: Literal["gaussian", "uniform"] = Field("uniform", description="'uniform': every coordinate uniform in "
                                                                                       "[-start_kick, start_kick] / sqrt(2) (EvAX's start); "
                                                                                       "'gaussian': normal with standard deviation start_kick.")
    paths_every: int = Field(10, ge=1, description="Generations between new searches of every state's multiple-scattering paths (between them "
                                                   "the paths follow the atoms).")
    generation_offset: Optional[int] = Field(None, ge=0, description="Generations done before this run (default: 0, or those of continue_from).")
    continue_from: Optional[str] = Field(None, description="A finished run of this command (its output directory): its population, S0^2, E0 "
                                                           "shifts, spectrum weights, running means and generation count continue here.")

    @field_validator("states")
    @classmethod
    def _even(cls, v):
        if v % 2:
            raise ValueError(f"states must be even, EvAX breeds the states in pairs (got {v})")
        return v


class HopSettings(_Model):
    """[moves.hops]: hops of the atoms of one element onto vacant sites of their sublattice."""
    element: str = Field(..., description="The element that hops.")
    sites: str = Field(..., description="All sites of its sublattice, occupied or vacant: 'fluorite' (the built-in generator: the anion sites of "
                                        "a fluorite structure) or a file (three columns x y z in A, or a structure file whose atoms are the sites).")
    radius: Optional[float] = Field(None, gt=0, description="Two sites are neighbors within this distance (A); default 1.1 times the nearest site "
                                                            "distance (the built-in generator: the six nearest sites).")
    source: Literal["neighbors", "any"] = Field("neighbors", description="Where the hopping atom comes from: a neighboring site of the vacant site, "
                                                                         "or any occupied site.")
    probability: float = Field(0.5, ge=0, le=1, description="Probability that a state proposes hops in a generation.")
    count: int = Field(1, ge=1, description="Hops per cell of a state that proposes them.")
    jitter: float = Field(0.05, ge=0, description="Normal displacement of a hopped atom from its new site, per coordinate (A).")


class MoveSettings(_Model):
    """[moves]: rearrangements in addition to EvAX's random move."""
    exchange: list[str] = Field(default_factory=list, description="Two elements whose atoms exchange positions and sites; empty: no exchanges. "
                                                                  "A sample without both elements makes none.")
    exchange_probability: float = Field(0.5, ge=0, le=1, description="Probability that a state proposes exchanges in a generation.")
    exchange_count: int = Field(1, ge=1, description="Exchanges per cell of a state that proposes them.")
    hops: Optional[HopSettings] = Field(None, description="Hops onto vacant sites (see [moves.hops]); absent: no hops.")
    from_generation: int = Field(1, ge=1, description="First generation of this run with exchanges and hops.")

    @field_validator("exchange")
    @classmethod
    def _pair(cls, v):
        if v and (len(v) != 2 or v[0] == v[1]):
            raise ValueError(f"exchange takes two different elements (got {v})")
        return v


class ConstraintSettings(_Model):
    """[constraints]: the displacement limit and the closest distances."""
    max_displacement: float = Field(0.4, ge=0, description="Every atom stays within this distance (A) of its site of the fixed start lattice "
                                                           "[Potential_type box, Maximal_displacement]; 0: no limit.")
    box_rule: Literal["auto", "project", "halve"] = Field("auto", description="How the limit acts: 'halve' (EvAX: a move that leaves the limit is "
                                                                              "halved, at most 100 times, then refused), 'project' (an atom "
                                                                              "outside is put back on the limit; needed with gradient steps), "
                                                                              "'auto' (halve without gradient steps, project with them).")
    closest_distance: dict[str, float] = Field(default_factory=dict, description="{\"A-B\": distance}: the shortest distance (A) between the "
                                                                                 "elements A and B that exists in the chemistry [Potential_type "
                                                                                 "hardsphere]. Pairs not listed have no limit.")
    closest_tolerance: float = Field(0.0, ge=0, description="A configuration is refused when a pair is closer than its limit by more than this "
                                                            "(A); bred children always, trial states when reject_trials.")
    reject_trials: bool = Field(True, description="Refuse a trial state with a pair below its limit (EvAX's rule); false: only bred children "
                                                  "are tested and the springs of the gradient steps keep pairs near the limits.")
    spring: float = Field(50.0, gt=0, description="Stiffness of the springs below the closest distances during the gradient steps (eV/A^2).")

    @field_validator("closest_distance")
    @classmethod
    def _pairs(cls, v):
        for key, val in v.items():
            parts = key.split("-")
            if len(parts) != 2 or not all(p.strip() for p in parts):
                raise ValueError(f"closest distance {key!r}: write the pair as 'A-B'")
            if val < 0:
                raise ValueError(f"closest distance {key}: must not be negative")
        return v


class RefineSettings(_Model):
    """[refine]: the gradient refinement (`rmc-exafs refine`) of one sample with one spectrum of one element."""
    copies: int = Field(4, ge=1, description="Copies of the cell refined together; their spectra are averaged.")
    steps: int = Field(..., ge=1, description="Steps of gradient descent.")
    step_size: float = Field(0.003, gt=0, description="Root-mean-square move of the atoms per step (A), decaying on a cosine to "
                                                      "step_size x final_fraction.")
    final_fraction: float = Field(0.1, gt=0, le=1, description="The step at the end, as a fraction of step_size.")
    optimizer: Literal["sgd", "adam"] = Field("sgd", description="'sgd': steepest descent with a normalized step and an accepted-step rule; "
                                                                 "'adam': Adam with the step as its learning rate.")
    ms_absorbers: int = Field(128, ge=1, description="Absorbers per step whose exact multiple-scattering gradient enters the running average.")
    ms_memory: float = Field(1.0, gt=0, description="Memory of the running average of the multiple-scattering gradient, in cycles through all "
                                                    "absorbers.")
    value_every: int = Field(10, ge=1, description="Steps between full passes of every absorber's multiple-scattering spectrum.")
    refresh_every: int = Field(250, ge=1, description="Steps between new searches of the path lists.")
    core_distance: float = Field(..., ge=0, description="Hard core (A): a harmonic penalty acts on every pair closer than this.")
    core_stiffness: float = Field(20.0, gt=0, description="Stiffness of the hard core (eV/A^2).")
    start_kick: float = Field(0.0, ge=0, description="Normal displacement of every atom of every copy at the start (A).")
    log_every: int = Field(50, ge=1, description="Steps between progress lines.")


class RmcConfig(_Model):
    """One RMC-EXAFS run."""
    run: RunSettings = Field(..., description="Outputs, seed, devices.")
    sample: list[SampleSettings] = Field(..., min_length=1, description="The samples, one [[sample]] block each.")
    amplitude: AmplitudeSettings = Field(default_factory=AmplitudeSettings, description="S0^2 and the E0 shift.")
    search: Optional[SearchSettings] = Field(None, description="The population search (`rmc-exafs run`).")
    moves: MoveSettings = Field(default_factory=MoveSettings, description="Exchanges and hops.")
    constraints: ConstraintSettings = Field(default_factory=ConstraintSettings, description="Displacement limit and closest distances.")
    refine: Optional[RefineSettings] = Field(None, description="The gradient refinement (`rmc-exafs refine`).")

    @model_validator(mode="after")
    def _consistent(self):
        labels = [s.label for s in self.sample]
        dup = sorted({v for v in labels if labels.count(v) > 1})
        if dup:
            raise ValueError(f"sample: the labels must differ (repeated: {dup})")
        absorber_of = {}
        for i, s in enumerate(self.sample):
            names = [e.name for e in s.edge]
            dup = sorted({v for v in names if names.count(v) > 1})
            if dup:
                raise ValueError(f"sample.{i}.edge: the spectrum names of one sample must differ (repeated: {dup})")
            for j, e in enumerate(s.edge):
                if absorber_of.setdefault(e.name, e.absorber) != e.absorber:
                    raise ValueError(f"sample.{i}.edge.{j}.absorber: spectra named {e.name!r} share S0^2 and the E0 shift, so they must have the "
                                     f"same absorber ({absorber_of[e.name]}, not {e.absorber})")
        for key in ("s02", "de0"):
            val = getattr(self.amplitude, key)
            if isinstance(val, dict):
                unknown = sorted(set(val) - set(absorber_of))
                if unknown:
                    raise ValueError(f"amplitude.{key}: no spectrum is named {unknown}")
        if self.moves.exchange or self.moves.hops is not None:
            for i, s in enumerate(self.sample):
                for j, e in enumerate(s.edge):
                    if e.criteria_rule == "evax":
                        raise ValueError(f"sample.{i}.edge.{j}.criteria_rule: 'evax' chooses the paths once on the start lattice, where every "
                                         "atom keeps its site; with exchanges or hops ([moves]) use 'feff'")
        for i, s in enumerate(self.sample):
            if s.hop_sites is not None and self.moves.hops is None:
                raise ValueError(f"sample.{i}.hop_sites: hop sites need a [moves.hops] section (the element that hops)")
        if self.search is not None and self.search.gradient_steps > 0 and self.constraints.box_rule == "halve":
            raise ValueError("constraints.box_rule: 'halve' (EvAX's rule) applies to random moves only; with gradient steps use 'project' or 'auto'")
        return self


# ---------------------------------------------------------------------------------------------------------------------
# loading and paths
# ---------------------------------------------------------------------------------------------------------------------
def _abs(value, base):
    return absolute_path(value, base, keep=("fluorite",))


def _resolve(data, base):
    """The raw dict of a configuration with every path made absolute (relative to ``base``)."""
    data = dict(data)
    if isinstance(data.get("run"), dict) and "output" in data["run"]:
        data["run"] = {**data["run"], "output": _abs(data["run"]["output"], base)}
    samples = []
    for s in data.get("sample", []) or []:
        if not isinstance(s, dict):
            samples.append(s)
            continue
        s = dict(s)
        for key in ("structure", "hop_sites"):
            if key in s:
                s[key] = _abs(s[key], base)
        edges = []
        for e in s.get("edge", []) or []:
            if isinstance(e, dict):
                e = dict(e)
                for key in ("chi", "phase"):
                    if key in e:
                        e[key] = _abs(e[key], base)
            edges.append(e)
        if "edge" in s:
            s["edge"] = edges
        samples.append(s)
    if "sample" in data:
        data["sample"] = samples
    if isinstance(data.get("amplitude"), dict) and data["amplitude"].get("from_run"):
        data["amplitude"] = {**data["amplitude"], "from_run": _abs(data["amplitude"]["from_run"], base)}
    if isinstance(data.get("search"), dict) and data["search"].get("continue_from"):
        data["search"] = {**data["search"], "continue_from": _abs(data["search"]["continue_from"], base)}
    mv = data.get("moves")
    if isinstance(mv, dict) and isinstance(mv.get("hops"), dict) and "sites" in mv["hops"]:
        data["moves"] = {**mv, "hops": {**mv["hops"], "sites": _abs(mv["hops"]["sites"], base)}}
    return data


def load_config(path):
    """Read and validate a TOML configuration; relative paths become absolute, relative to the file.

    Returns
    -------
    RmcConfig

    Raises
    ------
    FileNotFoundError, tomllib.TOMLDecodeError, pydantic.ValidationError
    """
    path = Path(path).expanduser().resolve()
    data = tomllib.loads(path.read_text())
    return RmcConfig.model_validate(_resolve(data, path.parent))


def json_schema():
    """The JSON Schema of the run configuration (the pydantic schema of ``RmcConfig``)."""
    return RmcConfig.model_json_schema()


def example_configs():
    """``{name: TOML text}`` of the example configurations that ship with the package (``feffgpu/rmc/templates``)."""
    out = {}
    for item in sorted(resources.files("feffgpu.rmc").joinpath("templates").iterdir(), key=lambda p: p.name):
        if item.name.endswith(".toml"):
            out[item.name[:-5]] = item.read_text()
    return out


def output_directory(path):
    """The absolute output directory that a run of the configuration file ``path`` writes to.

    Raises ``ValueError`` if the file cannot be read or has no ``run.output``."""
    path = Path(path).expanduser().resolve()
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise ValueError(f"{path}: {err}") from err
    out = (data.get("run") or {}).get("output") if isinstance(data.get("run"), dict) else None
    if not out:
        raise ValueError(f"{path}: run.output is not set")
    return Path(_abs(out, path.parent))


def validate_file(path):
    """Validate a TOML configuration without running anything and without a GPU.

    The structure of the keys is validated against the models; then the files the configuration names are checked
    (structure, chi, phase, site files, earlier runs), the structures and spectra are read (CPU only) and the absorbers,
    potentials, windows and cell sizes are checked against them.

    Returns
    -------
    list of str
        Empty if the file is valid; else one message per problem, each starting with the dotted path of the field (for
        example ``sample.0.edge.1.window.kmax: ...``). Never raises for a problem of the file.
    """
    path = Path(path).expanduser()
    if not path.exists():
        return [f"{path}: the file does not exist"]
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as err:
        return [f"{path.name}: not valid TOML: {err}"]
    except (OSError, UnicodeDecodeError) as err:
        return [f"{path.name}: cannot be read: {err}"]
    try:
        cfg = RmcConfig.model_validate(_resolve(data, path.resolve().parent))
    except ValidationError as err:
        return validation_messages(err)
    try:
        return _check_inputs(cfg)
    except Exception as err:                                                     # a problem of the inputs that the checks did not foresee
        return [f"inputs: {type(err).__name__}: {err}"]


def _check_inputs(cfg):
    """Problems with the files a valid configuration names (CPU only; pickled phase files are not loaded)."""
    import numpy as np
    problems = []
    if cfg.search is not None and cfg.search.continue_from and not (Path(cfg.search.continue_from) / "run.json").exists():
        problems.append(f"search.continue_from: {cfg.search.continue_from} is not the output directory of a finished run (no run.json)")
    if cfg.amplitude.from_run and not (Path(cfg.amplitude.from_run) / "run.json").exists():
        problems.append(f"amplitude.from_run: {cfg.amplitude.from_run} is not the output directory of a finished run (no run.json)")
    hops = cfg.moves.hops
    if hops is not None and hops.sites != "fluorite" and not Path(hops.sites).exists():
        problems.append(f"moves.hops.sites: file {hops.sites} does not exist")
    for i, s in enumerate(cfg.sample):
        symbols = cell = None
        if not Path(s.structure).exists():
            problems.append(f"sample.{i}.structure: file {s.structure} does not exist")
        else:
            try:
                from .structure import ModelCell
                model = ModelCell.from_file(s.structure, supercell=s.supercell, format=s.format)
                symbols, cell = model.symbols, model.cell
            except Exception as err:
                problems.append(f"sample.{i}.structure: cannot be read: {err}")
        if s.hop_sites not in (None, "fluorite") and not Path(s.hop_sites).exists():
            problems.append(f"sample.{i}.hop_sites: file {s.hop_sites} does not exist")
        if symbols is not None:
            present = set(symbols.tolist())
            if hops is not None and hops.element not in present and s.hop_sites is not None:
                problems.append(f"sample.{i}.hop_sites: the structure has no {hops.element} atom")
        for j, e in enumerate(s.edge):
            where = f"sample.{i}.edge.{j}"
            if not Path(e.chi).exists():
                problems.append(f"{where}.chi: file {e.chi} does not exist")
            else:
                try:
                    from ..exafs import read_chi, check_window_inside_data
                    kd, _cd = read_chi(e.chi, *e.chi_columns)
                    check_window_inside_data(kd, e.window.kmin, e.window.kmax, e.window.dk, name="the window")
                except Exception as err:
                    problems.append(f"{where}.chi: {err}")
            nph = None
            if not Path(e.phase).exists():
                problems.append(f"{where}.phase: file {e.phase} does not exist")
            elif not e.phase.lower().endswith((".pkl", ".pickle")):
                try:
                    from ..pad import read_phase_pad
                    nph = int(read_phase_pad(e.phase).nph)
                except Exception as err:
                    problems.append(f"{where}.phase: cannot be read as a phase.pad: {err}")
            if symbols is not None:
                seen = sorted(set(symbols.tolist()) - set(e.ignore))
                if e.absorber not in set(symbols.tolist()):
                    problems.append(f"{where}.absorber: the structure has no {e.absorber} atom")
                missing = [el for el in seen if el not in e.potentials]
                if missing and (nph is None or nph > 1) and (e.potentials or nph is not None):
                    problems.append(f"{where}.potentials: no scattering potential for {missing}")
                if nph is not None:
                    bad = {el: ip for el, ip in e.potentials.items() if not 1 <= ip <= nph}
                    if bad:
                        problems.append(f"{where}.potentials: {bad} are not scattering potentials of the phase file (1 to {nph})")
                if cell is not None:
                    c = np.asarray(cell, float)
                    vol = abs(np.linalg.det(c))
                    width = min(vol / np.linalg.norm(np.cross(c[(k + 1) % 3], c[(k + 2) % 3])) for k in range(3))
                    if width < 2 * e.rpath - 1e-9:
                        problems.append(f"{where}.rpath: the cell's smallest width ({width:.2f} A) must be at least twice rpath ({e.rpath} A); "
                                        "use a larger supercell or a smaller rpath")
    return problems
