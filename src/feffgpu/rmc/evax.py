"""Taking over an EvAX calculation: its parameter file, its structure and its output files, and the inputs of
``rmc-exafs`` for the same problem.

EvAX (Timoshenko, Kuzmin, Purans) is run from one parameter file (usually ``start.dat``) that names a structure file
(the unit cell, in the layout of a VASP POSCAR) and one measured spectrum per absorption edge. Its output directory
holds the fitted spectra and the final atoms. ``EvaxCalculation`` reads all of it; ``write_inputs`` writes the
structure, the spectra, the FEFF inputs of the potentials and a TOML configuration for ``rmc-exafs``, with EvAX's
settings translated (``rmc-exafs from-evax`` on the command line).

What the translation does, in EvAX's keywords:

- ``Edge``, ``Edge_type``, ``k_min``, ``k_max``, ``k_power``, ``R_min``, ``R_max``, ``R0_for_WT``, ``R_max_for_FEFF``,
  ``FEFF_criteria``, ``Space``: one ``[[sample.edge]]`` block per spectrum, with ``criteria_rule = "evax"`` (EvAX's way
  of choosing the paths) and, for ``Space w``, ``objective = "wavelet"`` and a window without sills.
- ``N_legs``: EvAX's default is 8; this package computes paths of up to four legs (``nleg = 4``).
- ``S02``, ``dE0``: EvAX divides the data by S02 and shifts the data; here both act on the model, so ``de0 = -dE0``.
  With ``target="evax"`` the spectrum written is EvAX's own target (``ipolEXAFS.dat`` of its output, already divided
  and shifted) and the configuration holds ``s02 = 1``, ``de0 = 0``.
- ``r_scf``, ``n_scf``, ``ca``, ``n_mix``: the FEFF input of the potentials, written as EvAX writes it (a negative
  ``r_scf`` makes EvAX choose the radius of the self-consistency loop itself; ``scf_radius`` is its rule).
- ``Supercell``, ``pbc``: the start structure. Without periodic boundaries the atoms are put in a box so wide that no
  atom has a periodic image within the cutoff.
- ``Number_of_states``, ``Froze_in``, ``Stop_after``, ``Maximal_step_length``, ``Acceptance_rate``,
  ``Initial_displacement``, ``Maximal_displacement``: the ``[search]`` and ``[constraints]`` sections.

EvAX runs the classic FEFF 8.5L; this package reproduces the open-source edition. Their Fermi levels can differ by
about 0.1 eV for the same input, which is a shift of the energy origin of the model (``energy_shift``).
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
import numpy as np
from ..exafs.transform import ETOK, KSTEP
from ..io.evax import read_parameters

__all__ = ["read_parameters", "scf_radius", "EvaxSpectrum", "EvaxCalculation"]

def scf_radius(distances, r_scf=-1.0):
    """EvAX's radius of the self-consistency loop of FEFF (A).

    A positive ``r_scf`` is used as given. Otherwise EvAX takes, from the sorted distances of the atoms around the
    absorber: the nearest distance d1; the first distance at least 1 A longer; then the first one at least 0.1 d1
    longer than that. If the result is not between 1 and 10 A it uses 5 A."""
    if r_scf > 0:
        return float(r_scf)
    d = np.unique(np.round(np.asarray(distances, float), 6))
    d = d[d > 1e-6]
    first = d[0]
    second = d[d - first >= 1.0]
    if not len(second):
        return 5.0
    third = d[d - second[0] >= 0.1 * first]
    if not len(third):
        return 5.0
    return float(third[0]) if 1.0 < third[0] < 10.0 else 5.0


@dataclass
class EvaxSpectrum:
    """The settings of one spectrum of an EvAX calculation."""
    element: str
    edge: str
    data_file: str
    k_column: int
    chi_column: int
    k_power_io: int
    s02: float
    de0: float
    kmin: float
    kmax: float
    kweight: int
    rmin: float
    rmax: float
    wavelet_r0: float | None
    rpath: float
    criteria: float
    nlegs: int
    space: str
    r_scf: float

    @property
    def name(self):
        return f"{self.element}_{self.edge}"


class EvaxCalculation:
    """An EvAX calculation: the parameter file, the files it names and, if given, its output files.

    Parameters
    ----------
    parameter_file : str or Path
        EvAX's parameter file (``start.dat``).
    inputs : str or Path, optional
        The directory of the structure file and the spectra. Default: the directory of the parameter file. The files
        are looked for by their names (the directory part that the parameter file gives is not used).
    outputs : str, Path, dict or list, optional
        EvAX's output directory. Projects with several spectra may have one output directory per spectrum:
        ``{element: directory}`` or a list in the order of the spectra.

    Attributes
    ----------
    parameters : dict
    spectra : list of EvaxSpectrum
    periodic : bool
    supercell : list of int
    lattice : numpy.ndarray, shape (3, 3)
        The unit cell (rows).
    """

    def __init__(self, parameter_file, inputs=None, outputs=None):
        self.parameter_file = Path(parameter_file)
        self.inputs = Path(inputs) if inputs is not None else self.parameter_file.parent
        p = self.parameters = read_parameters(self.parameter_file)
        if "Edge" not in p or "File_with_structure" not in p:
            raise ValueError(f"{self.parameter_file}: not an EvAX parameter file (no Edge or File_with_structure keyword)")
        n = len(p["Edge"])

        def pick(key, i, default=None):
            v = p.get(key)
            if not v:
                return default
            return v[i] if len(v) > i else v[0]

        files = p.get("File_with_experimental_EXAFS_signal", [])
        if len(files) < n:
            raise ValueError(f"{self.parameter_file}: {n} edges but {len(files)} spectra (File_with_experimental_EXAFS_signal)")
        self.spectra = []
        for i in range(n):
            r0 = pick("R0_for_WT", i)
            self.spectra.append(EvaxSpectrum(
                element=p["Edge"][i], edge=pick("Edge_type", i, "K"), data_file=Path(files[i].replace("\\", "/")).name,
                k_column=int(pick("Column_for_k", i, 1)) - 1, chi_column=int(pick("Column_for_chi", i, 2)) - 1,
                k_power_io=int(float(pick("k_power_in_IO", i, 2))), s02=float(pick("S02", i, 1.0)), de0=float(pick("dE0", i, 0.0)),
                kmin=float(pick("k_min", i, 0.05)), kmax=float(pick("k_max", i, 19.0)), kweight=int(float(pick("k_power", i, 2))),
                rmin=float(pick("R_min", i, 0.0)), rmax=float(pick("R_max", i, 5.0)), wavelet_r0=float(r0) if r0 is not None and float(r0) > 0 else None,
                rpath=float(pick("R_max_for_FEFF", i, 6.0)), criteria=float(pick("FEFF_criteria", i, 0.0)), nlegs=int(pick("N_legs", i, 8)),
                space=str(pick("Space", i, "w")), r_scf=float(pick("r_scf", i, -1.0))))
        self.periodic = int(float(pick("pbc", 0, 1))) == 1
        self.supercell = [int(v) for v in p.get("Supercell", ["1", "1", "1"])][:3]
        self.structure_file = self.inputs / Path(p["File_with_structure"][0].replace("\\", "/")).name
        self._unit = None
        if outputs is None:
            self.outputs = [None] * n
        elif isinstance(outputs, dict):
            self.outputs = [Path(outputs[s.element]) if s.element in outputs else None for s in self.spectra]
        elif isinstance(outputs, (list, tuple)):
            self.outputs = [Path(o) if o is not None else None for o in outputs] + [None] * (n - len(outputs))
        else:
            self.outputs = [Path(outputs)] * n
        self._shift = None

    # -- the structure -------------------------------------------------------------------------------------------
    def unit_cell(self):
        """The unit cell of the structure file: symbols (m,), positions (m, 3) in A, lattice vectors (3, 3) as rows."""
        if self._unit is None:
            from ase.io import read
            atoms = read(self.structure_file, format="vasp")
            self._unit = (np.array(atoms.get_chemical_symbols()), atoms.get_positions(), np.array(atoms.get_cell()))
        return self._unit

    @property
    def lattice(self):
        return self.unit_cell()[2]

    def _xyz(self, names):
        for out in self.outputs:
            if out is None:
                continue
            for name in names:
                f = out / name
                if f.exists():
                    lines = f.read_text().splitlines()
                    m = re.search(r"First\s+(\d+)\s+lines", lines[1])
                    count = int(m.group(1)) if m else int(lines[0].split()[0])
                    rows = [line.split() for line in lines[2:2 + count]]
                    return np.array([r[0] for r in rows]), np.array([[float(v) for v in r[1:4]] for r in rows])
        return None

    def _box(self, positions):
        """The cell used here and the shift of the positions: EvAX's supercell with periodic boundaries; without them a
        box so wide that no atom has a periodic image within the largest cutoff."""
        if self.periodic:
            return np.array(self.supercell, float)[:, None] * self.lattice, np.zeros(3)
        pad = max(s.rpath for s in self.spectra) + 0.5
        lo, hi = positions.min(0), positions.max(0)
        return np.diag(hi - lo + 2 * pad + 1.0), pad + 0.5 - lo

    def sites(self, supercell=None):
        """The start structure: symbols (n,), positions (n, 3), cell (3, 3).

        With EvAX's output files at hand and no other ``supercell`` asked for, the atoms come from them, in EvAX's own
        order (``phantoms_eq.xyz`` or ``initial.xyz``), so that its final atoms can be compared atom by atom.
        Otherwise the unit cell is repeated ``supercell`` times (default EvAX's ``Supercell``)."""
        own = supercell is None or [int(v) for v in supercell] == self.supercell
        found = self._xyz(("phantoms_eq.xyz", "initial.xyz")) if own else None
        if found is not None:
            sym, pos = found
        else:
            n = self.supercell if supercell is None else [int(v) for v in supercell]
            usym, upos, lat = self.unit_cell()
            cells = np.array([[i, j, k] for i in range(n[0]) for j in range(n[1]) for k in range(n[2])], float)
            pos = (upos[None, :, :] + (cells @ lat)[:, None, :]).reshape(-1, 3)
            sym = np.tile(usym, len(cells))
            if not own:
                if not self.periodic:
                    raise ValueError("another supercell needs periodic boundaries")
                return sym, pos, np.array(n, float)[:, None] * lat
        cell, shift = self._box(pos)
        self._shift = shift
        return sym, pos + shift, cell

    def final(self):
        """EvAX's final atoms (n, 3), in the order and the frame of ``sites()``; None without output files."""
        found = self._xyz(("phantoms.xyz", "final.xyz"))
        if found is None:
            return None
        if self._shift is None:
            self.sites()
        return found[1] + self._shift

    def minimum_supercell(self):
        """The smallest repetition of the unit cell that is at least twice as wide as the largest cutoff along every cell
        vector (what the forward model of this package needs)."""
        lat = self.lattice
        widths = abs(np.linalg.det(lat)) / np.array([np.linalg.norm(np.cross(lat[(i + 1) % 3], lat[(i + 2) % 3])) for i in range(3)])
        need = 2 * max(s.rpath for s in self.spectra)
        return [max(int(math.ceil(need / w - 1e-9)), 1) for w in widths]

    # -- the spectra ---------------------------------------------------------------------------------------------
    def data(self, i):
        """The measured spectrum of spectrum ``i`` as it is in the data file: k (1/A) and chi(k) without k weight."""
        s = self.spectra[i]
        rows = []
        for line in (self.inputs / s.data_file).read_text(errors="replace").splitlines():
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(parts[s.k_column]), float(parts[s.chi_column])])
            except (ValueError, IndexError):
                continue
        t = np.array(rows)
        t = t[np.argsort(t[:, 0])]
        k = t[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = np.where(k > 0, t[:, 1] / np.maximum(k, 1e-12) ** s.k_power_io, 0.0)
        return k, chi

    def _two_columns(self, i, name):
        out = self.outputs[i]
        if out is None or not (out / name).exists():
            return None
        t = np.loadtxt(out / name)
        a, b = t[:, 0], t[:, 1]
        k, v = (a, b) if np.allclose(np.diff(a), KSTEP, atol=1e-4) else (b, a)
        s = self.spectra[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            return k, np.where(k > 0, v / np.maximum(k, 1e-12) ** s.k_power_io, 0.0)

    def target(self, i):
        """EvAX's target of spectrum ``i``: k and chi(k), the data divided by S02 and shifted by dE0 the way EvAX does it.

        From EvAX's own file (``ipolEXAFS.dat``) when the output files are at hand; otherwise computed by EvAX's rule:
        the data are multiplied by ``k^k_power`` on their own k, read by linear interpolation at
        ``sqrt(k^2 - 0.2625 dE0)``, and divided by ``k^k_power`` and by S02."""
        own = self._two_columns(i, "ipolEXAFS.dat")
        if own is not None:
            return own
        s = self.spectra[i]
        kd, cd = self.data(i)
        k = KSTEP * np.arange(int(round(s.kmin / KSTEP)), int(round(s.kmax / KSTEP)) + 1)
        q = k ** 2 - ETOK * s.de0
        kq = np.sqrt(np.clip(q, 0.0, None))
        return k, np.interp(kq, kd, cd * kd ** s.kweight) / np.maximum(k, 1e-12) ** s.kweight / s.s02

    def fitted(self, i):
        """EvAX's final calculated spectrum of spectrum ``i`` (``EXAFS.dat``): k and chi(k); None without output files."""
        return self._two_columns(i, "EXAFS.dat")

    def path_groups(self, i):
        """EvAX's list of the groups of paths it kept for spectrum ``i`` (the head of ``partialEXAFS.dat``): a list of
        ``{legs, degeneracy, half_length}``, in the form of ``feffgpu.exafs.MultipleScattering.types``; None without
        that file."""
        out = self.outputs[i]
        if out is None or not (out / "partialEXAFS.dat").exists():
            return None
        head = (out / "partialEXAFS.dat").read_text().splitlines()[:3]
        legs, deg, half = ([float(v) for v in line.split()[1:]] for line in head)
        return [dict(legs=int(a), degeneracy=b, half_length=c) for a, b, c in zip(legs, deg, half)]

    # -- the potentials ------------------------------------------------------------------------------------------
    def feff_input(self, i):
        """The FEFF input of the potentials of spectrum ``i`` as EvAX writes it, and the scattering potential of every
        element: ``(text, {element: index})``.

        The absorber is the first atom of the edge's element; the cluster reaches to the larger of the path cutoff and
        the radius of the self-consistency loop; the potentials are numbered in the order in which the elements first
        appear in the structure."""
        from ..md.potentials import cut_cluster
        from ..io.structure import Z_OF
        s = self.spectra[i]
        sym, pos, cell = self.sites()
        absorber = int(np.nonzero(sym == s.element)[0][0])
        rel, csym, dist = cut_cluster(pos, sym, cell, absorber, 12.0)
        elements = list(dict.fromkeys(sym.tolist()))
        radius = scf_radius(dist[1:], s.r_scf)
        if s.r_scf <= 0:
            for el in elements:                                           # every element needs an atom inside
                nearest = dist[1:][csym[1:] == el]
                if len(nearest) and radius < nearest[0]:
                    radius = float(nearest[0]) + 0.1
        p = self.parameters
        n_scf, ca, n_mix = int(float(p.get("n_scf", ["30"])[0])), float(p.get("ca", ["0.1"])[0]), int(float(p.get("n_mix", ["1"])[0]))
        counts = [int((sym == el).sum()) for el in elements]
        g = reduce(math.gcd, counts)
        lines = [" TITLE   potentials as EvAX sets them up", f" EDGE      {s.edge}", " S02       1.0", " CONTROL   1 1 1 1 1 1", " PRINT     1 0 0 0 0 0",
                 f" SCF       {radius:.6f}   0      {n_scf}     {ca}     {n_mix}", " EXCHANGE  0      0   0", " EXAFS 20.0", f" RPATH    {s.rpath}", "",
                 " POTENTIALS", f"       0   {Z_OF[s.element]}    {s.element}     -1      -1       0"]
        lines += [f"       {j + 1}   {Z_OF[el]}    {el}     -1      -1       {c // g}" for j, (el, c) in enumerate(zip(elements, counts))]
        lines += ["", " ATOMS", f" 0.000000 0.000000 0.000000 0 {s.element}"]
        inside = dist <= max(s.rpath, radius) + 1e-6
        inside[0] = False
        lines += [f" {r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {elements.index(el) + 1} {el}" for r, el in list(zip(rel[inside], csym[inside]))[:998]]
        return "\n".join(lines + ["  END", ""]), {el: j + 1 for j, el in enumerate(elements)}

    # -- the inputs of rmc-exafs ---------------------------------------------------------------------------------
    def write_inputs(self, folder, target="data", supercell=None, gradient_steps=0, states=None, generations=None, energy_shift=None, label="sample"):
        """Write the inputs of ``rmc-exafs`` for this calculation into ``folder``.

        Files: ``start.extxyz`` (the start structure), one ``<element>_<edge>.chi`` per spectrum (k and chi(k)), one
        ``feff_<element>_<edge>.inp`` per spectrum (for ``feffgpu phase``, which writes the phase file the configuration
        names) and ``rmc.toml``.

        Parameters
        ----------
        folder : str or Path
        target : str
            ``"data"``: the measured spectra as they are, with S0^2 and the energy shift on the model
            (``s02 = S02``, ``de0 = -dE0``). ``"evax"``: EvAX's own targets (the data divided by S02 and shifted by
            EvAX), with ``s02 = 1``, ``de0 = 0``; for comparisons with EvAX's output.
        supercell : sequence of int, optional
            Another repetition of the unit cell than EvAX's (see ``minimum_supercell``).
        gradient_steps : int
            Gradient steps after every random move of the search (0: EvAX's rule).
        states, generations : int, optional
            Default EvAX's ``Number_of_states`` and ``Stop_after`` (or ``Froze_in`` when EvAX has no stop).
        energy_shift : dict or float, optional
            Added to ``de0``: the difference of the energy origin between the FEFF build EvAX ran and this package
            (eV), per spectrum name or one number.
        label : str
            Name of the sample (its output subdirectory).

        Returns
        -------
        dict
            ``config`` (the path of ``rmc.toml``), ``phase_commands`` (the ``feffgpu phase`` commands to run in the
            folder) and ``notes`` (what could not be taken over as it is).
        """
        from ase import Atoms
        from ase.io import write
        if target not in ("data", "evax"):
            raise ValueError("target must be 'data' or 'evax'")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        notes = []
        sym, pos, cell = self.sites(supercell)
        widths = abs(np.linalg.det(cell)) / np.array([np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])
        need = 2 * max(s.rpath for s in self.spectra)
        if widths.min() < need - 1e-6:
            raise ValueError(f"EvAX's supercell {self.supercell} is {widths.min():.2f} A wide and the forward model needs twice the cutoff ({need:.1f} A): "
                             f"give supercell={self.minimum_supercell()} or more")
        write(folder / "start.extxyz", Atoms(symbols=sym.tolist(), positions=pos, cell=cell, pbc=True), format="extxyz")
        p = self.parameters
        edges, commands, s02s, de0s = [], [], {}, {}
        for i, s in enumerate(self.spectra):
            k, chi = self.target(i) if target == "evax" else self.data(i)
            np.savetxt(folder / f"{s.name}.chi", np.column_stack([k, chi]), fmt="%.6f %.9e", header="k (1/A)  chi(k)" + ("  [EvAX's target]" if target == "evax" else ""))
            text, ipot = self.feff_input(i)
            (folder / f"feff_{s.name}.inp").write_text(text)
            commands.append(f"feffgpu phase feff_{s.name}.inp phase_{s.name}.pkl")
            shift = energy_shift.get(s.name, 0.0) if isinstance(energy_shift, dict) else float(energy_shift or 0.0)
            s02s[s.name] = 1.0 if target == "evax" else s.s02
            de0s[s.name] = (0.0 if target == "evax" else -s.de0) + shift
            wavelet = s.space.lower().startswith("w")
            if not wavelet:
                notes.append(f"{s.name}: EvAX's Space {s.space!r} is fitted here as the R-factor of the Fourier transform with a Hanning window (sills 1 1/A)")
            if s.nlegs > 4:
                notes.append(f"{s.name}: EvAX's N_legs is {s.nlegs}; paths of up to four legs are computed here")
            lines = ["  [[sample.edge]]", f'  name = "{s.name}"', f'  absorber = "{s.element}"', f'  chi = "{s.name}.chi"', f'  phase = "phase_{s.name}.pkl"',
                     "  potentials = { " + ", ".join(f"{el} = {j}" for el, j in ipot.items()) + " }",
                     f"  window = {{ kmin = {s.kmin}, kmax = {s.kmax}, dk = {0.0 if wavelet else 1.0}, kweight = {s.kweight}, rmin = {s.rmin}, rmax = {s.rmax} }}",
                     f"  rpath = {s.rpath}", f"  nleg = {min(s.nlegs, 4)}", f"  criteria = {s.criteria}", '  criteria_rule = "evax"',
                     f"  group_tolerance = {float(p.get('Pathfinder_robustness', ['0.5'])[0])}", "  pair_min = 1.0"]
            if wavelet:
                lines.append('  objective = "wavelet"')
                if s.wavelet_r0 is not None:
                    lines.append(f"  wavelet_r0 = {s.wavelet_r0}")
            edges.append("\n".join(lines))
        froze = int(float(p.get("Froze_in", ["1500"])[0]))
        stop = int(float(p.get("Stop_after", ["-1"])[0]))
        n_states = int(states or int(float(p.get("Number_of_states", ["32"])[0])))
        if n_states % 2:
            n_states += 1
            notes.append(f"the number of states is made even ({n_states}): the search breeds in pairs")
        if n_states < 2:
            n_states = 2
        acceptance = p.get("Acceptance_rate", p.get("Target_dicarding_rate", ["0.8"]))[0]
        text = ["# written by feffgpu.rmc.evax from " + self.parameter_file.name, "[run]", 'output = "out"', "", "[[sample]]", f'label = "{label}"',
                'structure = "start.extxyz"', *edges, "", "[amplitude]", 'mode = "fixed"',
                "s02 = { " + ", ".join(f"{n} = {v}" for n, v in s02s.items()) + " }", "de0 = { " + ", ".join(f"{n} = {v:.4f}" for n, v in de0s.items()) + " }",
                "", "[search]", f"states = {n_states}", f"generations = {int(generations or (stop if stop > 0 else froze))}", f"froze_in = {froze}",
                f"step = {float(p.get('Maximal_step_length', ['0.005'])[0])}", f"acceptance = {abs(float(acceptance))}",
                f"start_kick = {float(p.get('Initial_displacement', ['0.1'])[0])}", f"gradient_steps = {int(gradient_steps)}", "",
                "[constraints]", f"max_displacement = {float(p.get('Maximal_displacement', ['0.4'])[0])}", ""]
        frac = pos[:min(len(pos), 64)] @ np.linalg.inv(cell)                  # the shortest distance in the start structure (minimum image)
        d = frac[:, None, :] - (pos @ np.linalg.inv(cell))[None, :, :]
        d = np.linalg.norm((d - np.round(d)) @ cell, axis=2)
        nearest = float(d[d > 1e-6].min())
        text += ["# the gradient refinement (`rmc-exafs refine rmc.toml`): far fewer evaluations than the search (`rmc-exafs run rmc.toml`)",
                 "[refine]", "copies = 1", "steps = 300", "ms_absorbers = 64", f"core_distance = {0.7 * nearest:.2f}",
                 f"start_kick = {float(p.get('Initial_displacement', ['0.1'])[0]):g}", "log_every = 10", ""]
        (folder / "rmc.toml").write_text("\n".join(text))
        if not self.periodic:
            notes.append("no periodic boundaries in EvAX: the atoms are in a box so wide that no atom has an image within the cutoff")
        return dict(config=str(folder / "rmc.toml"), phase_commands=commands, notes=notes)
