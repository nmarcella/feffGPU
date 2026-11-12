"""Reader for FEFF's phase.pad (packed-ASCII data, feff85exafs COMMON/padlib.f and COMMON/rdxsph.f).

The file holds everything the path stage needs from the potentials and phase-shift stages: the complex energy
grid, the energy reference (interstitial potential, core-hole width and self-energy), the complex partial-wave
phase shifts of every unique potential, and the absorber's reduced dipole/quadrupole matrix elements.
Units are FEFF's code units: energies in Hartree, lengths in bohr.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

IBASE, IOFF, IBAS2 = 90, 37, 45
BOHR = 0.52917721067
HART = 2.0 * 13.60569301
PHMIN = 1.0e-7


def unpad(s, npack):
    """One packed number (padlib.f: unpad)."""
    iexp = (ord(s[0]) - IOFF) - IBAS2
    c1 = ord(s[1]) - IOFF
    isgn = (c1 % 2) * 2 - 1
    itmp = c1 // 2
    total = itmp / (IBASE * IBASE)
    for i in range(npack, 2, -1):                                   # i = npack .. 3, as in the Fortran loop
        total += (ord(s[i - 1]) - IOFF) / IBASE ** i
    return 2 * isgn * IBASE * total * 10.0 ** iexp


class _Lines:
    def __init__(self, text):
        self.lines = text.splitlines()
        self.i = 0

    def next(self):
        line = self.lines[self.i]
        self.i += 1
        return line

    def read_real(self, npack, npts):
        out = []
        while len(out) < npts:
            line = self.next().rstrip()
            if line[0] != "!":
                raise ValueError(f"expected a real packed line, got {line[:20]!r}")
            body, n = line[1:], len(line) // npack
            for j in range(n):
                out.append(unpad(body[j * npack:(j + 1) * npack], npack))
                if len(out) == npts:
                    break
        return np.array(out)

    def read_complex(self, npack, npts):
        out, np2 = [], 2 * npack
        while len(out) < npts:
            line = self.next().rstrip()
            if line[0] != "$":
                raise ValueError(f"expected a complex packed line, got {line[:20]!r}")
            body, n = line[1:], len(line) // np2
            for j in range(n):
                re = unpad(body[j * np2:j * np2 + npack], npack)
                im = unpad(body[j * np2 + npack:(j + 1) * np2], npack)
                out.append(complex(re, im))
                if len(out) == npts:
                    break
        return np.array(out)


@dataclass
class PhaseData:
    nsp: int
    ne: int                 # energy points in all
    ne1: int                # points on the real axis (the EXAFS grid)
    ne3: int
    nph: int                # unique potentials are 0..nph
    ihole: int
    ik0: int
    ixc: int
    rs: float
    vint: float
    rnrmav: float
    xmu: float
    edge: float
    em: np.ndarray          # (ne,) complex energy, Hartree
    eref: np.ndarray        # (ne, nsp) complex
    lmax0: np.ndarray       # (nph+1,) partial waves stored per potential
    iz: np.ndarray
    potlbl: list
    ph: np.ndarray          # (ne, 2*ltot+1, nsp, nph+1) complex phase shifts, index l + ltot for l in -lmax0..lmax0
    rkk: np.ndarray         # (ne, 8, nsp) complex reduced matrix elements
    ltot: int

    @property
    def lmax(self):
        """(ne, nph+1): largest l with a non-negligible phase shift, as rdxsph defines it."""
        out = np.zeros((self.ne, self.nph + 1), int)
        for iph in range(self.nph + 1):
            for ie in range(self.ne):
                lm = 0
                for l in range(self.lmax0[iph], -1, -1):
                    lm = l
                    if abs(np.sin(self.ph[ie, l + self.ltot, 0, iph])) > PHMIN or \
                       abs(np.sin(self.ph[ie, l + self.ltot, self.nsp - 1, iph])) > PHMIN:
                        break
                out[ie, iph] = lm
        return out

    def phase(self, ie, l, iph, isp=0):
        return self.ph[ie, l + self.ltot, isp, iph]

    @property
    def xk(self):
        """Real momentum relative to the edge (bohr^-1), FEFF's getxk."""
        e = self.em.real - self.edge
        return np.sign(e) * np.sqrt(np.abs(2 * e))

    @property
    def ck(self):
        """Complex momentum p = sqrt(2 (E - E_ref)), spin-averaged like genfmt_prep."""
        eref = self.eref[:, 0] if self.nsp == 1 else 0.5 * (self.eref[:, 0] + self.eref[:, -1])
        return np.sqrt(2 * (self.em - eref))


def read_phase_pad(filename):
    text = open(filename, encoding="latin-1").read()
    L = _Lines(text)
    head = L.next()
    ints = [int(head[1 + 5 * j:5 + 5 * j]) for j in range(9)]
    nsp, ne, ne1, ne3, nph, ihole, ik0, npadx, ixc = ints
    rs, vint = float(head[45:56]), float(head[56:67])
    rnrmav, xmu, edge = L.read_real(npadx, 3)
    em = L.read_complex(npadx, ne)
    eref = L.read_complex(npadx, ne * nsp).reshape(nsp, ne).T
    lmax0, iz, potlbl = [], [], []
    ph_list = []
    for iph in range(nph + 1):
        line = L.next()
        lmax0.append(int(line[1:4])); iz.append(int(line[5:8])); potlbl.append(line[9:15].strip())
        per_spin = []
        for isp in range(nsp):
            temp = L.read_complex(npadx, ne * (2 * lmax0[iph] + 1)).reshape(ne, 2 * lmax0[iph] + 1)   # ll fastest
            per_spin.append(temp)
        ph_list.append(np.stack(per_spin, -1))                       # (ne, 2 lmax0+1, nsp)
    ltot = max(lmax0)
    ph = np.zeros((ne, 2 * ltot + 1, nsp, nph + 1), complex)
    for iph in range(nph + 1):
        lo = ltot - lmax0[iph]
        ph[:, lo:lo + 2 * lmax0[iph] + 1, :, iph] = ph_list[iph]
    rkk = L.read_complex(npadx, ne * 8 * nsp).reshape(nsp, 8, ne).transpose(2, 1, 0)   # ie fastest, then kdif, isp
    return PhaseData(nsp, ne, ne1, ne3, nph, ihole, ik0, ixc, rs, vint, float(rnrmav), float(xmu), float(edge), em, eref,
                     np.array(lmax0), np.array(iz), potlbl, ph, rkk, ltot)


@dataclass
class PotData:
    """Everything the potentials stage hands to the phase-shift stage (COMMON/rdpot.f), FEFF's code units.
    Radial arrays live on the 251-point Loucks grid r_i = exp(-8.8 + 0.05 (i-1)) bohr."""
    ntitle: int
    title: list
    nph: int
    nohole: int
    ihole: int
    inters: int
    iafolp: int
    jumprm: int
    iunf: int
    rnrmav: float
    xmu: float
    vint: float
    rhoint: float
    emu: float
    s02: float
    erelax: float
    wp: float
    ecv: float
    rs: float
    xf: float
    qtotel: float
    totvol: float
    imt: np.ndarray
    rmt: np.ndarray
    inrm: np.ndarray
    iz: np.ndarray
    kappa: np.ndarray       # (30,) kappa of the atomic orbitals, as the atomic code ordered them
    rnrm: np.ndarray
    folp: np.ndarray
    folpx: np.ndarray
    xnatph: np.ndarray
    xion: np.ndarray
    dgc0: np.ndarray        # (251,) core-hole orbital, large component (times r)
    dpc0: np.ndarray        # (251,) small component
    dgc: np.ndarray         # (251, 30, nph+1) all orbitals of every potential
    dpc: np.ndarray
    adgc: np.ndarray        # (10, 30, nph+1) their development coefficients at the origin
    adpc: np.ndarray
    edens: np.ndarray       # (251, nph+1) overlapped density times 4 pi
    vclap: np.ndarray       # (251, nph+1) overlapped Coulomb potential
    vtot: np.ndarray        # (251, nph+1) total potential with ground-state exchange
    edenvl: np.ndarray
    vvalgs: np.ndarray
    dmag: np.ndarray
    xnval: np.ndarray       # (30, nph+1)
    eorb: np.ndarray
    iorb: np.ndarray        # (8, nph+1) index -4..3
    qnrm: np.ndarray
    xnmues: np.ndarray      # (5, nph+1)


def _ints(line, width=5):
    return [int(line[j * width:(j + 1) * width]) for j in range(len(line) // width) if line[j * width:(j + 1) * width].strip()]


def read_pot_pad(filename):
    text = open(filename, encoding="latin-1").read()
    L = _Lines(text)
    ntitle, nph, npadx, nohole, ihole, inters, iafolp, jumprm, iunf = _ints(L.next())
    title = [L.next().strip() for _ in range(ntitle)]
    dum = L.read_real(npadx, 13)
    n1 = nph + 1

    def int_list(n):
        out = []
        while len(out) < n:
            out += _ints(L.next())
        return np.array(out[:n])

    imt = int_list(n1)
    rmt = L.read_real(npadx, n1)
    inrm = int_list(n1)
    iz = int_list(n1)
    kappa = int_list(30)
    rnrm, folp, folpx, xnatph, xion = [L.read_real(npadx, n1) for _ in range(5)]
    dgc0, dpc0 = L.read_real(npadx, 251), L.read_real(npadx, 251)
    dgc = L.read_real(npadx, 251 * 30 * n1).reshape(n1, 30, 251).transpose(2, 1, 0)
    dpc = L.read_real(npadx, 251 * 30 * n1).reshape(n1, 30, 251).transpose(2, 1, 0)
    adgc = L.read_real(npadx, 10 * 30 * n1).reshape(n1, 30, 10).transpose(2, 1, 0)
    adpc = L.read_real(npadx, 10 * 30 * n1).reshape(n1, 30, 10).transpose(2, 1, 0)
    edens, vclap, vtot, edenvl, vvalgs, dmag = [L.read_real(npadx, 251 * n1).reshape(n1, 251).T for _ in range(6)]
    xnval = L.read_real(npadx, 30 * n1).reshape(n1, 30).T
    eorb = L.read_real(npadx, 30)
    iorb = np.array([[int(v) for v in L.next().split()] for _ in range(n1)]).T
    qnrm = L.read_real(npadx, n1)
    xnmues = L.read_real(npadx, 5 * n1).reshape(n1, 5).T
    return PotData(ntitle, title, nph, nohole, ihole, inters, iafolp, jumprm, iunf, *[float(v) for v in dum],
                   imt, rmt, inrm, iz, kappa, rnrm, folp, folpx, xnatph, xion, dgc0, dpc0, dgc, dpc, adgc, adpc,
                   edens, vclap, vtot, edenvl, vvalgs, dmag, xnval, eorb, iorb, qnrm, xnmues)


def load_phase(phase):
    """A phase file as ``PhaseData``.

    Parameters
    ----------
    phase : PhaseData, str or Path
        The object itself, a ``phase.pad`` written by FEFF 8.5L, or a pickle written by ``feffgpu phase``. A pickle can
        run code when it is loaded: load only phase files that you made yourself or that come from someone you trust.

    Returns
    -------
    PhaseData
    """
    def is_phase(obj):
        return isinstance(obj, PhaseData) or (hasattr(obj, "iz") and hasattr(obj, "ph"))

    if is_phase(phase):
        return phase
    from pathlib import Path
    p = Path(phase)
    if not p.is_file():
        raise FileNotFoundError(f"no such phase file: {p}")
    with open(p, "rb") as f:
        head = f.read(1)
    if head == b"\x80" or p.suffix.lower() in (".pkl", ".pickle"):          # a pickle (protocol 2 or later)
        import pickle
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not is_phase(obj):
            raise ValueError(f"{p}: the pickle does not hold a phase file (feffgpu.PhaseData, got {type(obj).__name__})")
        return obj
    return read_phase_pad(str(p))
