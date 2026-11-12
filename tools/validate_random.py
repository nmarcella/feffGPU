#!/usr/bin/env python3
"""Validation of the GPU FEFF on random environments, against FEFF itself.

Two families of random cases, one seed:
  A. synthetic: random elements (one to three species), a random structure (simple cubic, bcc / CsCl, fcc,
     diamond / zincblende, rock salt, fluorite, perovskite, or a random dense packing) at a lattice constant set by
     the covalent radii times a random factor, with random thermal disorder, vacancies, anisotropic strain and,
     sometimes, a surface cut; a random edge among those the absorber has; random cards (self-consistency radius
     or none, path cutoff, k range, importance filter, overlap, exchange shifts, an ion now and then).
  B. optional (--frames): absorbers of the frames of a bcc tungsten molecular-dynamics run that you supply as an npz
     file, with fixed cards (SCF 4.0, RPATH 9.4, EXAFS 38, lmax 56): large disordered clusters at high k.
For every case FEFF's six modules run on the feff.inp (a feff85exafs build, FEFF_HOME; fortran/README.md) and the GPU chain runs on the
same file; chi is compared on chi.dat's own grid, the potentials (Fermi level, interstitial potential and density),
the partial-wave limits and the path counts alongside. Output: validation_runs/<tag>/<case>/ (inputs,
FEFF outputs, chi arrays, comparison json), summary.json, summary.md and summary.png.
"""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEFF = Path(os.environ.get("FEFF_HOME", ROOT / "fortran"))
sys.path.insert(0, str(ROOT / "src"))

# covalent radii (A), used only to set sensible interatomic distances
RADII = {6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 19: 2.03,
         20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
         31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54,
         44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 55: 2.44,
         56: 2.15, 57: 2.07, 58: 2.04, 72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36,
         80: 1.32, 82: 1.46, 83: 1.48}
SYMBOL = {6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca",
          21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge",
          33: "As", 34: "Se", 35: "Br", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd",
          47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 72: "Hf",
          73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 82: "Pb", 83: "Bi"}
LIGHT = [6, 7, 8, 9]                                  # neighbours mostly; absorbers now and then
HEAVY = [z for z in RADII if z >= 11]
EDGE_NAMES = {1: "K", 2: "L1", 3: "L2", 4: "L3", 5: "M1", 6: "M2", 7: "M3", 8: "M4", 9: "M5"}
W_CARDS = ["CONTROL 1 1 1 1 1 1", "PRINT 0 0 0 0 0 0", "EDGE K", "S02 1.0", "SCF 4.0 0 30 .2 1", "RPATH 9.4", "EXAFS 38.0",
           "AFOLP 1.15", "NLEG 4", "CRITERIA 0.0 0.5"]

# structures: cubic cell, basis in fractional coordinates with a species slot each, and the nearest-neighbour
# distance in units of the lattice constant (between the two slots named)
STRUCTURES = {
    "simple cubic": dict(basis=[((0, 0, 0), 0)], nn=1.0, pair=(0, 0), nsp=1),
    "bcc": dict(basis=[((0, 0, 0), 0), ((.5, .5, .5), 0)], nn=3 ** .5 / 2, pair=(0, 0), nsp=1),
    "CsCl": dict(basis=[((0, 0, 0), 0), ((.5, .5, .5), 1)], nn=3 ** .5 / 2, pair=(0, 1), nsp=2),
    "fcc": dict(basis=[((0, 0, 0), 0), ((.5, .5, 0), 0), ((.5, 0, .5), 0), ((0, .5, .5), 0)], nn=2 ** .5 / 2, pair=(0, 0), nsp=1),
    "diamond": dict(basis=[(p, 0) for p in ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))] +
                    [((p[0] + .25, p[1] + .25, p[2] + .25), 0) for p in ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))],
                    nn=3 ** .5 / 4, pair=(0, 0), nsp=1),
    "zincblende": dict(basis=[(p, 0) for p in ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))] +
                       [((p[0] + .25, p[1] + .25, p[2] + .25), 1) for p in ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))],
                       nn=3 ** .5 / 4, pair=(0, 1), nsp=2),
    "rock salt": dict(basis=[(p, 0) for p in ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))] +
                      [((p[0] + .5, p[1], p[2]), 1) for p in ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))],
                      nn=0.5, pair=(0, 1), nsp=2),
    "fluorite": dict(basis=[(p, 0) for p in ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))] +
                     [((x, y, z), 1) for x in (.25, .75) for y in (.25, .75) for z in (.25, .75)],
                     nn=3 ** .5 / 4, pair=(0, 1), nsp=2),
    "perovskite": dict(basis=[((0, 0, 0), 0), ((.5, .5, .5), 1), ((.5, .5, 0), 2), ((.5, 0, .5), 2), ((0, .5, .5), 2)],
                       nn=0.5, pair=(1, 2), nsp=3),
}


def lattice_cluster(struct, a, rcl, rng):
    """Ideal lattice points within rcl + 2 A of a random origin: positions (n, 3) and slot (n,)."""
    S = STRUCTURES[struct]
    m = int(np.ceil((rcl + 2) / a)) + 1
    cells = np.stack(np.meshgrid(*[np.arange(-m, m + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pos, slot = [], []
    for frac, s in S["basis"]:
        pos.append((cells + np.array(frac)) * a)
        slot.append(np.full(len(cells), s))
    pos, slot = np.concatenate(pos), np.concatenate(slot)
    pos = pos - rng.uniform(0, a, 3)                                  # random origin inside the cell
    keep = np.linalg.norm(pos, axis=1) <= rcl + 2
    return pos[keep], slot[keep]


def random_packing(species, radii, rcl, rng, scale):
    """Random sequential addition in a sphere: an absorber at the origin, then random atoms at least
    scale x (r_i + r_j) from every atom already placed; stops when 4000 trials in a row fail."""
    pos, slot = [np.zeros(3)], [0]
    fails = 0
    while fails < 4000 and len(pos) < 600:
        s = int(rng.integers(len(species)))
        p = rng.uniform(-rcl, rcl, 3)
        if np.linalg.norm(p) > rcl:
            continue
        P = np.array(pos)
        dmin = scale * (radii[s] + radii[np.array(slot)])
        if (np.linalg.norm(P - p, axis=1) >= dmin).all():
            pos.append(p); slot.append(s); fails = 0
        else:
            fails += 1
    return np.array(pos), np.array(slot)


def available_holes(z):
    f = Path(os.environ.get("FEFFGPU_ATOMS", ROOT / "src/feffgpu/data/atoms")) / f"Z{z:03d}.npz"
    return [int(h) for h in np.load(f)["holes"]] if f.exists() else [1]


def pick_edge(z, rng):
    holes = set(available_holes(z))
    w = {1: 0.55 if z <= 60 else 0.1}
    if z >= 29:
        w.update({4: 0.3, 3: 0.05, 2: 0.05})
    if z >= 70:
        w.update({9: 0.1, 8: 0.03})
    w = {h: v for h, v in w.items() if h in holes}
    hs, ps = list(w), np.array(list(w.values()))
    return int(hs[rng.choice(len(hs), p=ps / ps.sum())])


def synthetic_case(idx, rng):
    """One random environment: returns dict(title, cards, potentials [(ipot, Z, label)], atoms [(x, y, z, ipot)], meta)."""
    nsp = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
    structs = [k for k, v in STRUCTURES.items() if v["nsp"] == nsp] + ["random packing"]
    struct = str(rng.choice(structs))
    pool = HEAVY + (LIGHT if nsp > 1 else [])
    species = list(rng.choice(pool, nsp, replace=False))
    if nsp > 1 and rng.random() < 0.5 and species[-1] not in LIGHT:        # oxides and the like are common
        species[-1] = int(rng.choice(LIGHT))
    species = [int(z) for z in species]
    radii = np.array([RADII[z] for z in species])
    scale = float(rng.uniform(0.95, 1.12))
    rcl = float(rng.uniform(7.0, 10.5))
    if struct == "random packing":
        pos, slot = random_packing(species, radii, rcl, rng, float(rng.uniform(0.9, 1.05)))
        a, dnn = None, None
    else:
        S = STRUCTURES[struct]
        dnn = scale * (radii[S["pair"][0]] + radii[S["pair"][1]])
        a = dnn / S["nn"]
        pos, slot = lattice_cluster(struct, a, rcl, rng)
        # absorber: a random atom of a random slot near the origin
        s_abs = int(rng.integers(nsp))
        cand = np.nonzero(slot == s_abs)[0]
        i0 = cand[np.argmin(np.linalg.norm(pos[cand], axis=1))]
        pos = pos - pos[i0]
        order = np.concatenate([[i0], np.delete(np.arange(len(pos)), i0)])
        pos, slot = pos[order], slot[order]
    mods = {}
    if rng.random() < 0.5 and struct != "random packing":
        eps = rng.uniform(-0.05, 0.05, (3, 3)); eps = (eps + eps.T) / 2
        pos = pos @ (np.eye(3) + eps)
        mods["strain"] = [round(float(v), 4) for v in np.linalg.eigvalsh(eps)]
    sd = float(rng.choice([0.0, rng.uniform(0.01, 0.06), rng.uniform(0.06, 0.15)], p=[0.2, 0.5, 0.3]))
    dmin_ref = dnn if dnn else float(scale * 2 * radii.min())
    for _ in range(50):
        p = pos + rng.normal(0, sd, pos.shape) * (sd > 0)
        p[0] = 0.0
        d = np.linalg.norm(p[1:], axis=1)
        from scipy.spatial.distance import pdist
        if d.min() > 0.6 * dmin_ref and pdist(p).min() > 0.6 * dmin_ref:
            break
    else:
        sd, p = 0.0, pos.copy()
    pos = p
    mods["disorder_sd"] = round(sd, 4)
    pv = float(rng.choice([0.0, rng.uniform(0.0, 0.12)], p=[0.5, 0.5]))
    if pv > 0:
        keep = np.concatenate([[True], rng.random(len(pos) - 1) > pv])
        pos, slot = pos[keep], slot[keep]
        mods["vacancy_fraction"] = round(pv, 3)
    if rng.random() < 0.2:
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        h = float(rng.uniform(0.3, 2.5))
        keep = pos @ n <= h
        pos, slot = pos[keep], slot[keep]
        mods["surface_height"] = round(h, 2)
    d = np.linalg.norm(pos, axis=1)
    keep = d <= rcl
    pos, slot, d = pos[keep], slot[keep], d[keep]
    if len(pos) > 500:                                             # FEFF's atom limit is 1000; keep the runs short
        rcl = float(np.sort(d)[500] - 1e-3)
        keep = d <= rcl
        pos, slot, d = pos[keep], slot[keep], d[keep]
    if len(pos) < 12:
        return None
    order = np.argsort(d, kind="stable")
    pos, slot, d = pos[order], slot[order], d[order]
    # potentials: 0 the absorber, then one per slot that has atoms (the absorber's slot included)
    z_abs = species[int(slot[0])]
    used = sorted(set(int(s) for s in slot[1:]))
    ipot_of = {s: k + 1 for k, s in enumerate(used)}
    potentials = [(0, z_abs, SYMBOL[z_abs])] + [(ipot_of[s], species[s], SYMBOL[species[s]]) for s in used]
    atoms = [(float(x), float(y), float(zc), 0 if i == 0 else ipot_of[int(slot[i])]) for i, (x, y, zc) in enumerate(pos)]
    ihole = pick_edge(z_abs, rng)
    # cards
    cards = ["CONTROL 1 1 1 1 1 1", "PRINT 0 0 0 0 0 0", f"EDGE {EDGE_NAMES[ihole]}", "S02 1.0"]
    ratmin = float(d[1])
    scf = None
    if rng.random() < 0.7:
        rfms = float(rng.uniform(max(3.0, ratmin + 0.2), max(5.2, ratmin + 1.0)))
        # FEFF's self-consistency cluster holds at most 100 atoms (nclusx)
        while (d <= rfms).sum() > 90 and rfms > ratmin + 0.3:
            rfms -= 0.2
        rfms = round(rfms, 2)
        cards.append(f"SCF {rfms} 0 30 .2 1")
        scf = rfms
    rmax = round(float(rng.uniform(5.0, min(8.0, rcl - 1.0))), 2)
    cards.append(f"RPATH {rmax}")
    kmax = float(rng.choice([15.0, 20.0, 25.0, 30.0]))
    cards.append(f"EXAFS {kmax}")
    if rng.random() < 0.5:
        cards.append("AFOLP 1.15")
    pw = float(rng.choice([0.5, 1.0, 2.0, 4.0]))
    cards.append(f"CRITERIA 0.0 {pw}")
    cards.append("NLEG 4")
    exch = None
    ixc = int(rng.choice([0, 2], p=[0.9, 0.1]))
    if ixc == 2 or rng.random() < 0.3:
        vr0 = round(float(rng.uniform(-3, 3)), 2) if rng.random() < 0.7 else 0.0
        vi0 = round(float(rng.uniform(0, 2)), 2) if rng.random() < 0.7 else 0.0
        cards.append(f"EXCHANGE {ixc} {vr0} {vi0}")
        exch = (ixc, vr0, vi0)
    ion = None
    if rng.random() < 0.1:
        ip = int(rng.integers(len(potentials)))
        xion = float(rng.choice([-1.0, 1.0, 0.5]))
        cards.append(f"ION {potentials[ip][0]} {xion}")
        ion = (potentials[ip][0], xion)
    desc = f"{'-'.join(SYMBOL[z] for z in species)} {struct}, {EDGE_NAMES[ihole]} edge of {SYMBOL[z_abs]}, {len(atoms)} atoms"
    meta = dict(family="A", species=species, structure=struct, absorber_Z=z_abs, edge=EDGE_NAMES[ihole], ihole=ihole,
                n_atoms=len(atoms), cluster_radius=round(rcl, 2), lattice_constant=None if a is None else round(float(a), 4),
                nearest=round(ratmin, 4), modifiers=mods, scf=scf, rpath=rmax, kmax=kmax, pwcrit=pw, afolp="AFOLP 1.15" in cards,
                exchange=exch, ion=ion, lmax_override=None)
    return dict(title=f"random environment A{idx:03d}: {desc}", cards=cards, potentials=potentials, atoms=atoms, meta=meta)


def md_case(idx, frames, rng, rcut=10.5):
    z = frames
    pos, box = z["positions"], float(z["box"])
    F, C, n, _ = pos.shape
    f, c, a = int(rng.integers(F)), int(rng.integers(C)), int(rng.integers(n))
    rel = pos[f, c] - pos[f, c, a]
    rel -= box * np.round(rel / box)
    r = np.linalg.norm(rel, axis=1)
    order = np.argsort(r, kind="stable")
    keep = order[r[order] < rcut]
    atoms = [(float(rel[i, 0]), float(rel[i, 1]), float(rel[i, 2]), 0 if i == a else 1) for i in keep]
    meta = dict(family="B", species=[74], structure="bcc W, molecular-dynamics frame", absorber_Z=74, edge="K", ihole=1, n_atoms=len(atoms),
                frame=f, copy=c, absorber=a, cluster_radius=rcut, scf=4.0, rpath=9.4, kmax=38.0, pwcrit=0.5, afolp=True,
                exchange=None, ion=None, lmax_override=56, modifiers={})
    return dict(title=f"random environment B{idx:03d}: W MD frame {f} copy {c} absorber {a}", cards=list(W_CARDS),
                potentials=[(0, 74, "W"), (1, 74, "W")], atoms=atoms, meta=meta)


def write_input(case, path):
    lines = [f"TITLE {case['title']}"] + case["cards"] + ["", "POTENTIALS"]
    lines += [f"    {ip:2d}  {z:3d}   {lab}" for ip, z, lab in case["potentials"]]
    lines += ["", "ATOMS"]
    lines += [f"  {x:11.6f} {y:11.6f} {z:11.6f}  {ip}" for x, y, z, ip in case["atoms"]]
    lines += ["END", ""]
    path.write_text("\n".join(lines))


def run_feff(job):
    """FEFF's six modules in the case directory. Returns (name, seconds per module or an error string)."""
    d, lmax = job
    d = Path(d)
    env = dict(os.environ, LD_LIBRARY_PATH=str(FEFF / "lib"))
    env.pop("FEFF_LMAX", None)
    if lmax:
        env["FEFF_LMAX"] = str(lmax)
    times = {}
    for m in ("rdinp", "pot", "xsph", "pathfinder", "genfmt", "ff2x"):
        t0 = time.time()
        try:
            r = subprocess.run([str(FEFF / "bin" / f"feff8l_{m}")], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return d.name, f"{m}: timeout"
        (d / f"{m}.log").write_text(r.stdout + r.stderr)
        times[m] = round(time.time() - t0, 2)
        if r.returncode != 0 or (m == "ff2x" and not (d / "chi.dat").exists()):
            return d.name, f"{m}: exit {r.returncode}: {(r.stdout + r.stderr)[-400:]}"
    for f in d.glob("feff0*.dat"):
        f.unlink()
    return d.name, times


def read_chi(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    tab = np.array([[float(r[0]), float(r[1])] for r in rows if len(r) >= 2 and r[0].replace(".", "", 1).isdigit()])
    return tab[:, 0], tab[:, 1]


def feff_path_counts(path):
    """{nleg: [entries, summed degeneracy]} from paths.dat (FEFF merges symmetry-equivalent paths into one entry)."""
    counts = {}
    for l in Path(path).read_text().splitlines():
        if "index, nleg" in l:
            w = l.split()
            nleg, deg = int(w[1]), float(w[2])
            c = counts.setdefault(nleg, [0, 0.0])
            c[0] += 1; c[1] += deg
    return counts


def compare(d, meta, out):
    """chi on chi.dat's grid, potentials, partial-wave limits, path counts. Returns a json-able dict."""
    from feffgpu.pad import read_pot_pad, read_phase_pad
    from feffgpu.compare import r_factor
    kf_ref, chi_ref = read_chi(d / "chi.dat")
    idx = np.round(kf_ref / 0.05).astype(int) - 1
    ok = (idx >= 0) & (idx < len(out["chi"]))
    kf_ref, chi_ref, idx = kf_ref[ok], chi_ref[ok], idx[ok]
    chi_gpu = out["chi"][idx]
    win = (kf_ref >= 2.0) & (kf_ref <= kf_ref.max() - 0.5)
    w = kf_ref[win] ** 2
    Rk = float(np.sum((w * (chi_ref[win] - chi_gpu[win])) ** 2) / np.sum((w * chi_ref[win]) ** 2))
    maxrel = float(np.abs(chi_ref[win] - chi_gpu[win]).max() / np.abs(chi_ref[win]).max())
    k0 = np.concatenate([[0.0], kf_ref]); z0 = lambda c: np.concatenate([[0.0], c])
    Rr = float(r_factor(k0, z0(chi_ref), z0(chi_gpu), kmin=2.0, kmax=float(kf_ref.max() - 0.5), rmin=1.0, rmax=float(meta["rpath"])))
    # the usual analysis window: k 3 to min(23, kmax - 1), R 1 to the path cutoff
    Rw = float(r_factor(k0, z0(chi_ref), z0(chi_gpu), kmin=3.0, kmax=float(min(23.0, kf_ref.max() - 1.0)), rmin=1.0, rmax=float(meta["rpath"])))
    ref_pot = read_pot_pad(d / "pot.pad"); ref_ph = read_phase_pad(d / "phase.pad")
    pot, ph = out["pot"], out["phase"]
    E = min(ph.ne1, ref_ph.ne1)
    lmax_gpu = [int(v) for v in ph.lmax[:E].max(0)]
    lmax_feff = [int(v) for v in ref_ph.lmax[:E].max(0)]
    Lc = min(ph.ltot, ref_ph.ltot)                                      # the two files may store different l ranges
    pa = ph.ph[:E, ph.ltot - Lc:ph.ltot + Lc + 1, 0, :]
    pb = ref_ph.ph[:E, ref_ph.ltot - Lc:ref_ph.ltot + Lc + 1, 0, :]
    dph = np.abs(np.exp(2j * pa) - np.exp(2j * pb))
    mask = np.abs(pb) > 1e-3
    scf_iter = sum(1 for l in (d / "pot.log").read_text().splitlines() if "mu_new" in l)
    np.savez_compressed(d / "chi_compare.npz", k=kf_ref, chi_feff=chi_ref, chi_gpu=chi_gpu)
    return dict(R_k=Rk, R_r=Rr, R_window=Rw, max_rel=maxrel, n_k=int(len(kf_ref)),
                xmu_diff_ha=float(pot.xmu - ref_pot.xmu), vint_diff_ha=float(pot.vint - ref_pot.vint),
                rhoint_rel=float((pot.rhoint - ref_pot.rhoint) / ref_pot.rhoint), rmt_max_diff=float(np.abs(pot.rmt - ref_pot.rmt).max()),
                phase_max_diff=float(dph[mask].max()) if mask.any() else 0.0,
                lmax_gpu=lmax_gpu, lmax_feff=lmax_feff, scf_iterations_feff=scf_iter,
                paths_feff=feff_path_counts(d / "paths.dat"), paths_gpu={int(k): [int(v[0].shape[0]), float(v[2].sum())] for k, v in out["kept"].items()},
                timing_gpu={k: round(v, 1) for k, v in out["timing"].items()})


def summarize(cases, out_dir):
    rows = []
    for c in cases:
        m, r = c["meta"], c.get("result")
        status = c.get("status", "ok")
        mods = ", ".join(f"{k} {v}" for k, v in m.get("modifiers", {}).items())
        extra = []
        if m.get("exchange"): extra.append("EXCHANGE " + " ".join(str(v) for v in m["exchange"]))
        if m.get("ion"): extra.append(f"ION {m['ion'][0]} {m['ion'][1]}")
        desc = f"{'-'.join(SYMBOL[z] for z in m['species'])} {m['structure']}"
        if r:
            pf, pg = r["paths_feff"], r["paths_gpu"]
            tot = lambda d: sum((v[1] if isinstance(v, list) else v) for v in d.values())        # older case files: entries only
            paths = f"{tot(pf):.0f} / {tot(pg):.0f}"
            rows.append(f"| {c['name']} | {desc} | {m['edge']} of {SYMBOL[m['absorber_Z']]} | {m['n_atoms']} | {mods or '-'} | "
                        f"{'SCF %.1f' % m['scf'] if m['scf'] else 'no SCF'}, RPATH {m['rpath']}, k {m['kmax']:.0f}, pw {m['pwcrit']}"
                        f"{', ' + ', '.join(extra) if extra else ''} | {paths} | {r['R_window']:.1e} | {r['R_k']:.1e} | {r['max_rel']:.1e} | "
                        f"{r['xmu_diff_ha']:.1e} | {'same' if r['lmax_gpu'] == r['lmax_feff'] else str(r['lmax_gpu']) + ' vs ' + str(r['lmax_feff'])} |")
        else:
            rows.append(f"| {c['name']} | {desc} | {m['edge']} of {SYMBOL[m['absorber_Z']]} | {m['n_atoms']} | {mods or '-'} | "
                        f"{', '.join(extra) if extra else '-'} | - | {status} | | | | |")
    head = ("| case | environment | edge | atoms | modifiers | cards | paths (total degeneracy) FEFF / GPU | R (R-space, k 3-23) | R (k-space, k>=2, all k) | "
            "max diff / max | Fermi level diff (Ha) | partial-wave limits |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    ok = [c["result"] for c in cases if c.get("result")]
    lines = [f"# GPU FEFF versus FEFF on {len(cases)} random environments ({len(ok)} compared, {len(cases) - len(ok)} failed)", ""]
    if ok:
        Rk = np.array([r["R_k"] for r in ok]); Rw = np.array([r["R_window"] for r in ok])
        lines += [f"R-factor in the analysis window (R-space, k 3 to min(23, kmax-1), R 1 to the path cutoff): median {np.median(Rw):.1e}, "
                  f"worst {Rw.max():.1e}; {(Rw < 1e-5).sum()} of {len(Rw)} below 1e-5, {(Rw < 1e-4).sum()} below 1e-4.",
                  f"R-factor over the whole k range (k-space, k^2 weight, k >= 2): median {np.median(Rk):.1e}, worst {Rk.max():.1e}.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n" + head + "\n".join(rows) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(cases, indent=1, default=float))


def plot(cases, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [c for c in cases if c.get("result")]
    if not ok:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    for fam, col in (("A", "#2a78d6"), ("B", "#eb6834")):
        sel = [c for c in ok if c["meta"]["family"] == fam]
        if sel:
            ax.scatter([ok.index(c) for c in sel], [c["result"]["R_window"] for c in sel], s=22, color=col,
                       label="synthetic random environments" if fam == "A" else "W MD-frame absorbers")
    ax.set_yscale("log"); ax.set_xlabel("case"); ax.set_ylabel("R-factor, GPU FEFF vs FEFF (k 3-23, R-space)")
    ax.axhline(1e-5, color="#8a8984", lw=0.8, ls="--"); ax.grid(color="#e4e3df"); ax.legend(frameon=False, fontsize=8.5)
    worst = max(ok, key=lambda c: c["result"]["R_window"])
    z = np.load(Path(worst["dir"]) / "chi_compare.npz")
    ax = axes[1]
    ax.plot(z["k"], z["k"] ** 2 * z["chi_feff"], color="#0b0b0b", lw=1.6, label="FEFF")
    ax.plot(z["k"], z["k"] ** 2 * z["chi_gpu"], color="#2a78d6", lw=0.9, label="GPU FEFF")
    ax.plot(z["k"], 100 * z["k"] ** 2 * (z["chi_gpu"] - z["chi_feff"]), color="#eb6834", lw=0.9, label="difference x 100")
    ax.set_xlabel("k (1/A)"); ax.set_ylabel("k^2 chi(k)")
    ax.set_title(f"worst case {worst['name']}: {worst['meta']['structure']}, R {worst['result']['R_window']:.1e}", fontsize=8.5)
    ax.grid(color="#e4e3df"); ax.legend(frameon=False, fontsize=8.5)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "summary.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="random")
    ap.add_argument("--n-synthetic", type=int, default=30)
    ap.add_argument("--n-md", type=int, default=6)
    ap.add_argument("--frames", default="", help="frames.npz of a molecular-dynamics run for the B family (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="FEFF processes in parallel")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="", help="comma-separated case names to (re)run")
    ap.add_argument("--reuse-feff", action="store_true", help="keep FEFF's outputs of a case directory that already has chi.dat")
    args = ap.parse_args()
    out_dir = ROOT / "validation_runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cases = []
    i = 0
    while len(cases) < args.n_synthetic:
        c = synthetic_case(i, rng)
        i += 1
        if c is not None:
            c["name"] = f"A{len(cases):03d}"
            cases.append(c)
    if args.n_md and Path(args.frames).exists():
        frames = np.load(args.frames)
        for j in range(args.n_md):
            c = md_case(j, frames, rng)
            c["name"] = f"B{j:03d}"
            cases.append(c)
    if args.only:
        names = set(args.only.split(","))
        cases = [c for c in cases if c["name"] in names]
    todo = []
    for c in cases:
        d = out_dir / c["name"]
        c["dir"] = str(d)
        if args.reuse_feff and (d / "chi.dat").exists():
            c["reuse"] = True
            continue
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        write_input(c, d / "feff.inp")
        todo.append(c)
        print(f"{c['name']}: {c['title'][len('random environment Axxx: '):]} | {' ; '.join(c['cards'][2:])}", flush=True)
    # FEFF on every case, in parallel on the CPU
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = dict(ex.map(run_feff, [(c["dir"], c["meta"]["lmax_override"]) for c in todo]))
    for c in cases:
        if c.get("reuse"):
            old = Path(c["dir"]) / "case.json"
            c["timing_feff"] = json.loads(old.read_text()).get("timing_feff", {}) if old.exists() else {}
            continue
        r = res[c["name"]]
        if isinstance(r, str):
            c["status"] = f"FEFF failed: {r}"
            print(f"{c['name']}: FEFF failed: {r[:200]}", flush=True)
            (Path(c["dir"]) / "case.json").write_text(json.dumps(c, indent=1, default=float))
        else:
            c["timing_feff"] = r
    print(f"FEFF done for {sum(1 for c in cases if 'timing_feff' in c)} of {len(cases)} cases in {time.time() - t0:.0f} s", flush=True)
    # the GPU chain, one case at a time
    from feffgpu.chain import run
    for c in cases:
        if "timing_feff" not in c:
            continue
        d = Path(c["dir"])
        os.environ.pop("FEFF_LMAX", None)
        if c["meta"]["lmax_override"]:
            os.environ["FEFF_LMAX"] = str(c["meta"]["lmax_override"])
        t0 = time.time()
        try:
            out = run(d / "feff.inp", device=args.device, lmax_override=c["meta"]["lmax_override"])
            c["result"] = compare(d, c["meta"], out)
            r = c["result"]
            print(f"{c['name']}: R {r['R_window']:.1e} in the window (k-space all k {r['R_k']:.1e}), max diff {r['max_rel']:.1e}, Fermi level diff "
                  f"{r['xmu_diff_ha']:.1e} Ha, paths (entries, total degeneracy) FEFF {r['paths_feff']} GPU {r['paths_gpu']}, lmax {r['lmax_gpu']} vs {r['lmax_feff']}, "
                  f"FEFF {sum(c['timing_feff'].values()):.0f} s, GPU {time.time() - t0:.0f} s", flush=True)
        except Exception as e:
            c["status"] = f"GPU chain failed: {e!r}"
            (d / "gpu_error.txt").write_text(traceback.format_exc())
            print(f"{c['name']}: GPU chain failed: {e!r}", flush=True)
        (d / "case.json").write_text(json.dumps(c, indent=1, default=float))
        summarize(all_cases(out_dir), out_dir)
    plot(all_cases(out_dir), out_dir)
    print((out_dir / "summary.md").read_text().split("\n")[0:3])


def all_cases(out_dir):
    """Every case with a case.json in the run folder (so a --only rerun updates the whole table)."""
    out = []
    for d in sorted(out_dir.glob("[AB][0-9][0-9][0-9]")):
        if (d / "case.json").exists():
            out.append(json.loads((d / "case.json").read_text()))
    return out


if __name__ == "__main__":
    main()
