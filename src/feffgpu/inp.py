"""feff.inp for the GPU FEFF: the cards FEFF's reader (RDINP/rdinp_l.f, iniall.f, setedg.f, setgam.f, ffsort.f)
understands for EXAFS, with FEFF's defaults, derived quantities (unique potentials, stoichiometry, model atoms,
core-hole width) and unit conversions. Returns a plain dictionary; the stages take what they need from it."""
from __future__ import annotations
import math
import numpy as np
from .pad import BOHR, HART
from .xsph import terp

EDGES = ["NO", "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1", "N2", "N3", "N4", "N5", "N6", "N7",
         "O1", "O2", "O3", "O4", "O5", "O6", "O7", "P1", "P2", "P3", "P4", "P5", "R1"]
_ZH = [[0.99, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 95.1], [0.99, 18.0, 22.0, 35.0, 50.0, 52.0, 75.0, 95.1],
       [0.99, 17.0, 28.0, 31.0, 45.0, 60.0, 80.0, 95.1], [0.99, 17.0, 28.0, 31.0, 45.0, 60.0, 80.0, 95.1],
       [0.99, 20.0, 28.0, 30.0, 36.0, 53.0, 80.0, 95.1], [0.99, 20.0, 22.0, 30.0, 40.0, 68.0, 80.0, 95.1],
       [0.99, 20.0, 22.0, 30.0, 40.0, 68.0, 80.0, 95.1], [0.99, 36.0, 40.0, 48.0, 58.0, 76.0, 79.0, 95.1],
       [0.99, 36.0, 40.0, 48.0, 58.0, 76.0, 79.0, 95.1], [0.99, 30.0, 40.0, 47.0, 50.0, 63.0, 80.0, 95.1],
       [0.99, 40.0, 42.0, 49.0, 54.0, 70.0, 87.0, 95.1], [0.99, 40.0, 42.0, 49.0, 54.0, 70.0, 87.0, 95.1],
       [0.99, 40.0, 50.0, 55.0, 60.0, 70.0, 81.0, 95.1], [0.99, 40.0, 50.0, 55.0, 60.0, 70.0, 81.0, 95.1],
       [0.99, 71.0, 73.0, 79.0, 86.0, 90.0, 95.0, 100.0], [0.99, 71.0, 73.0, 79.0, 86.0, 90.0, 95.0, 100.0]]
_GAMH = [[0.02, 0.28, 0.75, 4.8, 10.5, 21.0, 60.0, 105.0], [0.07, 3.9, 3.8, 7.0, 6.0, 3.7, 8.0, 19.0],
         [0.001, 0.12, 1.4, 0.8, 2.6, 4.1, 6.3, 10.5], [0.001, 0.12, 0.55, 0.7, 2.1, 3.5, 5.4, 9.0],
         [0.001, 1.0, 2.9, 2.2, 5.5, 10.0, 22.0, 22.0], [0.001, 0.001, 0.5, 2.0, 2.6, 11.0, 15.0, 16.0],
         [0.001, 0.001, 0.5, 2.0, 2.6, 11.0, 10.0, 10.0], [0.0006, 0.09, 0.07, 0.48, 1.0, 4.0, 2.7, 4.7],
         [0.0006, 0.09, 0.07, 0.48, 0.87, 2.2, 2.5, 4.3], [0.001, 0.001, 6.2, 7.0, 3.2, 12.0, 16.0, 13.0],
         [0.001, 0.001, 1.9, 16.0, 2.7, 13.0, 13.0, 8.0], [0.001, 0.001, 1.9, 16.0, 2.7, 13.0, 13.0, 8.0],
         [0.001, 0.001, 0.15, 0.1, 0.8, 8.0, 8.0, 5.0], [0.001, 0.001, 0.15, 0.1, 0.8, 8.0, 8.0, 5.0],
         [0.001, 0.001, 0.05, 0.22, 0.1, 0.16, 0.5, 0.9], [0.001, 0.001, 0.05, 0.22, 0.1, 0.16, 0.5, 0.9]]


def setgam(iz, ihole):
    """Core-hole lifetime width in eV (COMMON/setgam.f: Rahkonen-Krause tables, linear in log10)."""
    if ihole <= 0:
        return 0.0
    if ihole > 16:
        return 0.1
    gamkp = [math.log10(v) for v in _GAMH[ihole - 1]]
    return 10.0 ** terp(_ZH[ihole - 1], gamkp, 8, 1, float(iz))


def read_feff_inp(path):
    """Parse feff.inp. Returns a dict with FEFF's variables in its units after eu2au (bohr, Hartree), except rmax
    (Angstrom, for the path finder) and the atoms also kept in Angstrom (rat_ang)."""
    lines = open(path).read().splitlines()
    P = dict(title=[], ihole=1, s02=1.0, s02h=1.0, ixc=0, ixc0=-1, vr0=0.0, vi0=0.0, rgrd=float(np.float32(0.05)), xkmax=20.0, xkstep=0.07,
             critpw=2.5, critcw=4.0, nlegxx=10, rmax=-1.0, rfms1=-1.0, lfms1=0, nscmt=0, ca1=0.0, nmix=1, ecv=-40.0, icoul=0,
             iunf=0, nohole=-1, jumprm=0, inters=0, totvol=0.0, iafolp=0, folpx=1.15, lreal=0, iorder=2, pcritk=0.0, pcrith=0.0,
             sig2g=0.0, vrcorr=0.0, vicorr=0.0, le2=0, l2lp=0, ipol=0, ispin=0, rmult=1.0, control=[1] * 6, print_=[0] * 6,
             mpot=1, mphase=1, mpath=1, mfeff=1, mchi=1, ispec=0)
    nphx = 11
    iz, xion, folp, lmaxsc, lmaxph, xnatph, potlbl = [0] * (nphx + 1), [0.0] * (nphx + 1), [1.0] * (nphx + 1), [0] * (nphx + 1), \
        [0] * (nphx + 1), [0.0] * (nphx + 1), [""] * (nphx + 1)
    atoms, iphatx = [], []
    mode, ifolp, lxnat = 0, 0, 0
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        line = raw.split("*")[0] if raw.strip().startswith("*") else raw
        words = line.split()
        if not words:
            continue
        w = words[0][:4].upper()
        card = w if w in ("ATOM", "HOLE", "CONT", "EXCH", "ION", "TITL", "FOLP", "RPAT", "RMAX", "PRIN", "POTE", "NLEG", "CRIT",
                          "IORD", "PCRI", "SIG2", "CORR", "AFOL", "EXAF", "RGRI", "RPHA", "NOHO", "JUMP", "EDGE", "SCF", "INTE",
                          "S02", "RSIG", "MULT", "UNFR", "END", "OVER", "DEBY", "XANE", "FMS", "LDOS", "CFAV", "SPIN", "POLA",
                          "ELLI", "RMUL", "NSTA", "NOGE", "SS", "SIG3", "MBCO", "PLAS") else None
        if mode == 1 and card is None:
            x, y, z = float(words[0]), float(words[1]), float(words[2])
            ip = int(words[3])
            atoms.append((x, y, z))
            iphatx.append(ip)
            continue
        if mode == 3 and card is None:
            ip = int(words[0])
            iz[ip] = int(words[1])
            lmaxsc[ip] = 1 if iz[ip] < 6 else (2 if iz[ip] < 55 else 3)
            if len(words) >= 3:
                potlbl[ip] = words[2][:6]
            if len(words) >= 4 and 1 <= int(words[3]) <= 4:
                lmaxsc[ip] = int(words[3])
            lmaxph[ip] = 2 if iz[ip] < 6 else 3
            if len(words) >= 5 and 1 <= int(words[4]) <= 4:
                lmaxph[ip] = int(words[4])
            if len(words) >= 6:
                xnatph[ip] = float(words[5])
                lxnat = 1
            continue
        mode = 0
        if card is None:
            raise ValueError(f"unrecognised card in feff.inp: {raw!r}")
        if card == "ATOM":
            mode = 1
        elif card == "POTE":
            mode = 3
        elif card == "TITL":
            P["title"].append(line[6:].strip() if len(line) > 6 else "")
        elif card == "HOLE":
            P["ihole"] = int(words[1])
            if len(words) > 2:
                P["s02h"] = float(words[2])
        elif card == "EDGE":
            P["ihole"] = EDGES.index(words[1].upper()) if words[1].upper() in EDGES else int(words[1])
        elif card == "S02":
            P["s02"] = float(words[1])
        elif card == "CONT":
            vals = [int(v) for v in words[1:7]]
            P["mpot"], P["mphase"], P["mfms"], P["mpath"], P["mfeff"], P["mchi"] = (vals + [1] * 6)[:6]
        elif card == "PRIN":
            P["print_"] = [int(v) for v in words[1:7]]
        elif card == "EXCH":
            P["ixc"] = int(words[1])
            P["vr0"] = float(words[2])
            P["vi0"] = float(words[3])
            if len(words) > 4:
                P["ixc0"] = int(words[4])
        elif card == "ION":
            xion[int(words[1])] = float(words[2])
        elif card == "FOLP":
            ifolp = 1
            folp[int(words[1])] = float(words[2])
        elif card in ("RPAT", "RMAX"):
            P["rmax"] = float(words[1])
        elif card == "NLEG":
            P["nlegxx"] = int(words[1])
        elif card == "CRIT":
            P["critcw"], P["critpw"] = float(words[1]), float(words[2])
        elif card == "IORD":
            P["iorder"] = int(words[1])
        elif card == "PCRI":
            P["pcritk"], P["pcrith"] = float(words[1]), float(words[2])
        elif card == "SIG2":
            P["sig2g"] = float(words[1])
        elif card == "CORR":
            P["vrcorr"], P["vicorr"] = float(words[1]), float(words[2])
        elif card == "AFOL":
            P["folpx"] = float(words[1]) if len(words) >= 2 else 1.15
            P["afolp"] = True
        elif card == "EXAF":
            P["xkmax"] = float(words[1])
        elif card == "RGRI":
            P["rgrd"] = float(words[1])
        elif card == "RPHA":
            P["lreal"] = 2
        elif card == "RSIG":
            P["lreal"] = max(P["lreal"], 1)
        elif card == "NOHO":
            if P["nohole"] < 0:
                P["nohole"] = int(words[1]) if len(words) >= 2 else 0
        elif card == "JUMP":
            P["jumprm"] = 1
        elif card == "SCF":
            P["nscmt"], P["ca1"] = 30, 0.2
            P["rfms1"] = float(words[1])
            if len(words) > 2: P["lfms1"] = int(words[2])
            if len(words) > 3: P["nscmt"] = int(words[3])
            if len(words) > 4: P["ca1"] = float(words[4])
            if len(words) > 5: P["nmix"] = int(words[5])
            if len(words) > 6: P["ecv"] = float(words[6])
            if len(words) > 7: P["icoul"] = int(words[7])
            if P["nscmt"] <= 0 or P["nscmt"] > 30: P["nscmt"] = 30
            if P["lfms1"] > 0: P["lfms1"] = 1
            if P["ca1"] < 0: P["ca1"] = 0.0
            if P["ca1"] > 0.5: raise ValueError("SCF: ca1 too large")
            if P["ecv"] >= 0: P["ecv"] = -40.0
            P["nmix"] = min(max(P["nmix"], 1), 30)
        elif card == "INTE":
            P["inters"] = int(words[1])
            if len(words) >= 3: P["totvol"] = float(words[2])
        elif card == "MULT":
            P["le2"] = int(words[1])
            if len(words) > 2: P["l2lp"] = int(words[2])
        elif card == "UNFR":
            P["iunf"] = 1
        elif card == "RMUL":
            P["rmult"] = float(words[1])
        elif card == "END":
            break
        elif card in ("XANE", "FMS", "LDOS", "CFAV", "DEBY", "OVER", "SPIN", "POLA", "ELLI", "NSTA", "NOGE", "SS", "SIG3", "MBCO", "PLAS"):
            raise NotImplementedError(f"card {words[0]} is not supported by the GPU FEFF")
    if P["ixc0"] < 0:
        P["ixc0"] = 0
    if P["ixc"] % 10 >= 5 and P["rgrd"] > 0.03:
        P["rgrd"] = 0.03
    if P["ihole"] <= 0:
        raise ValueError("ihole must be positive")
    nph = max(ip for ip in range(nphx + 1) if iz[ip] > 0)
    if iz[0] <= 0:
        raise ValueError("no absorbing atom (unique potential 0)")
    for ip in range(nph + 1):
        if P["iunf"] == 0 and lmaxsc[ip] > 2:
            lmaxsc[ip] = 2
    natt = len(atoms)
    iphatx = list(iphatx)
    for ip in range(nph + 1):
        if lxnat == 0:
            xnatph[ip] = float(sum(1 for q in iphatx if q == ip))
        elif xnatph[ip] <= 0.01:
            xnatph[ip] = 0.01 if ip == 0 else xnatph[ip]
        if xnatph[ip] <= 0:
            xnatph[ip] = 1.0
    if lxnat != 0:
        for ip in range(1, nph + 1):
            xnatph[ip] = xnatph[ip] / xnatph[0]
        xnatph[0] = 1.0
    xnat = sum(xnatph[:nph + 1])
    iatabs = next(k for k, q in enumerate(iphatx) if q == 0)
    ratx = np.array(atoms, float) * P["rmult"]
    d = np.linalg.norm(ratx - ratx[iatabs], axis=1)
    others = d[np.arange(natt) != iatabs]
    ratmin, ratmax = float(others.min()), float(others.max())
    if P["totvol"] > 0:
        P["totvol"] = P["totvol"] * ratmin ** 3 * xnat
    if P["rfms1"] < ratmin:
        P["rfms1"] = -1.0
    if P["rmax"] <= 0:
        P["rmax"] = min(2.2 * np.float32(ratmin), 1.01 * np.float32(ratmax))
    P["gamach"] = setgam(iz[0], P["ihole"])                       # eV
    if P["s02"] == 1.0:
        P["s02"] = P["s02h"]
    P["rmax"] = float(np.float32(P["rmax"] * np.float32(P["rmult"])))
    P["rfms1"] = float(np.float32(P["rfms1"] * np.float32(P["rmult"])))
    if ifolp != 0:
        P["iafolp"] = -1
    if P["iafolp"] >= 0:
        folp = [P["folpx"]] * (nphx + 1)
    # ffsort: absorber first at the origin, the rest sorted by distance (selection sort, first minimum wins ties)
    rel = [(0.0, 0.0, 0.0, 0)]
    for k in range(natt):
        if k != iatabs and d[k] > 0.1:
            rel.append((ratx[k, 0] - ratx[iatabs, 0], ratx[k, 1] - ratx[iatabs, 1], ratx[k, 2] - ratx[iatabs, 2], iphatx[k]))
    rel = [list(r) for r in rel]
    for a in range(len(rel) - 1):
        r2min, imin = rel[a][0] ** 2 + rel[a][1] ** 2 + rel[a][2] ** 2, a
        for b in range(a + 1, len(rel)):
            r2 = rel[b][0] ** 2 + rel[b][1] ** 2 + rel[b][2] ** 2
            if r2 < r2min:
                r2min, imin = r2, b
        if imin != a:
            rel[a], rel[imin] = rel[imin], rel[a]
    rat_ang = np.array([r[:3] for r in rel])
    iphat = np.array([r[3] for r in rel], int)
    iatph = [0] * (nph + 1)
    for ip in range(nph + 1):
        rabs, found = 1.0e20, -1
        for k in range(len(rel)):
            if iphat[k] == ip:
                t = float(np.linalg.norm(rat_ang[k]))
                if t < rabs:
                    rabs, found = t, k
        iatph[ip] = found
    P.update(nph=nph, iz=iz[:nph + 1], xion=xion[:nph + 1], folp=folp[:nph + 1], lmaxsc=lmaxsc[:nph + 1], lmaxph=lmaxph[:nph + 1],
             xnatph=xnatph[:nph + 1], potlbl=potlbl[:nph + 1], nat=len(rel), rat_ang=rat_ang, rat=rat_ang / BOHR, iphat=iphat,
             iatph=iatph, gamach_ha=P["gamach"] / HART, ecv_ha=P["ecv"] / HART, vr0_ha=P["vr0"] / HART, vi0_ha=P["vi0"] / HART,
             rfms1_bohr=P["rfms1"] / BOHR, xkmax_bohr=P["xkmax"] * BOHR, xkstep_bohr=P["xkstep"] * BOHR, totvol_bohr=P["totvol"] / BOHR ** 3)
    return P
