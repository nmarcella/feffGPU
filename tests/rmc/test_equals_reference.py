"""The search of feffgpu.rmc reproduces the original research launcher of the method, seed for seed.

How the reference numbers were made. The original launcher (a script of the research code from which feffgpu.rmc was
written, run unmodified) was run on the toy below with the same seed and settings, three times: without gradient
steps ("nograd"), with five gradient steps, exchanges and refits of S0^2 and the E0 shift ("grad"), and with hops
onto vacant sites as well ("hops"). The numbers in REFERENCE are read from its output files. No measured data are
involved: the toy is two fluorite-type cells of Zn, Cu and C (Cu shares the Zn potential of the A000 phase file; the
second edge uses the W phase file as a stand-in), and the "measured" spectra are the forward model of the cells with
random displacements, S0^2 0.9 / 0.85 and E0 shifts +1.0 / -1.5 eV. The original launcher demanded an exchange pair
and a fluorite vacancy sublattice, so the toy has them (a bcc block cannot run through it).

Tolerances. The original code is not bitwise reproducible on a GPU (atomic additions in the pair histograms and the
multiple scattering): run twice with the same seed it differed by up to 1.5e-4 A in the positions and 5.5e-4 in the
relative misfits. Without gradient steps the new code is bit-identical to it; with gradient steps the tolerances are
those of the GPU; every discrete decision (accepted moves, exchanges, hops, best state, which atom sits on which
site) must be identical.

The stated exception: a hop. The original placed a hopped atom on the site of a lattice re-centered on the current
positions of the cations; the new code places it on the site of the fixed start lattice (the lattice the displacement
limit uses; nothing else in the general library is re-centered). The same hops happen (the final arrangements are
identical when the runs have not separated), the positions of hopped atoms differ by the drift of that center (here
up to 3e-3 A) and the misfits of the sample with vacancies by up to 2e-3. A later acceptance decision close to its
threshold can then come out differently; the hop case is therefore compared strictly before the first hop and while
the decisions agree, and by its statistics otherwise.
"""
import json
import numpy as np
import pytest
from rmc_helpers import GPU, PHASE_W, PHASE_ZNC, fluorite_toy, toy_truth, toy_target, write_chi, write_poscar

REFERENCE = {'nograd': {'A': {'best': [1.507025268, 0.083166805, 0.080261239, 0.078266321, 0.076201979, 0.07460759],
                  'accepted': [4, 4, 4, 4, 3, 4],
                  'exchanges': [0, 0, 0, 0, 0, 0],
                  'hops': [0, 0, 0, 0, 0, 0],
                  'population_misfits': [0.075559546, 0.075048564, 0.074915609, 0.07460759],
                  'best_state': 3,
                  's02': {'C': 0.5, 'Zn': 0.5},
                  'de0': {'C': 0.520853, 'Zn': -0.022115},
                  'msd_from_ideal': 0.0004377788,
                  'arrangement': None},
            'B': {'best': [2.738708784, 0.265352411, 0.259092666, 0.255153389, 0.252853445, 0.247621044],
                  'accepted': [4, 4, 4, 4, 4, 4],
                  'exchanges': [0, 0, 0, 0, 0, 0],
                  'hops': [0, 0, 0, 0, 0, 0],
                  'population_misfits': [0.247621044, 0.252398431, 0.253830186, 0.251425574],
                  'best_state': 0,
                  's02': {'C': 0.5},
                  'de0': {'C': 0.520853},
                  'msd_from_ideal': 0.0004488033,
                  'arrangement': None}},
 'grad': {'A': {'best': [0.771510087, 0.041794932, 0.040916985, 0.039729713, 0.039505688, 0.03546464],
                'accepted': [4, 4, 4, 2, 4, 2],
                'exchanges': [0, 0, 1, 3, 2, 2],
                'hops': [0, 0, 0, 0, 0, 0],
                'population_misfits': [0.037037375, 0.03546464, 0.036051617, 0.035715816],
                'best_state': 1,
                's02': {'C': 0.5307766, 'Zn': 0.5119838},
                'de0': {'C': 1.963202, 'Zn': -0.836655},
                'msd_from_ideal': 0.5149272586,
                'arrangement': None},
          'B': {'best': [1.590238817, 0.086816776, 0.075700727, 0.069615531, 0.063428362, 0.057981075],
                'accepted': [4, 4, 4, 4, 4, 4],
                'exchanges': [0, 0, 1, 3, 2, 2],
                'hops': [0, 0, 0, 0, 0, 0],
                'population_misfits': [0.071209055, 0.062746955, 0.057981076, 0.060434419],
                'best_state': 2,
                's02': {'C': 0.5307766},
                'de0': {'C': 1.963202},
                'msd_from_ideal': 1.8218674093,
                'arrangement': None}},
 'hops': {'A': {'best': [1.195601522, 0.058263402, 0.052233987, 0.051781092, 0.050435205, 0.049097256],
                'accepted': [4, 4, 4, 4, 3, 3],
                'exchanges': [0, 1, 1, 3, 2, 1],
                'hops': [0, 4, 4, 4, 4, 4],
                'population_misfits': [0.051953085, 0.050373395, 0.049097256, 0.050328058],
                'best_state': 2,
                's02': {'C': 0.5, 'Zn': 0.5},
                'de0': {'C': 0.416459, 'Zn': -0.02186},
                'msd_from_ideal': 6.5300285722,
                'arrangement': [[[0, 2, 4, 31, 32, 33, 54, 55], [5, 7, 10, 13, 15, 22, 27, 34, 37, 40, 42, 49, 51, 52, 60, 62]],
                                [[0, 2, 4, 31, 32, 47, 51, 58], [5, 7, 10, 13, 15, 22, 30, 34, 37, 40, 42, 47, 51, 57, 60, 62]],
                                [[0, 2, 5, 31, 32, 47, 51, 54], [5, 7, 10, 13, 15, 22, 30, 34, 37, 40, 42, 47, 51, 57, 60, 62]],
                                [[0, 2, 4, 31, 32, 47, 51, 54], [5, 7, 10, 13, 15, 22, 30, 34, 37, 40, 42, 47, 51, 57, 60, 62]]]},
          'B': {'best': [2.376868645, 0.214463183, 0.200747247, 0.186400085, 0.175108862, 0.160638549],
                'accepted': [4, 4, 4, 4, 4, 4],
                'exchanges': [0, 1, 3, 1, 1, 2],
                'hops': [0, 0, 0, 0, 0, 0],
                'population_misfits': [0.166569329, 0.164545414, 0.160638549, 0.165480627],
                'best_state': 2,
                's02': {'C': 0.5},
                'de0': {'C': 0.416459},
                'msd_from_ideal': 1.0571732059,
                'arrangement': [[[], [2, 5, 13, 20, 25, 28, 34, 37, 47, 49, 51, 52, 54, 57, 60, 62]],
                                [[], [2, 5, 13, 15, 20, 28, 34, 37, 47, 49, 51, 52, 54, 57, 60, 62]],
                                [[], [2, 5, 13, 15, 20, 27, 34, 37, 47, 49, 51, 52, 54, 57, 60, 62]],
                                [[], [2, 5, 13, 15, 20, 28, 34, 37, 47, 49, 51, 52, 54, 57, 60, 62]]]}}}

CASES = {"nograd": dict(gradient_steps=0, from_generation=100, hop_p=0.0),
         "grad": dict(gradient_steps=5, from_generation=3, hop_p=0.0),
         "hops": dict(gradient_steps=3, from_generation=2, hop_p=1.0)}
TOL = {"nograd": dict(misfit=1e-6, s02=1e-6, de0=1e-5, msd=1e-7),
       "grad": dict(misfit=3e-3, s02=2e-3, de0=0.02, msd=2e-3),
       "hops": dict(misfit=1e-2, s02=2e-3, de0=0.02, msd=2e-3)}


@pytest.fixture(scope="module")
def toy_inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    for label, seed, n_vac, edges in (("A", 0, 8, ["C", "Zn"]), ("B", 2, 0, ["C"])):
        sym, pos, cell = fluorite_toy(seed, n_vac)
        write_poscar(d / f"start_{label}.vasp", sym, pos, cell)
        tr = toy_truth(pos, seed + 1)
        for e in edges:
            k, chi = toy_target(e, sym, tr, cell)
            write_chi(d / f"target_{label}_{e}.chi", k, chi)
    return d


def config(d, case, out):
    c = CASES[case]
    win = "{ kmin = 3.0, kmax = 12.0, dk = 2.0, kweight = 2, rmin = 1.0, rmax = 4.0 }"

    def edge(label, e):
        phase, pots = (PHASE_ZNC, "{ Zn = 1, Cu = 1, C = 2 }") if e == "C" else (PHASE_W, "{ Zn = 1, Cu = 1, C = 1 }")
        return (f'\n  [[sample.edge]]\n  name = "{e}"\n  absorber = "{e}"\n  chi = "target_{label}_{e}.chi"\n  phase = "{phase}"\n'
                f"  potentials = {pots}\n  window = {win}\n  rpath = 4.0\n  nleg = 3\n  criteria = 2.5\n")
    text = (f'[run]\noutput = "{out}"\nseed = 1\nsave_every = 3\n\n[[sample]]\nlabel = "A"\nstructure = "start_A.vasp"\n' + edge("A", "C") + edge("A", "Zn")
            + '\n[[sample]]\nlabel = "B"\nstructure = "start_B.vasp"\n' + edge("B", "C")
            + '\n[amplitude]\nmode = "fit"\ns02_bounds = [0.5, 1.5]\nde0_bounds = [-8.0, 8.0]\nrefit_every = 2\nrefit_from = 2\nfit_iterations = 40\n'
            + f'\n[search]\nstates = 4\ngenerations = 6\nstep = 0.005\nacceptance = 0.8\nfroze_in = 10\nmeans_from = 3\ngradient_steps = {c["gradient_steps"]}\n'
            + 'gradient_force = 1.0\nstart_kick = 0.01\nstart_distribution = "gaussian"\npaths_every = 3\n'
            + f'\n[moves]\nexchange = ["Zn", "Cu"]\nexchange_probability = 0.5\nfrom_generation = {c["from_generation"]}\n'
            + f'hops = {{ element = "C", sites = "fluorite", probability = {c["hop_p"]}, count = 1, jitter = 0.05 }}\n'
            + '\n[constraints]\nmax_displacement = 0.4\nbox_rule = "project"\nreject_trials = false\nclosest_tolerance = 0.05\nspring = 50.0\n'
            + 'closest_distance = { "C-C" = 2.3, "C-Zn" = 1.9, "C-Cu" = 1.9, "Zn-Zn" = 3.4, "Cu-Cu" = 3.4, "Cu-Zn" = 3.4 }\n')
    p = d / f"config_{case}.toml"
    p.write_text(text)
    return p


def arrangement(pos, sym, L=4, h=2.75):
    """Per state: the vacant anion sites and the cation sites of Cu, as grid indices of the fluorite lattice."""
    t0 = pos[0][np.nonzero(sym != "C")[0][0]]
    out = []
    for P in pos:
        m = np.mod(np.round((P[sym == "C"] - t0) / h - 0.5).astype(int), L)
        n = np.mod(np.round((P[sym == "Cu"] - t0) / h).astype(int), L)
        out.append([sorted(set(range(L ** 3)) - set(((m[:, 0] * L + m[:, 1]) * L + m[:, 2]).tolist())), sorted(((n[:, 0] * L + n[:, 1]) * L + n[:, 2]).tolist())])
    return out


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("case", ["nograd", "grad", "hops"])
def test_search_reproduces_the_original_launcher(case, toy_inputs, tmp_path):
    from feffgpu.rmc import run_search, validate_file
    cfg = config(toy_inputs, case, tmp_path / "run")
    assert validate_file(cfg) == []
    run_search(cfg, log=None)
    tol = TOL[case]
    for label, ref in REFERENCE[case].items():
        d = tmp_path / "run" / label
        hist = json.loads((d / "history.json").read_text())["generations"]
        rep = json.loads((d / "report.json").read_text())
        z = np.load(d / "state.npz")
        accepted, best = [g["accepted"] for g in hist], [g["best"] for g in hist]
        pop, ideal = z["population"].astype(float), z["ideal"].astype(float)
        if case == "hops":
            # before the first hop the runs are the same; afterwards a hopped atom sits up to a few 1e-3 A elsewhere, and a
            # later decision close to its threshold can come out differently: then only the statistics are compared
            first = CASES[case]["from_generation"] - 1
            assert accepted[:first] == ref["accepted"][:first], label
            np.testing.assert_allclose(best[:first], ref["best"][:first], rtol=1e-4, err_msg=label)
            np.testing.assert_allclose(best, ref["best"], rtol=0.05, err_msg=label)
            np.testing.assert_allclose(z["misfits"], ref["population_misfits"], rtol=0.05, err_msg=label)
            assert sum(g["hops"] for g in hist) == pytest.approx(sum(ref["hops"]), abs=4), label
            if accepted == ref["accepted"]:                                                # the trajectories did not separate: the same hops and exchanges happened
                assert [g["exchanges"] for g in hist] == ref["exchanges"] and [g["hops"] for g in hist] == ref["hops"], label
                assert rep["best_state"] == ref["best_state"], label
                np.testing.assert_allclose(best, ref["best"], rtol=tol["misfit"], err_msg=label)
                assert arrangement(pop[:, 0], z["symbols"].astype(str)) == ref["arrangement"], label
            continue
        assert accepted == ref["accepted"], label
        assert [g["exchanges"] for g in hist] == ref["exchanges"] and [g["hops"] for g in hist] == ref["hops"], label
        assert rep["best_state"] == ref["best_state"], label
        np.testing.assert_allclose(best, ref["best"], rtol=tol["misfit"], err_msg=label)
        np.testing.assert_allclose(z["misfits"], ref["population_misfits"], rtol=tol["misfit"], err_msg=label)
        for e in ref["s02"]:
            assert rep["final_refit"]["s02"][e] == pytest.approx(ref["s02"][e], rel=tol["s02"]), (label, e)
            assert rep["final_refit"]["de0"][e] == pytest.approx(ref["de0"][e], abs=tol["de0"]), (label, e)
        assert float(((pop - ideal[None]) ** 2).sum(-1).mean()) == pytest.approx(ref["msd_from_ideal"], rel=tol["msd"], abs=1e-9), label
