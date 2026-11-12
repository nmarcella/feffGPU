"""The multiple scattering of every state from its own atoms: per-state path lists equal a full enumeration, a partial
search around moved atoms equals a full one, every state is scored with its own signal, and the misfit's gradient in
the positions agrees with finite differences."""
import numpy as np
import pytest
import torch
from rmc_helpers import GPU, DEVICE, PHASE_W, PHASE_ZNC, bcc, rock_salt
from feffgpu.exafs import FitWindow, k_grid
from feffgpu.exafs.edge import EdgeModel
from feffgpu.rmc import StatePaths, GradientObjective, ClosestDistances

pytestmark = [GPU, pytest.mark.gpu]
K = k_grid(24.0)
WIN = FitWindow(kmin=3.0, kmax=16.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.0)


def ms_edge(target=None):
    """A W edge with three-leg paths to 5 A; the filter at 0.2 % (the kicked bcc cell has no three-leg path above 2 %)."""
    sym, pos, cell = bcc(4)
    e = EdgeModel("W", K, np.zeros(len(K)) if target is None else target, WIN, str(PHASE_W), sym, cell, "W", rpath=5.0, nleg=3, criteria=0.2,
                  device=DEVICE)
    return e, pos, cell


def kicked(pos, seed, kick, states=1):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return torch.tensor(pos, device=DEVICE, dtype=torch.float32)[None] + kick * torch.randn(states, len(pos), 3, device=DEVICE, generator=g)


def rows(L):
    return sorted((int(a), *[int(v) for v in at], *[int(v) for v in ip], float(d))
                  for nleg in sorted(L) for a, at, ip, d in zip(L[nleg]["absorber"].tolist(), L[nleg]["atoms"].tolist(), L[nleg]["ipots"].tolist(), L[nleg]["deg"].tolist()))


def test_state_paths_equal_a_full_enumeration_and_a_partial_search_equals_a_full_one():
    e, pos, cell = ms_edge()
    p = kicked(pos, 7, 0.03, states=2)
    sp = StatePaths(e)
    lists = [sp.find(p[j]) for j in range(2)]
    sig = sp.signal(lists, p)
    n_paths = sum(StatePaths.count(L)[3] for L in lists)
    ref = e.ms_pass(p)                                                                        # the path finder and the path stage over all absorbers of both
    assert e.ms.counts == {3: n_paths} and n_paths > 1000
    scale = float(ref.abs().max())
    assert sig.shape == (2, len(K)) and scale > 0 and float((sig - ref).abs().max()) < 1e-4 * scale
    assert float((sig[0] - sig[1]).abs().max()) > 1e-2 * scale                                # two configurations, two signals
    assert float((sp.signal([lists[1]], p[1:2])[0] - sig[1]).abs().max()) < 1e-4 * scale       # alone or inside the batch: the same
    before = rows(lists[0])                                                                   # an atom jumps: only the absorbers near its old or new position are searched again
    moved = p[0].clone()
    old = moved[5].clone()
    moved[5] = old + torch.tensor([1.6, 0.4, -0.3], device=DEVICE)
    near = sp.near(torch.stack([old, moved[5]]), moved)
    assert 0 < near.numel() < e.nA and 5 in near.tolist()
    part = sp.refind(lists[0], moved, near)
    full = sp.find(moved)
    assert rows(part) == rows(full) and rows(full) != before
    assert rows(lists[0]) == before                                                           # the old list is not changed in place
    sig_full = sp.signal([full], moved[None])
    assert float((sp.signal([part], moved[None]) - sig_full).abs().max()) < 1e-4 * scale
    assert float((sp.signal([lists[0]], moved[None]) - sig_full).abs().max()) > 1e-3 * scale  # the old paths on the new atoms: another signal
    assert sp.refind(lists[0], moved, torch.zeros(0, dtype=torch.long)) is lists[0]
    assert sp.near(torch.zeros(0, 3), moved).numel() == 0


def test_every_state_is_scored_with_its_own_multiple_scattering():
    e0, pos, cell = ms_edge()
    truth = kicked(pos, 11, 0.05)
    sp0 = StatePaths(e0)
    chi_t = (e0.spectrum(e0.static_histograms(truth)) + sp0.signal([sp0.find(truth[0])], truth))[0].cpu().numpy()
    e, _, _ = ms_edge(chi_t)                                                                  # the data: single plus multiple scattering of a kicked lattice
    obj = GradientObjective([e], 1, ClosestDistances(e.symbols, {"W-W": 2.2}, cell, device=DEVICE), torch.ones(e.n, device=DEVICE))
    p = torch.cat([truth, kicked(pos, 12, 0.05, states=2)])                                   # three states; state 0 is the truth
    sp = StatePaths(e)
    lists = [sp.find(p[j]) for j in range(3)]
    ms = [sp.signal(lists, p)]
    R = obj.edge_ratios(p, ms)
    assert R.shape == (3, 1) and float(R[0, 0]) < 1e-6 and float(R[1:, 0].min()) > 1e-3       # the truth's own atoms and paths reproduce the data
    for j in range(3):
        Rj = e.residual(e.spectrum(e.static_histograms(p[j:j + 1]), e.tables, ms[0][j:j + 1]))
        assert float(Rj[0]) == pytest.approx(float(R[j, 0]), rel=1e-5, abs=1e-9)
    swapped = obj.edge_ratios(p, [ms[0][[1, 2, 0]]])                                          # another state's multiple scattering is not the state's own
    assert float(swapped[0, 0]) > 20 * float(R[0, 0]) and float(swapped[0, 0]) > 1e-5
    obj.ms = ms
    pg = p.clone().requires_grad_(True)
    M = obj.misfits(pg)
    assert torch.allclose(M.detach(), R[:, 0], rtol=1e-5, atol=1e-9)
    (g,) = torch.autograd.grad(M.sum(), pg)
    assert float(g[1:].abs().max()) > 0
    assert float(obj.closest.worst(p).max()) == 0.0                                           # nearest neighbors at 2.74 A, closest distance 2.2 A
    close = p[:1].clone()
    close[0, 1] = close[0, 0] + torch.tensor([1.2, 1.1, 0.0], device=DEVICE)
    d = close[0][:, None] - close[0][None]
    box = float(cell[0, 0])
    r = (d - box * torch.round(d / box)).norm(dim=-1) + 1e9 * torch.eye(e.n, device=DEVICE)
    assert float(r.min()) < 2.0 and float(obj.closest.worst(close)[0]) == pytest.approx(2.2 - float(r.min()), abs=1e-4)


def test_the_misfit_gradient_agrees_with_finite_differences():
    """A target with a planted sublattice shift; at a noisy start the gradient of the misfit in the positions agrees with a
    central difference along a random direction (at a perfect lattice every single-atom derivative vanishes by symmetry)."""
    e0, pos, cell = ms_edge()
    shifted = pos.copy()
    shifted[64:] += 0.15
    chi_t = e0.spectrum(e0.static_histograms(torch.tensor(shifted[None], device=DEVICE, dtype=torch.float32)))[0].cpu().numpy()
    e, _, _ = ms_edge(chi_t)
    obj = GradientObjective([e], 4, None, torch.ones(e.n, device=DEVICE))
    p = kicked(pos, 3, 0.02, states=4)
    ms = [torch.zeros(1, len(K), device=DEVICE)]
    M0, g = obj.state_misfit_and_gradient(p, ms)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(5)
    v = torch.randn(p.shape, device=DEVICE, generator=gen)
    v /= v.norm()
    h = 0.01
    fd = (obj.state_misfit_and_gradient(p + h * v, ms)[0] - obj.state_misfit_and_gradient(p - h * v, ms)[0]) / (2 * h)
    an = float((g * v).sum())
    assert abs(fd) > 1e-3 and abs(fd - an) < 0.15 * abs(fd) + 1e-4, (fd, an)


def test_multiple_scattering_rows_follow_the_absorbers_in_a_sheared_cell():
    """Two elements, the C atoms as absorbers, a monoclinic cell: one row per absorber, the evaluation of a subset agrees
    with the full pass, and per-state lists reproduce the edge's own pass."""
    sym, pos, cell = rock_salt(2)
    box = float(cell[0, 0])
    sheared = np.array([[box, 0, 0], [0.15 * box, box, 0], [0, 0, box]])
    e = EdgeModel("C", K, np.zeros(len(K)), FitWindow(kmin=3.0, kmax=12.0, rmin=1.0, rmax=4.5), str(PHASE_ZNC), sym, sheared, "C",
                  potentials={"Zn": 1, "C": 2}, rpath=4.0, nleg=3, criteria=1.0, device=DEVICE)
    rng = np.random.default_rng(1)
    p = torch.tensor(pos[None] + rng.normal(0, 0.05, (2, len(pos), 3)), device=DEVICE, dtype=torch.float32)
    ref = e.ms_pass(p)
    assert e.ms.rows.shape[0] == 2 * e.nA and 4 not in e.ms.lists
    sub = e.ms.abs_atoms[::5]
    assert float((e.ms.chi_absorbers(p.reshape(-1, 3), sub) - e.ms.rows[e.ms.row_of[sub]]).abs().max()) < 1e-5
    sp = StatePaths(e)
    sig = sp.signal([sp.find(p[0]), sp.find(p[1])], p)
    assert float((sig - ref).abs().max()) < 1e-4 * float(ref.abs().max())
