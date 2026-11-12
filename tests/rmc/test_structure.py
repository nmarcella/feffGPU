"""The model cell and its sites: reading structures with a supercell, the fluorite site generator, sites from a list,
and the arrangement statistics. CPU only."""
import numpy as np
import pytest
from rmc_helpers import fluorite_toy, rock_salt, write_poscar
from feffgpu.rmc import ModelCell, fluorite_sites, sublattice_sites, read_site_file, ArrangementStatistics


def test_model_cell_from_a_file_with_a_supercell(tmp_path):
    sym, pos, cell = rock_salt(1)
    write_poscar(tmp_path / "rs.vasp", sym, pos, cell)
    m = ModelCell.from_file(tmp_path / "rs.vasp", supercell=(2, 2, 1))
    assert m.n == 32 and np.allclose(np.diag(m.cell), np.diag(cell) * [2, 2, 1]) and m.elements == ["C", "Zn"]
    assert m.counts() == {"C": 16, "Zn": 16} and m.sites.m == 32 and list(m.sites.start_site_of) == list(range(32))
    assert "32 atoms" in m.describe()
    with pytest.raises(ValueError):
        ModelCell.from_file(tmp_path / "rs.vasp", supercell=(0, 1, 1))


def test_the_fluorite_generator():
    sym, pos, cell = fluorite_toy(0, 8)
    s = fluorite_sites(sym, pos, cell, "C")
    L = 4
    assert s.n_ids == 2 * L ** 3 and len(s.hop_sites) == L ** 3 and s.hop_element == "C"
    assert len(s.vacant_hop_sites(s.start_site_of)) == 8
    rel = s.positions[s.start_site_of] - pos
    assert np.allclose(rel - 11.0 * np.round(rel / 11.0), 0.0, atol=1e-9)                # the atoms of the ideal start sit on their sites (or images)
    cat_ids = s.ids[s.start_site_of[sym != "C"]]
    assert len(set(cat_ids.tolist())) == 32 and cat_ids.max() < L ** 3                    # cation identifiers on the grid, anions after
    assert s.ids[s.hop_sites].min() == L ** 3
    assert s.hop_neighbors.shape == (L ** 3, 6)


def test_sites_from_a_list_and_from_files(tmp_path):
    sym, pos, cell = fluorite_toy(0, 8)
    an_sites = fluorite_sites(sym, pos, cell, "C")
    listed = an_sites.positions[an_sites.hop_sites]
    s = sublattice_sites(sym, pos, cell, "C", listed)
    assert len(s.hop_sites) == 64 and len(s.vacant_hop_sites(s.start_site_of)) == 8 and s.n_ids == s.m
    assert (s.hop_neighbors >= 0).sum(1).max() == 6
    np.savetxt(tmp_path / "s.txt", listed)
    assert np.allclose(read_site_file(tmp_path / "s.txt"), listed)
    np.savetxt(tmp_path / "s.csv", listed, delimiter=",")
    assert np.allclose(read_site_file(tmp_path / "s.csv"), listed)
    write_poscar(tmp_path / "sites.vasp", np.array(["X"] * 0 + ["He"] * len(listed)), listed, cell)
    assert np.allclose(read_site_file(tmp_path / "sites.vasp"), listed)
    with pytest.raises(ValueError, match="no free site"):
        sublattice_sites(sym, pos, cell, "C", listed[:56] + 1.3)


def test_arrangement_statistics():
    sym, pos, cell = fluorite_toy(0, 8)
    model = ModelCell(sym, pos, cell, sites=fluorite_sites(sym, pos, cell, "C"))
    st = ArrangementStatistics(model, ("Zn", "Cu"))(model.sites.start_site_of[None])
    assert st["vacancies"] == 8 and st["vacancies_moved"] == 0 and st["atoms_off_start_site"] == 0 and st["fraction_Zn"] == 0.5
    # the Warren-Cowley parameter by hand: 12 cation neighbors at a / sqrt(2)
    cat = np.nonzero(sym != "C")[0]
    d = pos[cat][:, None] - pos[cat][None]
    d = np.linalg.norm(d - 11.0 * np.round(d / 11.0), axis=-1)
    nb = np.abs(d - 5.5 / np.sqrt(2)) < 1e-6
    zn = sym[cat] == "Zn"
    unlike = (nb & ~zn[None]).sum(1)[zn] / nb.sum(1)[zn]
    assert st["unlike_neighbors_of_Zn"] == pytest.approx(unlike.mean()) and st["warren_cowley_Zn"] == pytest.approx(1 - unlike.mean() / 0.5)
    assert 0 <= st["Zn_around_vacancies"] <= 1
