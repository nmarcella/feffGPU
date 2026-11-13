# feffgpu

FEFF 8.5L's EXAFS calculation rewritten in PyTorch for NVIDIA GPUs, with two tools built on it. One installation
gives all three:

| Tool | Command | What it does |
|---|---|---|
| **feffgpu** | `feffgpu` | a FEFF input file to chi(k): potentials with the self-consistency loop, phase shifts, the path finder and the path stage, validated against FEFF 8.5L; the path stage is batched over hundreds of thousands of paths and differentiable in the atom positions |
| **MD-EXAFS** | `md-exafs` | the EXAFS spectrum of any molecular dynamics trajectory, averaged over frames and absorbing atoms, with its decomposition by path type, radial distribution tools, and a comparison with a measured spectrum |
| **RMC-EXAFS** | `rmc-exafs` | reverse Monte Carlo fitting: the atoms of a model structure are moved until its spectra fit the measured ones, by a population search that follows the rules of EvAX and by gradient refinement |

EXAFS is the extended X-ray absorption fine structure; chi(k) is the EXAFS signal as a function of the
photoelectron wave number k; FEFF is the program that calculates chi(k) from atom positions.

**Documentation:** <https://feffgpu.readthedocs.io> (sources in [`docs/`](docs/)).

## The same output as FEFF 8.5L

`feffgpu` is validated against FEFF 8.5L itself: the same `feff.inp` goes through both codes and the two chi(k) are
compared. The measure is the R-factor between the two codes: the squared difference of the Fourier transforms of the
two spectra, divided by the squared transform of FEFF's.

**Every element, edge by edge.** Every element from hydrogen to einsteinium, with each of its K, L3 and M5 edges:
267 cases. In the periodic table below, the three squares of an element are its three edges, colored by the R-factor.
The inset plots every point of every `feffgpu` spectrum against FEFF's.

![feffgpu against FEFF 8.5L for the K, L3 and M5 edge of every element: a periodic table colored by the R-factor between the two codes, the spectra of one code against the other, and the R-factor against the atomic number](docs/validation/agreement_all_elements.png)

- The median R-factor between the two codes is 1.7e-6. 264 of the 267 cases are below 2e-5, the limit that the
  regression tests enforce; the three above it are at 1.0e-4, 4.6e-5 and 2.5e-5.
- The Fermi levels of the self-consistent potentials agree to 5e-8 hartree in the median, 1.4e-5 at most.
- The number of paths, weighted by degeneracy, is identical in 252 cases. In the other 15 the two codes differ by a few
  three- and four-leg paths that sit at the threshold of the path filter, never in the single-scattering paths.
- The recipe, the two spectra of every case drawn over each other, and the table are in
  [docs/validation/elements.md](docs/validation/elements.md).

**Mixed environments.** 34 environments drawn at random over elements, edges, structures and input cards, one panel
per absorbing element and edge, with FEFF 8.5L in black and `feffgpu` in blue on top, and above them the R-factor
between the two codes.

![feffgpu against FEFF 8.5L for 34 random environments: the R-factor between the two codes per element and edge, and the two spectra of every environment drawn over each other](docs/validation/agreement_by_element_and_edge.png)

- 24 absorbing elements from C to Au; K, L2, L3 and M5 edges; lattices and random packings with disorder, vacancies,
  strain and surfaces; self-consistent and overlapped potentials; a range of FEFF input cards.
- The median R-factor between the two codes is 1.3e-6, and 33 of the 34 are below 2e-5. The one above it (2.4e-3) is a
  random packing with an `ION` card.
- The number of paths, weighted by degeneracy, is identical in 29 of the 34. In the other 5 the two codes differ by
  one path that sits at the importance threshold. The Fermi levels agree to 2e-6 hartree or better.

The full table is in [docs/validation/random_environments.md](docs/validation/random_environments.md). The spectra
behind both figures are in `data/validation/`, and `tools/plot_validation_elements.py` and
`tools/plot_validation_agreement.py` redraw them.

## Please cite FEFF and EvAX

This package exists because FEFF exists, and its reverse Monte Carlo search follows the rules of EvAX. If you use
it, please cite that work as you would if you had used those programs themselves.

Always, FEFF:

- J. J. Rehr, J. J. Kas, F. D. Vila, M. P. Prange and K. Jorissen, Phys. Chem. Chem. Phys. **12**, 5503–5513 (2010).
  https://doi.org/10.1039/B926434E
- A. L. Ankudinov, B. Ravel, J. J. Rehr and S. D. Conradson, Phys. Rev. B **58**, 7565–7576 (1998).
  https://doi.org/10.1103/PhysRevB.58.7565
- J. J. Rehr and R. C. Albers, Rev. Mod. Phys. **72**, 621–654 (2000). https://doi.org/10.1103/RevModPhys.72.621
- The open-source FEFF 8.5L release that this package follows: `feff85exafs`, https://github.com/xraypy/feff85exafs
  (copyright University of Washington), distributed with Larch: M. Newville, J. Phys.: Conf. Ser. **430**, 012007
  (2013). https://doi.org/10.1088/1742-6596/430/1/012007

When you use RMC-EXAFS, EvAX:

- J. Timoshenko, A. Kuzmin and J. Purans, Comput. Phys. Commun. **183**, 1237–1245 (2012).
  https://doi.org/10.1016/j.cpc.2012.02.002
- J. Timoshenko, A. Kuzmin and J. Purans, J. Phys.: Condens. Matter **26**, 055401 (2014).
  https://doi.org/10.1088/0953-8984/26/5/055401
- For the wavelet residual: J. Timoshenko and A. Kuzmin, Comput. Phys. Commun. **180**, 920–925 (2009).
  https://doi.org/10.1016/j.cpc.2008.12.020

A citation entry for this package is in [CITATION.cff](CITATION.cff). The full list with titles is in the
documentation page "How to cite".

## Install

Requirements: Linux, Python 3.11 or newer, an NVIDIA GPU with a CUDA build of PyTorch.

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126     # the CUDA build that matches your driver
pip install feffgpu                                                       # or: pip install git+https://github.com/nmarcella/feffGPU.git
feffgpu doctor                                                            # check the installation
```

FEFF's free-atom table, which the potentials need, ships inside the package (81 MB), so nothing else has to be
downloaded.

Optional extras: `feffgpu[mace]` (run new dynamics with MACE potentials), `feffgpu[mda]` (MDAnalysis trajectory
reader), `feffgpu[larch]` (xraylarch), `feffgpu[mcp]` (the server for agents), `feffgpu[docs]`, `feffgpu[test]`.

## Use

**FEFF on the GPU**

```bash
feffgpu feff.inp --out chi.dat            # chi(k) with S0² = 1 and no Debye-Waller factor, like FEFF's chi.dat
feffgpu phase feff.inp phase.pkl          # potentials and phase shifts only: the phase file for the other two tools
```

```python
from feffgpu import run
out = run("feff.inp", device="cuda")      # out["k"], out["chi"], out["phase"], out["kept"], out["counts"], out["timing"]
```

**MD-EXAFS**: one TOML file names the trajectory, the absorbing element and the phase file.

```bash
md-exafs validate config.toml
md-exafs run config.toml                  # the averaged spectrum, its parts by path type, an HDF5 cache
md-exafs rdf config.toml                  # radial distributions and shell statistics
md-exafs compare mdexafs_out              # against a measured spectrum: R-factor, S0² and the E0 shift with intervals
```

**RMC-EXAFS**: one TOML file names the start structure, the measured spectra and the phase files.

```bash
rmc-exafs validate config.toml
rmc-exafs run config.toml                 # the search; fitted states, spectra, history, report
rmc-exafs shells runs/my_fit --shell Zn-C:1.8:2.6      # N, R, MSRD by integration, with the spread over states
```

`md-exafs example` and `rmc-exafs example` list commented example configurations; `md-exafs schema` and
`rmc-exafs schema` print every key. Examples that generate their own synthetic inputs are in
[`examples/`](examples/). The quick starts are in the documentation.

## What to expect

- **Conventions** are xraylarch's (and so Athena's and Artemis's): the Hanning window by `kmin`, `kmax`, `dk`; the
  R-factor on the R window; S0² and the E0 shift on the model, one value per edge, shared between samples. FEFF's
  chi(k) has no Debye-Waller factor: disorder comes from the atom positions.
- **Validation** is against FEFF 8.5L itself (random environments with absorbers from C to Au, K, L and M edges,
  many structures and cards: median R-factor 1.3e-6 between the two codes, worst 2.4e-3) and against synthetic
  targets with known answers. Literature values are not used as a yardstick.
- **Limits**: EXAFS only (no XANES, polarization or spin); paths up to four legs; NVIDIA GPUs; one GPU per run
  today. MD-EXAFS can be split over several GPUs and nodes with shards; a pool over several GPUs is planned and its
  access points are in place (`feffgpu.parallel`).
- **Agents**: an MCP server (`feffgpu-mcp`) lets an agent validate configurations, start runs as jobs, follow them
  and read results; `AGENTS.md` is the guide for agents that work on the code.

## Repository layout

```
src/feffgpu/        the package: the FEFF pipeline, exafs/ io/ analysis/ parallel/ (shared), md/, rmc/, mcp/
tests/              validation against FEFF's own outputs, the shared layer, MD-EXAFS, RMC-EXAFS, the agent interface
docs/               the documentation (Sphinx, Read the Docs)
examples/           runnable examples with synthetic inputs
tools/              the free-atom table builder and the random-environment validation (need a FEFF 8.5L build)
fortran/            the patch for feff85exafs and how to build the optional FEFF library
data/validation/    four random environments with FEFF's own outputs (regression cases)
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md); agents see [AGENTS.md](AGENTS.md).

## License

MIT (see [LICENSE](LICENSE)). `feffgpu` is based on FEFF 8.5L, copyright University of Washington; the notices that
its license requires, and the acknowledgments of EvAX and of the original MD-EXAFS, are in [NOTICE](NOTICE). FEFF's
own source code is not included.
