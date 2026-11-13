# Commissioning against EvAX (underway)

**Status, 13 November 2026: underway.** RMC-EXAFS is being commissioned against published calculations of EvAX
(J. Timoshenko, A. Kuzmin and co-workers), the established program for reverse Monte Carlo fits of EXAFS. This page
shows what has been compared so far and lists what is still open. The numbers are from single runs on one 12 GB
NVIDIA GPU.

## What is compared

EvAX's authors publish a [library of finished calculations](https://www.dragon.lv/evax/library.html): for each, the
input files and the output files (the fitted spectrum, the final atoms, the list of scattering paths). For each
calculation below:

1. `rmc-exafs from-evax` translates EvAX's parameter file into the inputs of this package ([](evax_differences.md)).
   The fit gets what EvAX had: EvAX's own target spectrum, its cutoff, its choice of scattering paths
   (`criteria_rule = "evax"`), its wavelet misfit, its limit on the displacements, and potentials set up as EvAX sets
   them up.
2. `rmc-exafs refine`, the default engine ([](refinement.md)), fits the structure from EvAX's start lattice.
3. The result is put next to EvAX's output files. EvAX itself is not run.

The misfit in the table is EvAX's own measure: the root of the summed squared difference of the wavelet transforms
of model and target over the summed square of the target's. Three numbers are given for each spectrum: EvAX's own
(from its fitted spectrum), EvAX's final atoms put through the forward model of this package, and this package's fit.

## Results so far

| Calculation | Atoms | EvAX's run | `rmc-exafs refine` | Misfit: EvAX's own | EvAX's atoms, this forward model | This package's fit |
|---|---|---|---|---|---|---|
| Pt foil, Pt L3 edge | 500 | 17 h | 8.5 min (100 steps) | 0.108 | 0.108 | 0.095 |
| Cu2O, Cu K edge | 384 | 44 h | 12 min (300 steps) | 0.193 | 0.199 | 0.185 |
| Ga2O3, Ga K edge | 480 (here 600) | 23 h | 19 min (300 steps) | 0.121 | not comparable | 0.135 |
| Au147 particle, Au L3 edge | 147 | 31 h | 7 min (300 steps) | 0.176 | 0.188 | 0.170 |

- **EvAX's run times** are read from the dates of its output files (from the first file written to the last); its
  parameter files ask for 16 threads. They were not measured here.
- **Ga2O3**: EvAX's supercell (2 x 4 x 3) is narrower than twice the cutoff, which this package's forward model does
  not accept, so the fit here uses 2 x 5 x 3 cells and EvAX's atoms cannot be put through the same model.

![Four calculations of EvAX's library fitted with rmc-exafs refine: for each, k squared chi(k), the magnitude of its Fourier transform and the distribution of neighbor distances; EvAX's target in black, EvAX's fit in orange, this package's fit in blue](commissioning_evax.png)

*Black: EvAX's target (the measured spectrum as EvAX prepared it). Orange: EvAX's fit and its final atoms. Blue:
this package's fit and its final atoms. The gray band is the R range of the fit.*

**The neighbor shells** of the two final structures, counted in the same distance ranges (mean distance R and the
variance of the distance, EvAX's value first):

| Calculation | Shell (range, Å) | N | R (Å): EvAX, here | Variance (Å²): EvAX, here |
|---|---|---|---|---|
| Pt foil | Pt-Pt 2.30-3.30 | 12 | 2.777, 2.777 | 0.0062, 0.0062 |
| | Pt-Pt 3.45-4.35 | 6 | 3.926, 3.926 | 0.0068, 0.0070 |
| | Pt-Pt 4.40-5.18 | 24 | 4.808, 4.808 | 0.0069, 0.0070 |
| | Pt-Pt 5.20-5.90 | 12 | 5.551, 5.551 | 0.0070, 0.0074 |
| Cu2O | Cu-O 1.30-2.50 | 2 | 1.856, 1.857 | 0.0125, 0.0136 |
| | Cu-Cu 2.50-3.60 | 12 | 3.025, 3.027 | 0.0276, 0.0271 |
| | Cu-O 3.00-4.10 | 6 | 3.543, 3.544 | 0.0163, 0.0166 |
| | Cu-Cu 3.80-4.80 | 6 | 4.280, 4.283 | 0.0268, 0.0276 |
| | Cu-O 4.20-5.10 | 6 | 4.654, 4.656 | 0.0150, 0.0178 |
| | Cu-Cu 4.80-5.70 | 24 | 5.236, 5.238 | 0.0221, 0.0245 |
| Ga2O3 | Ga-O 1.40-2.60 | 5 | 1.943, 1.943 | 0.0355, 0.0290 |
| | Ga-Ga 2.60-3.90 | 11 | 3.293, 3.292 | 0.0424, 0.0438 |
| | Ga-O 2.90-3.90 | 10 | 3.487, 3.485 | 0.0310, 0.0304 |
| Au147 | Au-Au 2.30-3.50 | 9.5 | 2.878, 2.881 | 0.0220, 0.0203 |
| | Au-Au 3.60-4.40 | 3.3 | 4.051, 4.048 | 0.0216, 0.0167 |
| | Au-Au 4.40-5.30 | 12 | 4.915, 4.929 | 0.0361, 0.0348 |

The numbers of neighbors are those of the start structure in both programs: every atom stays near its site. The
ranges of Ga2O3 and Au147 hold several distances of the structure itself, so their variances include that spread.

**The pieces underneath were checked separately:**

- *The forward model.* On EvAX's final atoms and EvAX's paths, the FEFF program that EvAX runs and this package agree
  to an R-factor of 1e-7 (a Cu foil), 1e-6 (Au147), 2e-6 (Ga2O3) and 7e-5 (Cu2O). EvAX's own spectrum of the same
  atoms differs from FEFF's by 0.003 to 0.007: EvAX computes clusters of similar paths with an error it allows
  itself. That is why EvAX's atoms score slightly worse through an exact forward model than EvAX reports.
- *The choice of paths.* The groups of paths that `criteria_rule = "evax"` keeps are EvAX's own list for 20 of 20
  groups (Pt foil), 28 of 29 (Cu2O), 56 of 57 (Ga2O3), 18 of 18 (Au147), 27 of 27 (Co edge of CoWO4 particles) and
  29 of 30 (Au edge of a PdAu particle); the groups that differ lie at the threshold.
- *The misfit.* EvAX's own residual is reproduced from its two spectra to three digits in every case.

## What is still open

- **The PdAu particle** (336 atoms, the Pd K and Au L3 edges fitted together): the first run reached 0.148 and 0.203
  in 23 minutes against EvAX's 0.109 and 0.174. It is not converged; the start and the length of the refinement for
  two edges with very different numbers of absorbers (10 Pd, 326 Au) are being worked on.
- **The spread between independent refinements** (other seeds, other starts) has not been measured for these
  calculations. A gradient refinement moves the atoms only as far as the spectra ask for; where the spectra say
  little, shell widths stay nearer to those of the start than in a random search. Distances are much less affected.
  The table above is one run each.
- **Paths of more than four legs** are not computed by this package. Three calculations of the library whose EvAX
  path lists contain them (anatase TiO2, ReO3, the W L3 edge of CoWO4 particles) are therefore not compared yet.
- **The population search with gradient steps** (`rmc-exafs run` with `gradient_steps`) holds the multiple scattering
  fixed during those steps and stalls on these systems ([](refinement.md)); the refinement is the route to use.
- **Cells narrower than twice the cutoff** need a larger supercell here (Ga2O3 above).

## How to repeat a comparison

```bash
# the files of one library entry: in/ (start.dat and the files it names) and out/ (EvAX's output)
rmc-exafs from-evax in/start.dat run --outputs out --target evax
cd run
feffgpu phase feff_Cu_K.inp phase_Cu_K.pkl
rmc-exafs refine rmc.toml
cd ..
python tools/compare_with_evax.py run in/start.dat "Cu-O:1.3:2.5,Cu-Cu:2.5:3.6" out
python tools/plot_evax_comparison.py comparison.png "run:Cu_K:Cu2O, Cu K"
```

The library's files are not part of this repository.

## Sources

The calculations are from the EvAX library, <https://www.dragon.lv/evax/library.html>. The library asks for the
citation of J. Timoshenko, A. Kuzmin and J. Purans, J. Phys.: Condens. Matter **26**, 055401 (2014), together with
the paper of each entry:

- Pt foil: J. Timoshenko et al., J. Phys. Chem. Lett. **8**, 5091 (2017).
- Cu2O and Ga2O3: J. Timoshenko et al., Chem. Sci. **11**, 3727 (2020).
- Au147: J. Timoshenko and A. I. Frenkel, Catal. Today **280**, 274 (2017).
- PdAu particle: J. Timoshenko et al., Nano Lett. **19**, 520 (2019).

See also [](../citing.md).
