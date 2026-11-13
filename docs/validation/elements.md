# Every element, edge by edge

The random environments of [](random_environments.md) vary the structure and the input cards. This page varies the
element and the edge, one at a time, across the periodic table: for every element from hydrogen (Z = 1) to
einsteinium (Z = 99), and for each of its K, L3 and M5 edges that exists, the same `feff.inp` went through FEFF 8.5L
and through `feffgpu`, and the two chi(k) were compared. An edge exists for an element when the core level is occupied
in FEFF's free atom: every element has a K edge, the L3 edge starts at nitrogen, the M5 edge at manganese.

```{figure} agreement_all_elements.png
:alt: Periodic table with the R-factor between FEFF 8.5L and feffgpu for the K, L3 and M5 edge of every element, a plot of feffgpu's spectra against FEFF's, and the R-factor against the atomic number

Top: the periodic table; the three squares of an element are its K, L3 and M5 edges, colored by the R-factor between
the two codes (darker is a larger difference; the mark on the scale is the limit of the regression tests). Inset: every
point of every spectrum, `feffgpu` against FEFF 8.5L; they lie on the diagonal. Bottom: the same R-factors against
the atomic number, with the cases above the limit named.
```

## Result

```{include} elements_summary.md
```

The R-factor is the one of [](../concepts/conventions.md): the squared difference of the two Fourier transforms over
the R window, divided by the squared transform of FEFF's spectrum. A value of 2e-5 means that the two spectra differ
by about 0.4 % of the signal; the median here corresponds to about 0.1 %.

What is known about the three cases above the limit:

- **Bromine K (1.0e-4).** The potentials and phase shifts of the two codes agree (Fermi level to 7e-9 hartree, phase
  shifts to 7e-6). The path lists differ: `feffgpu` keeps four more four-leg paths, counted with their degeneracy,
  than FEFF. Their estimated importance sits at the 2 % threshold of the path filter, where the two codes can decide
  differently.
- **Thorium K (4.6e-5) and L3 (2.5e-5).** The Fermi levels agree to 4e-8 hartree. At the K edge the path counts are
  equal and the two spectra agree to 2e-6 of the largest signal below k = 6 1/Å; the difference grows with k and
  reaches 0.9 % of the largest signal between 15 and 20 1/Å. At the L3 edge the largest difference is at the same
  high k (0.9 %), and `feffgpu` keeps two more four-leg paths. The cause of the difference at high k has not been
  traced. The M5 edge of thorium is at 1.3e-5.

## What this sweep found

The first run of this sweep gave Fermi levels that differed from FEFF's by about 3e-5 hartree, where the random
environments had agreed to 2e-6 or better. The cause was one of the two compiled GPU kernels of the radial solver, which a
newer version of the kernel compiler compiled differently ([](../feff/limits.md)). It was corrected, and every number
and figure on this page comes from the corrected code. With the defect the median R-factor of the sweep was 1.9e-6
instead of 1.7e-6 and seven cases instead of three were above the limit, the largest at 9e-4.

## How each case is built

Every case uses the same recipe, so that only the element and the edge change. These are test inputs, not models of
real materials.

- **Structure.** A cluster of the element itself on a face-centered cubic lattice: the nearest-neighbor distance d
  is twice the covalent radius of the element (at least 1.4 Å), and all atoms within 2.55 d of the absorber are kept
  (87 atoms). Every atom except the absorber is displaced by a random vector with a standard deviation of 0.01 d per
  coordinate, so that the cluster has no exact symmetry. The three edges of an element use the same cluster.
- **Cards.** `EDGE` K, L3 or M5; `S02 1.0`; `SCF 1.55 d 0 30 0.2 1` (self-consistent potentials on the first two
  shells); `RPATH 2.05 d` (four shells); `EXAFS 20`; `AFOLP 1.15`; `CRITERIA 0 2.0`; `NLEG 4`.
- **Two elements without self-consistency.** For nitrogen and francium in this structure, FEFF 8.5L itself stops with
  an error at the start of its self-consistency loop (in the step that separates core from valence states). Their
  five cases were run without the `SCF` card, with overlapped potentials in both codes; the table marks them.

## The spectra, element by element

One panel per element: FEFF 8.5L in black and `feffgpu` in blue on top, k² chi(k) from 0 to 20 1/Å, each panel scaled
to its own maximum. The number is the R-factor between the two codes.

```{figure} elements_K.png
:alt: K-edge spectra of every element from FEFF 8.5L and from feffgpu, drawn over each other

K edges, hydrogen to einsteinium.
```

```{figure} elements_L3.png
:alt: L3-edge spectra of every element from nitrogen on, from FEFF 8.5L and from feffgpu, drawn over each other

L3 edges, nitrogen to einsteinium.
```

```{figure} elements_M5.png
:alt: M5-edge spectra of every element from manganese on, from FEFF 8.5L and from feffgpu, drawn over each other

M5 edges, manganese to einsteinium.
```

## Run it again

The sweep needs a FEFF 8.5L build ([](../installation.md)) and takes about an hour and a half on one GPU.

```bash
FEFF_HOME=/path/to/feff85exafs/local_install python tools/validate_elements.py     # all 267 cases
python tools/validate_elements.py --zmin 7 --zmax 7 --no-scf                       # an element where FEFF's own loop stops
python tools/validate_elements.py --collect                                        # writes data/validation/elements.npz and .json
python tools/plot_validation_elements.py                                           # redraws the figures and the table
```

The spectra of both codes for every case are stored in `data/validation/elements.npz`, and the numbers of the table
in `data/validation/elements.json`.

## Table of all cases

```{include} elements_table.md
```
