- **267 cases compared**: 99 elements (Z = 1 to 99); 99 K edges, 93 L3 edges, 75 M5 edges.
- **R-factor between the two codes** (R space, k from 3 to 19 1/Å, R from 1 Å to the path cutoff, k² weight): median 1.7e-6; 246 of 267 below 1e-5; 264 below 2e-5, the limit that the regression tests enforce; the largest is 1.0e-4 (Br K).
- **By edge**: K: median 1.2e-6, largest 1.0e-4; L3: median 1.6e-6, largest 2.5e-5; M5: median 2.0e-6, largest 1.5e-5.
- **R-factor in k space** (k from 2 1/Å, k² weight, no Fourier transform): median 1.8e-6, largest 1.0e-4.
- **Paths**: the number of paths, weighted by degeneracy, is identical in 252 of the 267 cases. In the other 15 the two codes differ by at most 6 in the weighted count (of 114 to 785 paths), never in the single-scattering paths: these are three- and four-leg paths whose estimated importance sits at the threshold of the path filter.
- **Fermi level** of the self-consistent potentials (262 cases): the two codes differ by 5.0e-8 hartree in the median and by 1.4e-5 hartree at most.

The 3 cases above 2e-5:

| element and edge | R-factor | Fermi level difference (hartree) | path counts, feffgpu minus FEFF 8.5L |
|---|---|---|---|
| Br K | 1.0e-4 | 6.8e-9 | +4 with 4 legs |
| Th K | 4.6e-5 | 4.1e-8 | equal |
| Th L3 | 2.5e-5 | 2.3e-8 | +2 with 4 legs |
