# FEFF 8.5L against feffgpu on random environments

35 environments were drawn; 34 were compared, and in one FEFF itself failed.

R-factor in the analysis window (R-space, k 3 to min(23, kmax-1), R 1 to the path cutoff): median 1.3e-06, worst 2.4e-03; 31 of 34 below 1e-5, 33 below 1e-4.
R-factor over the whole k range (k-space, k^2 weight, k >= 2): median 5.5e-06, worst 2.1e-04.

| case | environment | edge | atoms | modifiers | cards | paths (total degeneracy) FEFF / GPU | R (R-space, k 3-23) | R (k-space, k>=2, all k) | max diff / max | Fermi level diff (Ha) | partial-wave limits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A000 | Zn-C rock salt | K of C | 294 | disorder_sd 0.0, vacancy_fraction 0.032 | SCF 3.4, RPATH 7.48, k 25, pw 2.0, EXCHANGE 0 0.0 0.62 | 10537 / 10537 | 1.1e-06 | 2.4e-06 | 4.2e-03 | 1.8e-08 | same |
| A001 | Sb diamond | K of Sb | 99 | disorder_sd 0.031 | SCF 4.9, RPATH 7.13, k 25, pw 1.0, EXCHANGE 0 -0.71 1.5 | 94 / 94 | 4.4e-06 | 8.4e-06 | 2.9e-03 | -3.7e-08 | same |
| A002 | Ca-Ir-N random packing | K of Ca | 398 | disorder_sd 0.0115, vacancy_fraction 0.097 | no SCF, RPATH 6.6, k 30, pw 0.5, ION 3 1.0 | 4940 / 4940 | 2.4e-03 | 1.1e-04 | 4.5e-01 | -2.2e-06 | same |
| A003 | Sn-Cs rock salt | L3 of Sn | 50 | strain [-0.0353, -0.0133, 0.0599], disorder_sd 0.1052, vacancy_fraction 0.027 | SCF 5.1, RPATH 5.1, k 30, pw 1.0 | 6 / 6 | 4.7e-08 | 2.1e-04 | 8.5e-04 | 1.6e-08 | same |
| A004 | Mn simple cubic | K of Mn | 171 | disorder_sd 0.0 | SCF 3.9, RPATH 5.32, k 15, pw 2.0, EXCHANGE 0 -1.4 1.72 | 98 / 98 | 5.1e-07 | 8.0e-07 | 2.4e-03 | -1.0e-07 | same |
| A005 | Se-Ru fluorite | L3 of Se | 252 | strain [-0.0523, -0.0249, 0.0082], disorder_sd 0.0512 | SCF 5.0, RPATH 5.44, k 20, pw 0.5, EXCHANGE 0 0.95 0.0 | 212 / 212 | 1.4e-06 | 2.7e-06 | 7.6e-03 | -5.2e-08 | same |
| A006 | C-Se random packing | K of C | 500 | disorder_sd 0.0871 | SCF 3.1, RPATH 6.57, k 15, pw 1.0 | 3820 / 3820 | 2.4e-06 | 6.2e-06 | 3.8e-03 | 6.7e-08 | same |
| A007 | Sr diamond | K of Sr | 39 | disorder_sd 0.0507, vacancy_fraction 0.016 | no SCF, RPATH 6.07, k 15, pw 1.0, EXCHANGE 0 0.0 1.22 | 4 / 4 | 7.3e-12 | 1.4e-11 | 2.2e-06 | -1.1e-09 | same |
| A008 | Cl diamond | K of Cl | 159 | disorder_sd 0.0473 | SCF 5.0, RPATH 6.7, k 30, pw 0.5 | 831 / 832 | 6.2e-06 | 2.7e-05 | 1.4e-02 | 2.3e-08 | same |
| A009 | Ta-C zincblende | M5 of Ta | 214 | strain [-0.0602, -0.0164, 0.0199], disorder_sd 0.1144 | SCF 4.0, RPATH 6.68, k 30, pw 4.0, EXCHANGE 2 -1.75 1.48 | 241 / 241 | 2.3e-06 | 6.3e-06 | 8.9e-03 | -1.5e-07 | same |
| A010 | Pd-N random packing | L2 of Pd | 499 | disorder_sd 0.0192, vacancy_fraction 0.042 | SCF 3.0, RPATH 5.75, k 25, pw 2.0, EXCHANGE 0 1.82 1.28 | 1459 / 1458 | 1.0e-06 | 8.8e-06 | 3.0e-03 | -1.7e-08 | same |
| A011 | Ta-Al-N random packing | L3 of Ta | 500 | disorder_sd 0.0799 | SCF 4.1, RPATH 7.34, k 15, pw 4.0 | 681 / 681 | 1.6e-06 | 4.9e-06 | 3.9e-03 | 9.4e-08 | same |
| A012 | Ir-As CsCl | L3 of Ir | 170 | disorder_sd 0.0366, vacancy_fraction 0.026 | SCF 4.8, RPATH 5.66, k 20, pw 1.0 | 276 / 276 | 1.8e-06 | 2.0e-06 | 4.6e-03 | 6.8e-09 | same |
| A013 | Re simple cubic | L3 of Re | 58 | disorder_sd 0.148, vacancy_fraction 0.031 | SCF 4.1, RPATH 6.34, k 20, pw 2.0 | 32 / 32 | 1.2e-07 | 9.6e-08 | 1.2e-03 | -5.6e-08 | same |
| A014 | Zn simple cubic | L3 of Zn | 131 | strain [-0.0548, -0.0469, 0.0161], disorder_sd 0.0203 | SCF 4.7, RPATH 5.57, k 15, pw 4.0, EXCHANGE 2 -1.36 1.15 | 210 / 210 | 1.2e-06 | 1.4e-06 | 1.5e-03 | -1.0e-07 | same |
| A015 | Sr simple cubic | L2 of Sr | 31 | strain [-0.043, 0.0119, 0.0374], disorder_sd 0.0422 | EXCHANGE 0 0.33 0.79, ION 1 0.5 | - | FEFF itself failed in its potentials stage (a crash in `coulom.f`); no comparison | | | | |
| A016 | Ge-La rock salt | L3 of Ge | 57 | disorder_sd 0.056 | SCF 5.1, RPATH 6.25, k 15, pw 4.0 | 68 / 68 | 2.5e-07 | 9.0e-07 | 2.0e-02 | 3.7e-08 | same |
| A017 | Mn random packing | K of Mn | 169 | disorder_sd 0.0104 | SCF 3.6, RPATH 7.52, k 20, pw 0.5, EXCHANGE 2 0.0 1.43 | 553 / 553 | 3.0e-06 | 8.7e-06 | 3.0e-03 | 1.2e-07 | same |
| A018 | Ge-V rock salt | K of V | 238 | strain [0.0159, 0.035, 0.0453], disorder_sd 0.0126, vacancy_fraction 0.048 | SCF 4.2, RPATH 5.84, k 25, pw 4.0, ION 0 0.5 | 86 / 86 | 1.5e-07 | 4.9e-07 | 6.8e-03 | 7.0e-09 | same |
| A019 | Zn-Te-O perovskite | K of O | 106 | disorder_sd 0.14, surface_height 1.47 | SCF 3.6, RPATH 6.91, k 25, pw 0.5 | 843 / 843 | 1.1e-05 | 5.5e-05 | 4.9e-02 | 1.0e-07 | same |
| A020 | Rb bcc | K of Rb | 21 | strain [-0.0488, -0.0403, 0.049], disorder_sd 0.0, vacancy_fraction 0.048 | SCF 5.5, RPATH 5.43, k 20, pw 1.0 | 11 / 11 | 2.9e-12 | 5.4e-12 | 1.0e-06 | 5.0e-10 | same |
| A021 | Mn-Hg rock salt | K of Mn | 82 | strain [-0.065, -0.0403, 0.0304], disorder_sd 0.0563 | SCF 4.2, RPATH 5.54, k 25, pw 0.5, EXCHANGE 2 0.0 1.21 | 94 / 94 | 5.6e-06 | 1.1e-05 | 4.5e-03 | -1.3e-07 | same |
| A022 | Au-F random packing | K of Au | 500 | disorder_sd 0.0426 | SCF 4.9, RPATH 7.55, k 25, pw 2.0, EXCHANGE 2 -1.88 0.1 | 140 / 140 | 3.2e-07 | 9.9e-07 | 1.9e-03 | -2.5e-07 | same |
| A023 | Y-P fluorite | K of Y | 97 | disorder_sd 0.0427 | SCF 5.0, RPATH 5.86, k 20, pw 0.5 | 69 / 69 | 1.9e-06 | 3.3e-06 | 9.8e-03 | 7.6e-09 | same |
| A024 | As-N random packing | K of As | 357 | disorder_sd 0.0878 | SCF 4.7, RPATH 6.31, k 25, pw 2.0 | 3141 / 3141 | 3.4e-06 | 1.3e-05 | 1.5e-03 | -9.0e-09 | same |
| A025 | W bcc | L3 of W | 137 | disorder_sd 0.0112 | SCF 5.0, RPATH 5.31, k 25, pw 0.5 | 62 / 62 | 4.2e-09 | 1.5e-08 | 5.9e-03 | -1.0e-08 | same |
| A026 | Te-N random packing | K of Te | 346 | disorder_sd 0.1382 | SCF 4.8, RPATH 5.85, k 20, pw 0.5, EXCHANGE 0 0.0 1.93 | 1165 / 1165 | 4.6e-06 | 6.9e-06 | 6.2e-03 | -3.5e-08 | same |
| A027 | Mg-N zincblende | K of Mg | 82 | strain [-0.0394, -0.0289, 0.0411], disorder_sd 0.0353, vacancy_fraction 0.059 | SCF 4.5, RPATH 6.29, k 25, pw 4.0 | 126 / 126 | 5.3e-07 | 1.8e-06 | 4.1e-03 | 1.2e-08 | same |
| A028 | Cs diamond | K of Cs | 17 | disorder_sd 0.0365, vacancy_fraction 0.03 | SCF 5.6, RPATH 5.03, k 15, pw 2.0 | 4 / 4 | 2.0e-12 | 3.1e-12 | 6.1e-06 | 6.3e-09 | same |
| B000 | W bcc W, 300 K MD frame | K of W | 314 | - | SCF 4.0, RPATH 9.4, k 38, pw 0.5 | 675 / 675 | 3.0e-06 | 1.1e-05 | 2.0e-03 | 1.7e-08 | same |
| B001 | W bcc W, 300 K MD frame | K of W | 317 | - | SCF 4.0, RPATH 9.4, k 38, pw 0.5 | 754 / 753 | 1.4e-06 | 2.1e-05 | 4.9e-03 | -3.6e-08 | same |
| B002 | W bcc W, 300 K MD frame | K of W | 317 | - | SCF 4.0, RPATH 9.4, k 38, pw 0.5 | 655 / 656 | 7.0e-07 | 7.2e-06 | 2.4e-03 | 6.1e-08 | same |
| B003 | W bcc W, 300 K MD frame | K of W | 317 | - | SCF 4.0, RPATH 9.4, k 38, pw 0.5 | 713 / 713 | 1.4e-05 | 3.4e-05 | 4.6e-03 | 3.7e-08 | same |
| B004 | W bcc W, 300 K MD frame | K of W | 317 | - | SCF 4.0, RPATH 9.4, k 38, pw 0.5 | 742 / 742 | 1.2e-06 | 1.6e-05 | 6.0e-03 | -5.4e-08 | same |
| B005 | W bcc W, 300 K MD frame | K of W | 317 | - | SCF 4.0, RPATH 9.4, k 38, pw 0.5 | 696 / 696 | 7.4e-07 | 3.9e-06 | 2.4e-03 | 6.3e-08 | same |

```{figure} random_environments.png
:alt: R-factor between FEFF 8.5L and feffgpu for each random environment

The R-factor between FEFF 8.5L and feffgpu for each environment.
```
