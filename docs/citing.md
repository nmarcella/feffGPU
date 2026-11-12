# How to cite

`feffgpu` exists because FEFF exists; its reverse Monte Carlo search follows the rules of EvAX; its MD-EXAFS
continues the program of that name. If you publish work that used this package, please cite the work it is built
on, as you would if you had used those programs themselves.

## Always: FEFF

- J. J. Rehr, J. J. Kas, F. D. Vila, M. P. Prange and K. Jorissen, *Parameter-free calculations of X-ray spectra
  with FEFF9*, Phys. Chem. Chem. Phys. **12**, 5503–5513 (2010). <https://doi.org/10.1039/B926434E>
- A. L. Ankudinov, B. Ravel, J. J. Rehr and S. D. Conradson, *Real-space multiple-scattering calculation and
  interpretation of x-ray-absorption near-edge structure*, Phys. Rev. B **58**, 7565–7576 (1998).
  <https://doi.org/10.1103/PhysRevB.58.7565>
- J. J. Rehr and R. C. Albers, *Theoretical approaches to x-ray absorption fine structure*, Rev. Mod. Phys. **72**,
  621–654 (2000). <https://doi.org/10.1103/RevModPhys.72.621>
- The open-source FEFF 8.5L release that this package follows, `feff85exafs`
  (<https://github.com/xraypy/feff85exafs>, copyright University of Washington), distributed with Larch:
  M. Newville, *Larch: an analysis package for XAFS and related spectroscopies*, J. Phys.: Conf. Ser. **430**,
  012007 (2013). <https://doi.org/10.1088/1742-6596/430/1/012007>

The FEFF project's own list of references is at <https://feff.phys.washington.edu>.

## When you use RMC-EXAFS: EvAX

- J. Timoshenko, A. Kuzmin and J. Purans, *Reverse Monte Carlo modeling of thermal disorder in crystalline
  materials from EXAFS spectra*, Comput. Phys. Commun. **183**, 1237–1245 (2012).
  <https://doi.org/10.1016/j.cpc.2012.02.002>
- J. Timoshenko, A. Kuzmin and J. Purans, *EXAFS study of hydrogen intercalation into ReO3 using the evolutionary
  algorithm*, J. Phys.: Condens. Matter **26**, 055401 (2014). <https://doi.org/10.1088/0953-8984/26/5/055401>
- If you use the wavelet residual: J. Timoshenko and A. Kuzmin, *Wavelet data analysis of EXAFS spectra*, Comput.
  Phys. Commun. **180**, 920–925 (2009). <https://doi.org/10.1016/j.cpc.2008.12.020>
- A review of the method: A. Kuzmin, J. Timoshenko, A. Kalinko, I. Jonane and A. Anspoks, *Treatment of disorder
  effects in X-ray absorption spectra beyond the conventional approach*, Radiat. Phys. Chem. **175**, 108112
  (2020). <https://doi.org/10.1016/j.radphyschem.2019.02.030>

EvAX itself is the program of J. Timoshenko and A. Kuzmin. This package does not contain or require it.

## This package

A citation entry is in the file `CITATION.cff` of the repository (GitHub shows it under "Cite this repository").
Please give the version you used.

## The conventions

The Fourier transform, the window and the R-factor follow xraylarch: M. Newville, J. Phys.: Conf. Ser. **430**,
012007 (2013).
