"""A finished `rmc-exafs` run next to the output of the EvAX calculation it was translated from.

The run folder is one written by `rmc-exafs from-evax <start.dat> <folder> --outputs <EvAX's output> --target evax` and
then run with `rmc-exafs refine rmc.toml` (or `rmc-exafs run rmc.toml`). This script scores this package's final
structure and EvAX's final atoms with the same forward model and the same misfit (the edge's objective; for EvAX's
`Space w` its wavelet residual, printed as EvAX prints it: the root of the ratio), compares the fitted spectra, and
counts the neighbor shells of both structures in the same distance ranges. It writes `compare.json`,
`compare_<spectrum>.npz` and `compare_atoms.npz` into the run folder (`tools/plot_evax_comparison.py` draws them).
EvAX is not run; only its files are read (`feffgpu.rmc.EvaxCalculation`). Needs a GPU.

usage: python tools/compare_with_evax.py <run folder> <EvAX start.dat> <shells: A-B:lo:hi,...> [ELEMENT=]<EvAX output dir> ...
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from feffgpu.rmc import EvaxCalculation, load_config, build_sample
from feffgpu.analysis.pairs import shell_table
from feffgpu.exafs import SpectrumComparison, FitWindow

folder, start = Path(sys.argv[1]), sys.argv[2]
shells = [(s.split(":")[0].split("-")[0], s.split(":")[0].split("-")[1], float(s.split(":")[1]), float(s.split(":")[2])) for s in sys.argv[3].split(",") if s]
outs = sys.argv[4:]
outputs = {o.split("=")[0]: o.split("=")[1] for o in outs} if "=" in outs[0] else outs[0]
calc = EvaxCalculation(start, outputs=outputs)
cfg = load_config(folder / "rmc.toml")
sample = build_sample(cfg.sample[0], cfg)
model, edges = sample.model, sample.edges
for e in edges:
    if e.ms is not None and e.criteria_rule == "evax":
        e.ms.set_lattice(torch.as_tensor(model.positions, dtype=torch.float32)[e.keep.cpu()], e.ipot[e.keep], e.abs_keep, kweight=e.kweight, tolerance=e.group_tolerance)
out = Path(cfg.run.output) / cfg.sample[0].label
if (out / "refine.npz").exists():
    z = np.load(out / "refine.npz"); ours = z["positions"][0]
    rep = json.loads((out / "refine.json").read_text()); seconds, method = rep["seconds"], f"rmc-exafs refine, {rep['settings']['steps']} steps"
else:
    z = np.load(out / "state.npz"); ours = z["positions"].reshape(-1, model.n, 3)[0]
    rep = json.loads((out / "report.json").read_text()); seconds = rep.get("seconds", float("nan"))
    S = cfg.search
    method = f"rmc-exafs run, {S.states} states, {S.generations} generations, {S.gradient_steps} gradient steps"
sym_e, sites_e, cell_e = calc.sites()
theirs = calc.final()
same_cell = len(sym_e) == model.n
k = edges[0].k
result = dict(folder=str(folder), method=method, seconds=seconds, edges=[], shells=[])
print(f"== {folder.name}: {method}; {seconds / 60:.1f} min")
for i, e in enumerate(edges):
    j = [s.name for s in calc.spectra].index(e.name)
    with torch.no_grad():
        chi = e.spectra(torch.as_tensor(ours, device=e.device, dtype=torch.float32)[None])
        m_ours = float(e.residual(chi)[0]) ** 0.5
        m_theirs = None
        if same_cell:
            chi_t = e.spectra(torch.as_tensor(theirs, device=e.device, dtype=torch.float32)[None])
            m_theirs = float(e.residual(chi_t)[0]) ** 0.5
        kf, cf = calc.fitted(j)
        fitted = np.zeros_like(k); fitted[np.round(kf / 0.05).astype(int)] = cf
        m_own = float(e.residual(torch.as_tensor(fitted, device=e.device, dtype=torch.float32)[None])[0]) ** 0.5
    w = e.fit_window
    cmp = SpectrumComparison("x", k, e.data, FitWindow(kmin=w.kmin, kmax=w.kmax, dk=1.0, kweight=w.kweight, rmin=w.rmin, rmax=w.rmax), device="cpu")
    ours_chi = chi[0].double().cpu().numpy()
    between = SpectrumComparison("x", k, fitted, FitWindow(kmin=w.kmin, kmax=w.kmax, dk=1.0, kweight=w.kweight, rmin=w.rmin, rmax=w.rmax), device="cpu").r_factor(ours_chi)
    row = dict(name=e.name, misfit_ours=m_ours, misfit_evax_own=m_own, misfit_evax_atoms=m_theirs, r_ours=cmp.r_factor(ours_chi), r_evax=cmp.r_factor(fitted), r_between=between)
    result["edges"].append(row)
    np.savez(folder / f"compare_{e.name}.npz", k=k, ours=ours_chi, evax=fitted, target=e.data)
    print(f"   {e.name}: misfit by EvAX's measure: ours {m_ours:.4f}; EvAX's own spectrum {m_own:.4f}; EvAX's atoms through this model "
          + (f"{m_theirs:.4f}" if m_theirs is not None else "n/a (another supercell)") + f"; R-factor to the target: ours {row['r_ours']:.4f}, EvAX's {row['r_evax']:.4f}; "
          f"between the two fits {between:.4f}")
if shells:
    a = shell_table(ours[None], model.symbols, model.cell, shells)
    b = shell_table(theirs[None], sym_e, cell_e, shells)
    print("   shell (range, A)        | N: EvAX  ours | mean R (A): EvAX   ours | variance (A^2): EvAX    ours")
    for x, y in zip(a, b):
        print(f"   {x['center']}-{x['neighbor']} {x['r_min']:.2f}-{x['r_max']:.2f}".ljust(27) + f"| {y['N']:8.2f} {x['N']:5.2f} |           {y['R']:.3f}  {x['R']:.3f} |"
              f"               {y['sigma2']:.4f}  {x['sigma2']:.4f}")
        result["shells"].append(dict(shell=f"{x['center']}-{x['neighbor']}", r_min=x["r_min"], r_max=x["r_max"], evax=dict(N=y["N"], R=y["R"], sigma2=y["sigma2"]),
                                     ours=dict(N=x["N"], R=x["R"], sigma2=x["sigma2"])))
d = ours - model.positions
result["rms_displacement_ours"] = float(np.sqrt((d ** 2).sum(1).mean()))
(folder / "compare.json").write_text(json.dumps(result, indent=1))
np.savez(folder / "compare_atoms.npz", ours=ours, evax=theirs, symbols=model.symbols, symbols_evax=sym_e, cell=model.cell, cell_evax=cell_e)
