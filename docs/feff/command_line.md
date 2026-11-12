# Command line

One command, `feffgpu`, with subcommands. `feffgpu <subcommand> --help` lists the options of each.

| Command | Does |
|---|---|
| `feffgpu feff.inp --out chi.dat` | every stage: a FEFF input to chi(k). The same as `feffgpu run feff.inp`. |
| `feffgpu phase feff.inp phase.pkl` | potentials and phase shifts only: the phase file for MD-EXAFS and RMC-EXAFS |
| `feffgpu fetch-atoms [directory]` | download and check the free-atom table |
| `feffgpu doctor` | check the installation |
| `feffgpu md ...` | MD-EXAFS, the same as the command `md-exafs` |
| `feffgpu rmc ...` | RMC-EXAFS, the same as the command `rmc-exafs` |
| `feffgpu mcp ...` | the MCP server for agents, the same as `feffgpu-mcp` |
| `feffgpu --version` | the version |

`feffgpu-phase` and `feffgpu-fetch-atoms` exist as separate commands with the same options as the subcommands.

## `feffgpu run`

```text
feffgpu [run] feff.inp [--out chi_gpu.dat] [--device cuda] [--lmax N] [--json] [--verbose]
```

Reads the FEFF input, computes the potentials, the phase shifts, the paths and their sum, and writes chi(k) on
FEFF's grid (0.05 to 30 1/Å) with S0² = 1 and no Debye-Waller factor, like FEFF's `chi.dat`.

- `--device`: a torch device, `cuda` (default), `cuda:1`, or `cpu` for a small input.
- `--lmax`: a fixed partial-wave limit in place of FEFF's rule.
- `--json`: print one JSON object (output file, paths per number of legs, Fermi level, seconds per stage) in place
  of the text line.

## `feffgpu phase`

```text
feffgpu phase feff.inp out.pkl [--xkmax 24.5] [--lmax N] [--device cuda] [--json]
```

Computes the potentials and phase shifts and writes them as a pickled `feffgpu.PhaseData`. `--xkmax` is the largest
k of the phase grid in 1/Å; choose it above the largest k that you will ask of a spectrum.

## `feffgpu fetch-atoms`

```text
feffgpu fetch-atoms [directory] [--from-file archive.tar.gz] [--url URL] [--no-checksum]
```

See [](../installation.md).

## `feffgpu doctor`

```text
feffgpu doctor [--json]
```

Exit code 0 if the installation is complete, 1 if something required is missing.

## Errors

A user error ends with one message that starts with `feffgpu:` and says what to do, without a traceback: a missing
input file, a CPU-only PyTorch, no usable GPU, a missing free-atom table, a FEFF card that is not supported.
