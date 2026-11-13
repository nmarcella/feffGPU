# Path types

The spectrum of every frame is stored together with its decomposition by path type, so that the contribution of a
kind of path (and of a range of path lengths) can be averaged and compared later without a new calculation.

## Names

A scattering path starts at the absorbing atom, visits one to three scattering atoms and returns. Its type is the
sequence of elements along the path, absorber first, joined by dashes:

| Type | Path | Legs |
|---|---|---|
| `Zn-C` | Zn absorber → C neighbor → back | 2 (single scattering) |
| `Zn-Zn` | Zn absorber → Zn neighbor → back | 2 |
| `Zn-C-C` | Zn → C → C → back | 3 (double scattering) |
| `Zn-C-Zn` | Zn → C → Zn → back, or Zn → Zn → C → back | 3 |
| `Zn-C-Zn-C` | Zn → C → Zn → C → back | 4 (triple scattering) |

A path and its time reverse (the same atoms visited in the opposite order) give the same signal. FEFF's path finder
keeps one of the two, with degeneracy 2, and which one it keeps can differ from absorber to absorber. Both spellings
therefore name one type: the one whose scatterer sequence comes first alphabetically (`Zn-C-Zn`, not `Zn-Zn-C`). In
`[average]` either spelling may be used. In a four-leg path the middle atom may be the absorbing atom itself (the
photoelectron returns to it and leaves again); it is named by its element.

The list of types of an edge is fixed before the first frame: every combination of the system's elements up to
`nleg` legs. Every frame and every shard store the same types in the same order; a type that does not occur is zero.
`md-exafs run` prints the list, and `chi.h5` stores it ([](outputs.md)).

## How the parts are computed

**Single scattering.** All pairs within `rpath` count, without an importance filter. For each scatterer potential
(each neighbor element) the absorber-neighbor distances of the frame are histogrammed on the 0.002 Å grid of the
pair table, with linear-interpolation weights, and multiplied by the exact two-leg signal of the path stage tabulated
on that grid (`feffgpu.exafs.PairTables`). The histogram is stored, so the part of any distance range can be
recomputed later. The pair table agrees with the path stage of each pair to its interpolation error (checked in the
tests to better than 1e-3 of the signal's maximum).

**Multiple scattering.** For every absorber, FEFF's path finder (`feffgpu.exafs.MultipleScattering`) lists the
paths of three to `nleg` legs up to `rpath` that pass FEFF's plane-wave importance filter `criteria` (in percent of the
shortest single-scattering path's importance; 0 keeps every path). The path stage evaluates them, and each path is
added to the part of its type and of its effective length, in bins of `ms_bin_width` (0.5 Å by default). The
effective length is half the total path length (FEFF's reff).

The parts add up to the total: the sum over types equals the spectrum of the frame, the bins of a type add up to the
type, and single plus multiple scattering is the total (all three checked in the tests).

## Selecting parts

```toml
[average]
path_types = ["Zn-C", "Zn-Zn", "Zn-C-C"]
max_distance = [3.0, 4.5, 5.0]       # Å: one limit per type, or one number for all
```

`md-exafs average` (and `compare`, `plot`) then use only these parts:

- a single-scattering type with a limit keeps the pairs up to that distance, to the 0.002 Å step of the table;
- a multiple-scattering type with a limit keeps the bins of effective length below it, so the limit must be a
  multiple of the edge's `ms_bin_width`; any other value is refused with a message. Choose `ms_bin_width` before the
  run if you need finer limits (smaller bins make a larger cache);
- a type without a limit, or with a limit of at least `rpath`, is taken whole;
- in a run with several edges, each edge takes the types that start with its absorbing element, so one list can name
  the types of every edge; an edge with none of its types in the list keeps its total spectrum.

## Paths of five and more legs

`nleg` goes from 2 to 8; the default is 4. Paths of five and more legs exist only where `rpath` is at least 2.5 bond
lengths, and they are strongest along chains of atoms ([](../concepts/paths.md)). They are few and their element
sequences are many, so they are not split by element: each number of legs is one path type, named
`"Zn-(5 legs)"`, `"Zn-(6 legs)"` and so on. In the cache, `n_paths` has one column per number of legs from three to
`nleg`.
