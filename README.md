# TPMS Builder

Fills imported or generated geometry with Triply Periodic Minimal Surface lattices and
exports a watertight, printable mesh.

Desktop application (PySide6 + PyVista) with a headless CLI. Every feature is a
standalone module — see [docs/architecture.md](docs/architecture.md).

---

## Install

Python 3.10 or newer. Verified on Python 3.14.3, Windows.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

STEP and IGES import needs OpenCascade, kept separate because it pulls ~400 MB. The app
runs without it and greys the formats out.

```bash
.venv\Scripts\python -m pip install -r requirements-step.txt
```

To run the tests as well:

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

## Run

```bash
python run.py
```

Optionally with a file: `python run.py part.stl` or `python run.py saved-setup.json`.

---

## What it does

**Patterns** — Gyroid, Schwarz P, Schwarz D, Neovius. Each is a 2π-periodic field with
an analytic gradient.

**Solid modes** — *Sheet* thickens the surface into a wall, leaving two open void
networks and the best stiffness for the mass. *Solid network* fills one labyrinth,
giving a single connected void that de-powders easily. *Inverse* fills the other.

**Grading** — cell size and wall thickness can vary across the part:

| Mode | Driven by |
| --- | --- |
| Uniform | nothing — one size everywhere |
| Axial | position along X, Y or Z |
| Radial | distance from a point or a line |
| Distance to surface | depth below the skin — dense shell, open core |
| Expression | your own `f(x, y, z)` |

**Base shapes** — box, sphere, cylinder, cone/frustum, torus, capsule. Analytic, so the
boundary is exact at any resolution and nothing needs importing to get started.

**Import** — STL, OBJ, PLY, 3MF, OFF, glTF; STEP and IGES with the optional extra.
**Export** — STL, OBJ, PLY, 3MF, OFF, GLB.

---

## Command line

Same engine, no GUI:

```bash
python -m tpms.cli --shape cylinder --size 50 --cell 7 --wall 1.1 -o lattice.stl
```

```bash
python -m tpms.cli -i bracket.step -p schwarz_d --grading surface_distance -r 256 -o out.stl
```

```bash
python -m tpms.cli --grading expression --cell-expression "5+4*w" --wall-expression "1.2" -o graded.stl
```

`python -m tpms.cli --list` prints every pattern, grading law, shape and format.
`--help` covers the rest.

---

## Choosing settings

**Wall thickness relative to cell size** is the number that matters. Below about 4 % the
wall is too thin for most printers; above about 35 % neighbouring walls merge and the
lattice becomes a solid with holes in it. The panel warns at both ends.

**Resolution** must resolve the wall. A wall thinner than ~2 voxels comes out pitted or
vanishes — the most common reason a run returns nothing. The quality panel reports the
voxel size and says so before you run.

**Resolution costs triangles, not field memory.** Measured, 60 mm box, 8 mm cells:

| Grid | Triangles | Peak RAM | Time |
| --- | --- | --- | --- |
| 128³ | 1.7 M | 0.34 GB | 1.8 s |
| 256³ | 6.9 M | 1.15 GB | 6.9 s |
| 384³ | 15.7 M | 2.5 GB | 18 s |
| 512³ | 27.7 M | 4.4 GB | 38 s |

192³–256³ is the sweet spot for most parts. The panel predicts triangles and memory
within about 2 % before you commit, and generation is cancellable throughout.

---

## Accuracy

Wall thickness is a real millimetre dimension, not a field threshold. A TPMS field is
not a distance function, so it is converted to one and then refined by Newton iteration.
Measured against true wall thickness, two refinement passes hold every pattern within
0.8 % (Neovius 2.3 %), against 6–15 % error for the single-step estimate.

Output is verified watertight: across sheet and network modes, with and without a solid
skin, 100 % of edges are shared by exactly two faces, with no boundary or non-manifold
edges.

Imported geometry is voxelised to a signed distance field whose boundary lands within
about 0.02 voxels on average, with sharp CAD edges reproduced cleanly. A mesh that is
not watertight is reported rather than silently filled wrong.

---

## Tests

```bash
.venv\Scripts\python -m pytest tests -q
```

59 tests, about 45 seconds. Covers every pattern, mode, grading law, shape and file
format, and enforces the layering rule — `tests/test_standalone.py` fails if any feature
starts depending on Qt.

---

## Project layout

```
tpms/
  core/       grid, mesh, SDF ops, marching cubes, expression sandbox
  io/         import and export, one module per format family
  features/   tpms · grading · shapes · voxelize · generate
  ui/         Qt window, viewport, worker, one panel per feature
  cli.py      headless entry point
docs/
  architecture.md   layering, pipeline, contracts, measured performance
  tasks.md          build order
```

Dependencies run one way only: `ui → features → core`, `io → core`. Nothing in `core`
imports from the project, which is what lets any feature be driven from a plain script.

Adding a pattern, grading law or shape means writing one file and registering it. See
§6 of the architecture document.

---

## Not included

Stress analysis is deliberately not built. The namespace `features/analysis/` is
reserved for it, and the hooks it needs are already in place — serialisable parameters,
and expression-driven grading so a solved stress field can feed straight back in as a
thickness expression.
