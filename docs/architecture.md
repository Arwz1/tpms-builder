# TPMS Builder — Architecture

A desktop application that fills imported or generated geometry with Triply Periodic
Minimal Surface (TPMS) lattices and exports the result as a printable mesh.

Reference for scope and interaction model: Hyperganic hyDesign.

---

## 1. Guiding rules

Taken from `claude.md` and applied throughout:

- **Every feature is a directory.** A feature owns its logic, its parameters and its UI
  panel. No feature imports another feature's internals.
- **Every feature runs standalone.** Each package under `tpms/features/` is importable
  and usable with nothing but `numpy` — no Qt, no viewport, no application object. The
  CLI proves this.
- **Modular imports only.** Features talk to each other through the small, stable
  contracts in `tpms/core/`, never by reaching across into each other's modules.
- **No placeholders.** Every file is complete and runnable.

---

## 2. Stack

| Layer | Choice | Why |
| --- | --- | --- |
| UI | PySide6 (Qt 6) | Native desktop, no browser memory ceiling, real threads |
| Viewport | PyVista + pyvistaqt | VTK-backed 3D view that embeds directly in a Qt layout |
| Numerics | NumPy + SciPy | Vectorised field evaluation, EDT, KD-trees |
| Isosurface | scikit-image `marching_cubes` | Well-tested Lorensen implementation |
| Mesh I/O | trimesh | STL / OBJ / PLY / 3MF read and write |
| CAD I/O | cadquery-ocp (OpenCascade) | STEP / IGES, lazily imported |

The desktop choice is what makes the 512³ ceiling safe: a native process can hold a
several-hundred-megabyte working set without the tab-level limits a browser imposes.

---

## 3. Directory map

```
tpms/
├── app.py                  Qt bootstrap, high-DPI setup, window launch
├── cli.py                  Headless generation — proves features stand alone
│
├── core/                   Shared contracts. Depends on nothing but numpy/scipy.
│   ├── grid.py             Grid: bounds, spacing, slab iteration, coordinate arrays
│   ├── sdf.py              CSG ops (union/intersect/subtract/shell/offset)
│   ├── marching.py         Slab-streamed marching cubes
│   ├── mesh.py             Mesh container: welding, stats, transforms
│   └── expressions.py      Sandboxed x/y/z math expression evaluator
│
├── io/                     Format handling. One module per family.
│   ├── registry.py         Extension -> loader dispatch
│   ├── mesh_loader.py      STL / OBJ / PLY / 3MF via trimesh
│   ├── step_loader.py      STEP / IGES via OCP, imported only when called
│   └── exporters.py        Mesh writing, same format families
│
├── features/               The standalone features.
│   ├── tpms/               Periodic scalar fields
│   │   ├── base.py         TPMSPattern ABC + name registry
│   │   ├── gyroid.py       Schoen Gyroid
│   │   ├── schwarz_p.py    Schwarz Primitive
│   │   ├── schwarz_d.py    Schwarz Diamond
│   │   ├── neovius.py      Neovius
│   │   └── modes.py        Field -> solid: sheet / solid-network / inverse
│   │
│   ├── grading/            Spatially varying cell size and wall thickness
│   │   ├── base.py         Grading ABC + registry
│   │   ├── uniform.py      Constant
│   │   ├── axial.py        Along X / Y / Z, exact phase-integral warp
│   │   ├── radial.py       From a point or an axis
│   │   ├── surface_distance.py  Driven by distance to the imported skin
│   │   └── expression.py   User-typed f(x, y, z)
│   │
│   ├── shapes/             Analytic base shapes, no import needed
│   │   ├── base.py         BaseShape ABC + registry
│   │   └── primitives.py   Box, sphere, cylinder, cone, torus, capsule
│   │
│   ├── voxelize/           Imported mesh -> signed distance field
│   │   └── mesh_sdf.py     Shell raster, flood fill, EDT, narrow-band refine
│   │
│   └── generate/           Orchestration
│       ├── params.py       GenerationParams dataclass, fully serialisable
│       └── pipeline.py     Domain -> field -> intersect -> march -> mesh
│
└── ui/                     Qt layer. Imports features; features never import this.
    ├── main_window.py      Window, menus, docks, generation lifecycle
    ├── viewport.py         PyVista viewport with a no-VTK fallback
    ├── worker.py           QThread generation worker, progress + cancellation
    ├── theme.py            Dark palette and stylesheet
    └── panels/             One panel per feature, mirroring features/
        ├── source_panel.py     Import file or pick a base shape
        ├── tpms_panel.py       Pattern, mode, cell size, thickness
        ├── grading_panel.py    Grading mode and its parameters
        ├── quality_panel.py    Resolution, memory estimate, warnings
        └── export_panel.py     Format, path, write
```

**Dependency direction is strictly one-way:**

```
ui  ->  features  ->  core
        io        ->  core
```

`core` imports nothing from the project. This is what lets any feature be used from a
script, a test, or the CLI without dragging in Qt.

---

## 4. The generation pipeline

`features/generate/pipeline.py` is the only place that knows the full sequence.

```
1. DOMAIN      Imported mesh -> SDF grid        (features/voxelize)
               or base shape -> analytic SDF    (features/shapes)
               Result: phi_domain, negative inside the part.

2. GRID        Bounding box + padding -> Grid   (core/grid)
               Resolution capped, memory estimated, user warned above 384³.

3. FIELD       For each z-slab:
                 a. grading -> cell size and thickness fields  (features/grading)
                 b. pattern -> periodic scalar field F         (features/tpms)
                 c. mode    -> phi_lattice from F and thickness (features/tpms/modes)
                 d. CSG     -> phi = max(phi_lattice, phi_domain)  (core/sdf)
                 e. optional solid skin unioned in
                 f. marching cubes on the slab               (core/marching)

4. MESH        Concatenate slabs, weld the seams, drop degenerates  (core/mesh)

5. OUT         Viewport, or file via io/exporters
```

### Why slab streaming

A full 512³ `float32` field is 537 MB. Marching a slab at a time keeps the resident
field at `nx * ny * slab_depth`, so memory becomes **O(n²)** rather than O(n³) — a 512³
run holds roughly 8 MB of field instead of 537 MB. Only the output triangles accumulate.

### Two independent resolutions

The domain SDF and the TPMS field are sampled on **different grids**:

- The TPMS field is analytic, so it is evaluated at full resolution, slab by slab, for free.
- The domain SDF is expensive to build, so it is computed once at a capped resolution
  (default 192³, never above 256³) and **trilinearly interpolated** up to the marching grid.

The boundary of the part is smooth and low-frequency; the lattice is not. Spending the
grid budget where the detail actually is means a 512³ lattice costs a 192³ SDF.

### Voxelising an imported mesh

Four steps, in `features/voxelize/mesh_sdf.py`: rasterise the surface into a one-voxel
shell, flood-fill the exterior to separate inside from outside, take a Euclidean
distance transform, then re-measure the narrow band against the real surface samples
with a KD-tree. The result lands within about **0.02 voxels** on average.

Two failure modes were found by testing against analytic references, and both are worth
knowing because both produce *plausible-looking* wrong answers rather than errors.

**Sampling must be finer than a voxel, with no cap.** Triangles are subdivided until
their sample spacing is under half a voxel. An early subdivision cap meant a large
triangle — and a CAD block is a handful of very large triangles — sampled more coarsely
than the grid. The shell then has gaps, the exterior flood fill pours through them, and
a perfectly watertight part reports *no interior at all*. The cap is now a safety valve
that raises rather than silently under-sampling.

**Sample normals must be angle-weighted vertex normals, not face normals.** Voxels that
straddle the surface are signed by which side of it their centre falls on. Using the
nearest triangle's own normal fails just outside a convex edge, where a wedge of space
lies *behind* one of the two faces' planes and is therefore called "inside". Every hard
edge in the model grows a fringe of spikes. Angle-weighted vertex normals
(Bærentzen & Aanæs) bisect the wedge and remove the artifact completely.

### Thickness in real millimetres

A TPMS field `F` is not a distance function, so thresholding `|F| < t` gives a wall whose
thickness varies with position. Each pattern therefore also exposes its analytic
`gradient()`, and the field is converted to a distance by

```
d ≈ (F - level) / |∇F|
```

One such step is not enough — at a wall thickness of 0.2 cells it is 12 % low for a
gyroid and 15 % low for a Schwarz D. So the estimate is refined by Newton iteration:
step to the predicted surface point, measure the residual there, add it on. Two passes
bring every pattern below 0.8 % error (Neovius, whose gradient varies fivefold over its
own surface, reaches 2.3 %). Each pass costs one field and one gradient evaluation, so
the default of two makes lattice evaluation roughly 3× the cost of the raw field — a
trade worth making, because it is what allows wall thickness to be a millimetre
dimension the user can dial in rather than an opaque threshold.

The probe step is clamped to a quarter cell so refinement cannot jump onto the
neighbouring wall.

---

## 5. Feature contracts

Four small ABCs. Adding a pattern, a grading law or a shape means writing one file that
implements one of these and registering it — no other file changes.

```python
# features/tpms/base.py
class TPMSPattern:
    name: str
    def field(self, x, y, z) -> np.ndarray      # periodic, period 2π
    def grad_norm(self, x, y, z) -> np.ndarray  # |∇field|, for mm thickness

# features/grading/base.py
class Grading:
    name: str
    def cell_size(self, x, y, z, ctx) -> np.ndarray | float
    def thickness(self, x, y, z, ctx) -> np.ndarray | float

# features/shapes/base.py
class BaseShape:
    name: str
    def sdf(self, x, y, z) -> np.ndarray        # negative inside
    def bounds(self) -> (min_xyz, max_xyz)

# io/registry.py
loader(path) -> Mesh        # registered against a set of extensions
```

`ctx` carries the domain SDF and bounds, which is how `surface_distance` grading reads
the skin distance without importing the voxelize feature.

---

## 6. Where new work goes

| Adding | Put it in | Also touch |
| --- | --- | --- |
| A TPMS pattern | `features/tpms/<name>.py` | register in `features/tpms/__init__.py` |
| A grading law | `features/grading/<name>.py` | register in `features/grading/__init__.py` |
| A base shape | `features/shapes/primitives.py` | register in the same file |
| An import format | `io/<name>_loader.py` | register in `io/registry.py` |
| An export format | `io/exporters.py` | — |
| A UI control | `ui/panels/<feature>_panel.py` | — |
| **Stress analysis (step 2)** | **`features/analysis/`** | new panel, new pipeline stage |

### Reserved for step 2 — not yet built

`features/analysis/` is deliberately absent. The seams it will need already exist:

- `GenerationParams` is serialisable, so a solve can be reproduced from a saved setup.
- The pipeline returns the voxel occupancy alongside the mesh, which is what a
  voxel-based FE solver wants as input.
- `grading/expression.py` evaluates an arbitrary field, so stress-driven grading becomes
  "feed the solved field back in as the thickness expression" rather than new plumbing.

---

## 7. Performance and memory

Measured, not estimated: gyroid sheet filling a 60 mm cube, 8 mm cells, 1.2 mm walls.

| Grid | Field slab | Full field *would* be | Triangles | Peak RSS | Time |
| --- | --- | --- | --- | --- | --- |
| 128³ | 1.0 MB | 8 MB | 1.7 M | 339 MB | 1.8 s |
| 256³ | 4.2 MB | 67 MB | 6.9 M | 1.15 GB | 6.9 s |
| 384³ | 9.4 MB | 227 MB | 15.7 M | 2.5 GB | 18 s |
| 512³ | 16.8 MB | 537 MB | 27.7 M | 4.4 GB | 38 s |

Two things this table settles.

**Slab streaming does its job.** The resident field never exceeds 17 MB even where a
full-size one would be 537 MB.

**The mesh is the real cost.** Above 256³ almost all of that peak is accumulated
triangles, not field. Three changes cut it roughly in half — float32 vertices and int32
indices instead of float64/int64, a preallocated concatenate instead of
`np.concatenate` on a list, and welding restricted to the slab seam planes rather than
every vertex. 512³ went from 8.5 GB to 4.4 GB; without them the top of the range was
not usable at all.

Welding only the seams is the interesting one: slab marching can *only* duplicate
vertices on the planes two slabs share, which is an O(n²) subset of an O(n³) mesh —
a few hundred thousand out of fourteen million. Building a KD-tree over everything asks
a question whose answer is already known for 98 % of the input.

The quality panel estimates triangles and peak memory before running and warns past
384³. Generation runs on a `QThread` with progress and cancellation, so the UI stays
live and a run that turns out too big can be stopped.

Output is verified watertight: every generated lattice tested — sheet, solid network,
with and without a solid skin — has 100 % of edges shared by exactly two faces, no
boundary edges and no non-manifold edges.
