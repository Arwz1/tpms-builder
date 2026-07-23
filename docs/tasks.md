# TPMS Builder — Implementation Tasks

Build order. Each row is one file, written complete before the next begins, per `claude.md`.
Dependencies always point at rows already finished, so the tree compiles at every step.

Legend: `[x]` done · `[ ]` pending · `[-]` deliberately deferred

---

## Stage 0 — Environment

| | Task | Notes |
| --- | --- | --- |
| [x] | Confirm Python and wheel availability | Python 3.14.3; all wheels resolve |
| [x] | Create `.venv` and install core stack | PySide6 6.11.1, numpy 2.5.1, scipy 1.18.0, skimage 0.26.0, trimesh 4.12.2, pyvista 0.48.4 |
| [x] | Install OpenCascade for STEP | cadquery-ocp 7.9.3.1.1, STEP reader verified |
| [x] | `docs/architecture.md` | Layering, pipeline, contracts |
| [x] | `docs/tasks.md` | This file |
| [x] | `requirements.txt`, `requirements-step.txt` | STEP kept as an optional extra |

## Stage 1 — Core (`tpms/core/`) — no project imports, numpy/scipy only

| | Task | File | Depends on |
| --- | --- | --- | --- |
| [x] | Grid: bounds, spacing, coordinate arrays, slab iteration, memory estimate | `core/grid.py` | — |
| [x] | Mesh container: welding, degenerate removal, stats, transform | `core/mesh.py` | — |
| [x] | CSG ops on SDF arrays: union, intersect, subtract, shell, offset, smooth-union | `core/sdf.py` | — |
| [x] | Slab-streamed marching cubes with seam welding | `core/marching.py` | grid, mesh |
| [x] | Sandboxed `f(x, y, z)` expression evaluator | `core/expressions.py` | — |

## Stage 2 — I/O (`tpms/io/`)

| | Task | File | Depends on |
| --- | --- | --- | --- |
| [x] | Loader registry, extension dispatch, capability probe | `io/registry.py` | core/mesh |
| [x] | STL / OBJ / PLY / 3MF read via trimesh | `io/mesh_loader.py` | registry |
| [x] | STEP / IGES read via OCP, lazily imported, tessellation tolerance | `io/step_loader.py` | registry |
| [x] | Export STL / OBJ / PLY / 3MF | `io/exporters.py` | core/mesh |

## Stage 3 — Features (`tpms/features/`) — each standalone

### 3a. TPMS patterns
| | Task | File |
| --- | --- | --- |
| [x] | `TPMSPattern` ABC, registry, `get_pattern` / `list_patterns` | `features/tpms/base.py` |
| [x] | Schoen Gyroid + analytic gradient norm | `features/tpms/gyroid.py` |
| [x] | Schwarz Primitive | `features/tpms/schwarz_p.py` |
| [x] | Schwarz Diamond | `features/tpms/schwarz_d.py` |
| [x] | Neovius | `features/tpms/neovius.py` |
| [x] | Solidification modes: sheet, solid network, inverse network | `features/tpms/modes.py` |

### 3b. Grading
| | Task | File |
| --- | --- | --- |
| [x] | `Grading` ABC, `GradingContext`, registry | `features/grading/base.py` |
| [x] | Uniform | `features/grading/uniform.py` |
| [x] | Axial along X/Y/Z, exact phase-integral warp | `features/grading/axial.py` |
| [x] | Radial from point or axis | `features/grading/radial.py` |
| [x] | Distance-to-surface driven | `features/grading/surface_distance.py` |
| [x] | User expression driven | `features/grading/expression.py` |

### 3c. Base shapes
| | Task | File |
| --- | --- | --- |
| [x] | `BaseShape` ABC + registry | `features/shapes/base.py` |
| [x] | Box, sphere, cylinder, cone, torus, capsule — analytic SDFs | `features/shapes/primitives.py` |

### 3d. Voxelisation
| | Task | File |
| --- | --- | --- |
| [x] | Mesh -> SDF: shell raster, flood fill, EDT sign, narrow-band KD-tree refine | `features/voxelize/mesh_sdf.py` |

### 3e. Generation
| | Task | File |
| --- | --- | --- |
| [x] | `GenerationParams` dataclass, serialisable, validated | `features/generate/params.py` |
| [x] | Pipeline: domain -> grid -> slab field -> CSG -> march -> weld | `features/generate/pipeline.py` |

## Stage 4 — UI (`tpms/ui/`)

| | Task | File |
| --- | --- | --- |
| [x] | Dark theme palette and stylesheet | `ui/theme.py` |
| [x] | PyVista viewport with graceful no-VTK fallback | `ui/viewport.py` |
| [x] | `QThread` generation worker: progress, cancel, error marshalling | `ui/worker.py` |
| [x] | Source panel — import file or choose base shape | `ui/panels/source_panel.py` |
| [x] | TPMS panel — pattern, mode, cell size, thickness | `ui/panels/tpms_panel.py` |
| [x] | Grading panel — mode and its parameters, stacked | `ui/panels/grading_panel.py` |
| [x] | Quality panel — resolution, live memory/triangle estimate, >384³ warning | `ui/panels/quality_panel.py` |
| [x] | Export panel — format, path, write | `ui/panels/export_panel.py` |
| [x] | Main window — docks, menus, lifecycle, status bar | `ui/main_window.py` |
| [x] | App bootstrap | `app.py`, `run.py` |

## Stage 5 — Standalone proof and verification

| | Task | File |
| --- | --- | --- |
| [x] | Headless CLI driving the pipeline with no Qt | `cli.py` |
| [x] | Smoke test: every pattern, mode, grading, shape, format | `tests/test_smoke.py` |
| [x] | Verify each feature imports and runs on its own | `tests/test_standalone.py` |
| [x] | `README.md` — install, run, CLI usage | `README.md` |

**59 tests, ~45 s, all passing.**

### Defects found by testing, and fixed

Recorded because each produced a plausible-looking wrong answer rather than an error.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Hairline cracks at slab seams | Absolute weld tolerance sat at the float32 noise floor; rounding ties split coincident vertices | Proximity weld via KD-tree, tolerance relative to voxel size |
| Every exported solid inside-out | An unnecessary winding flip after marching | Removed; scikit-image already winds outward |
| Walls 6–15 % thin, Neovius far worse | Single-step field-to-distance conversion | Newton refinement, clamped to a quarter cell |
| 8.5 GB peak at 512³ | float64/int64 mesh, list concatenate, welding all 14 M vertices | float32/int32, preallocated concatenate, weld seam planes only → 4.4 GB |
| 8 boundary edges in network modes | Exact zeros at grid nodes emit one vertex per incident edge; seam-only weld missed them | Node-coincident vertices added to weld candidates |
| Imported part 4–10 % oversized | Every shell voxel forced solid, adding a voxel-thick skin | Shell voxels signed by surface normal |
| Watertight CAD part reports "no interior" | Subdivision cap under-sampled large triangles; flood fill leaked through shell gaps | Cap raised and made an error, not silent degradation |
| Spikes along every sharp CAD edge | Per-face normal calls the wedge behind a face "inside" | Angle-weighted vertex normals |
| App segfaults with no OpenGL | VTK dies in C++, uncatchable from Python | Probe the GL context before building the viewport |
| Triangle/memory estimates 2–45 % out | Guessed constants; cells counted across grid padding | Refitted to measurement, cells counted across the part |

## Stage 6 — Stress analysis

| | Task | File |
| --- | --- | --- |
| [-] | **Deferred — awaiting instruction.** Reserved namespace `features/analysis/` | — |

Nothing in stages 0–5 writes to `features/analysis/`. The hooks it will use
(serialisable params, voxel occupancy returned from the pipeline, expression-driven
grading for stress-based thickness) are in place and documented in
`docs/architecture.md` §6.
