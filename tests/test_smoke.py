"""End-to-end coverage: every pattern, mode, grading, shape and file format.

Kept at low resolution so the whole file runs in well under a minute. The numerical
accuracy work lives in the assertions here — watertightness, wall thickness and volume
are checked against values that were measured, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpms.core import Grid, march_field, march_slabs, sdf
from tpms.core.mesh import Mesh
from tpms.features.generate import GenerationParams, SourceType, generate
from tpms.features.grading import grading_names
from tpms.features.shapes import shape_names
from tpms.features.tpms import SolidMode, list_patterns, pattern_names, solidify

RESOLUTION = 96


def manifold_fraction(mesh: Mesh) -> float:
    """Fraction of edges shared by exactly two faces. 1.0 means watertight."""
    faces = mesh.faces
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return float((counts == 2).mean())


def base_params(**overrides) -> GenerationParams:
    params = GenerationParams(
        source_type=SourceType.SHAPE,
        shape_name="box",
        shape_params={"width": 40.0, "depth": 40.0, "height": 40.0},
        resolution=RESOLUTION,
        grading_params={"cell": 8.0, "wall": 1.4},
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


# ------------------------------------------------------------------------ core

def test_slab_streaming_matches_full_march():
    """Slab streaming must be an optimisation, not an approximation."""
    grid = Grid.from_bounds([-10] * 3, [10] * 3, 64)
    x, y, z = grid.meshgrid()

    whole = march_field(grid, sdf.sphere(x, y, z, 7.0))

    def field_fn(g, slab):
        sx, sy, sz = g.slab_meshgrid(slab)
        return sdf.sphere(sx, sy, sz, 7.0)

    for depth in (4, 8, 16):
        streamed = march_slabs(grid, field_fn, slab_depth=depth)
        assert streamed.num_faces == whole.num_faces
        assert streamed.num_vertices == whole.num_vertices
        assert streamed.volume() == pytest.approx(whole.volume(), rel=1e-6)


def test_sphere_volume_and_orientation():
    grid = Grid.from_bounds([-10] * 3, [10] * 3, 80)
    x, y, z = grid.meshgrid()
    mesh = march_field(grid, sdf.sphere(x, y, z, 7.0))

    # Positive volume means outward-facing normals.
    assert mesh.volume() > 0
    assert mesh.volume() == pytest.approx(4 / 3 * np.pi * 7 ** 3, rel=0.01)
    assert manifold_fraction(mesh) == 1.0


# -------------------------------------------------------------------- patterns

@pytest.mark.parametrize("name", pattern_names())
def test_pattern_gradient_matches_numerical(name):
    from tpms.features.tpms import get_pattern

    pattern = get_pattern(name)
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 2 * np.pi, (500, 3))
    u, v, w = points[:, 0], points[:, 1], points[:, 2]

    h = 1e-6
    numerical = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        high = pattern.field(u + step[0], v + step[1], w + step[2])
        low = pattern.field(u - step[0], v - step[1], w - step[2])
        numerical.append((high - low) / (2 * h))

    expected = np.sqrt(sum(n ** 2 for n in numerical))
    assert np.allclose(pattern.gradient_norm(u, v, w), expected, atol=1e-6)


@pytest.mark.parametrize("name", pattern_names())
def test_wall_thickness_is_accurate(name):
    """Refinement must keep the wall within a few percent of the requested value.

    Measured as solid volume divided by the area of the underlying minimal surface,
    over an exact number of unit cells.
    """
    from tpms.features.tpms import get_pattern

    pattern = get_pattern(name)
    cell = 6.0
    thickness = 0.8            # 0.13 cells: comfortably inside the printable range
    length = cell * 2

    grid = Grid.from_bounds([0, 0, 0], [length] * 3, 150)
    scale = np.float32(2 * np.pi / cell)

    def zero_level(g, slab):
        sx, sy, sz = g.slab_meshgrid(slab)
        return pattern.field(sx * scale, sy * scale, sz * scale).astype(np.float32)

    area = march_slabs(grid, zero_level, slab_depth=24, min_component_faces=0).area()

    axis = np.linspace(0, length, 200, dtype=np.float32)
    solid = 0
    for chunk in np.array_split(axis, 8):
        gx, gy, gz = np.meshgrid(axis, axis, chunk, indexing="ij")
        field = solidify(
            pattern, gx * scale, gy * scale, gz * scale, scale,
            SolidMode.SHEET, thickness=thickness, refine_steps=2,
        )
        solid += int((field < 0).sum())

    fraction = solid / 200 ** 3
    measured = fraction * length ** 3 / area

    # The metric itself under-reads by a t^3 Gaussian-curvature term, so a few percent
    # low is expected and correct; well outside that means the refinement broke.
    assert measured == pytest.approx(thickness, rel=0.12)


def test_refinement_beats_first_order():
    """The refinement has to actually earn its cost."""
    from tpms.features.tpms import get_pattern

    pattern = get_pattern("schwarz_d")
    cell, thickness = 6.0, 1.2
    scale = np.float32(2 * np.pi / cell)
    length = cell * 2

    axis = np.linspace(0, length, 140, dtype=np.float32)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")

    grid = Grid.from_bounds([0, 0, 0], [length] * 3, 140)

    def zero_level(g, slab):
        sx, sy, sz = g.slab_meshgrid(slab)
        return pattern.field(sx * scale, sy * scale, sz * scale).astype(np.float32)

    area = march_slabs(grid, zero_level, slab_depth=24, min_component_faces=0).area()

    def measure(refine):
        field = solidify(
            pattern, gx * scale, gy * scale, gz * scale, scale,
            SolidMode.SHEET, thickness=thickness, refine_steps=refine,
        )
        return float((field < 0).mean()) * length ** 3 / area

    crude = abs(measure(0) - thickness)
    refined = abs(measure(2) - thickness)
    assert refined < crude


# --------------------------------------------------------------------- shapes

@pytest.mark.parametrize("name", shape_names())
def test_every_shape_generates(name):
    from tpms.cli import _shape_params

    params = base_params(shape_name=name, shape_params=_shape_params(name, 40.0))
    result = generate(params)

    assert not result.mesh.is_empty
    assert manifold_fraction(result.mesh) == 1.0


# ------------------------------------------------------------------- lattices

@pytest.mark.parametrize("name", pattern_names())
def test_every_pattern_generates(name):
    result = generate(base_params(pattern=name))
    assert result.mesh.num_faces > 1000
    assert manifold_fraction(result.mesh) == 1.0


@pytest.mark.parametrize("mode", list(SolidMode))
def test_every_mode_generates(mode):
    result = generate(base_params(mode=mode))
    assert result.mesh.num_faces > 1000
    assert manifold_fraction(result.mesh) == 1.0


@pytest.mark.parametrize("name", grading_names())
def test_every_grading_generates(name):
    grading_params = {
        "uniform": {"cell": 8.0, "wall": 1.4},
        "axial": {"axis": 2, "cell_start": 6.0, "cell_end": 12.0,
                  "wall_start": 1.4, "wall_end": 1.4},
        "radial": {"axis": None, "radius_inner": 0.0, "radius_outer": 20.0,
                   "cell_inner": 6.0, "cell_outer": 12.0,
                   "wall_inner": 1.4, "wall_outer": 1.4},
        "surface_distance": {"depth": 10.0, "cell_surface": 6.0, "cell_core": 12.0,
                             "wall_surface": 1.4, "wall_core": 1.4},
        "expression": {"cell_expression": "6 + 6*w", "wall_expression": "1.4"},
    }[name]

    result = generate(base_params(grading=name, grading_params=grading_params))
    assert result.mesh.num_faces > 1000
    assert manifold_fraction(result.mesh) == 1.0


def test_solid_skin_is_watertight():
    result = generate(base_params(skin_thickness=1.5))
    assert manifold_fraction(result.mesh) == 1.0
    # A skin can only add material.
    plain = generate(base_params())
    assert result.relative_density > plain.relative_density


# ------------------------------------------------------------------- expression

def test_expression_rejects_unsafe_input():
    from tpms.core.expressions import validate_expression

    for bad in ("__import__('os')", "x.__class__", "open('f')", "1/0", "log(0)"):
        assert validate_expression(bad) is not None

    for good in ("4 + 3*sin(x/10)", "clamp(u*8, 2, 9)", "lerp(3, 9, w)"):
        assert validate_expression(good) is None


# ------------------------------------------------------------------------- io

@pytest.mark.parametrize("extension", ["stl", "obj", "ply", "3mf", "off"])
def test_export_import_round_trip(tmp_path, extension):
    from tpms.io import export_mesh, load

    grid = Grid.from_bounds([-10] * 3, [10] * 3, 48)
    x, y, z = grid.meshgrid()
    mesh = march_field(grid, sdf.sphere(x, y, z, 7.0))

    path = tmp_path / f"round_trip.{extension}"
    export_mesh(mesh, path)
    reloaded = load(path)

    assert reloaded.num_faces == mesh.num_faces
    assert reloaded.volume() == pytest.approx(mesh.volume(), rel=1e-3)


def test_step_round_trip(tmp_path):
    """STEP import, exercised against a file OpenCascade wrote itself."""
    ocp = pytest.importorskip("OCP", reason="cadquery-ocp is not installed")

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer
    from OCP.gp import gp_Pnt

    from tpms.io import load

    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 40, 30, 20).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)

    path = tmp_path / "block.step"
    writer.Write(str(path))

    mesh = load(path)
    assert mesh.volume() == pytest.approx(40 * 30 * 20, rel=0.01)


def test_unsupported_format_is_reported(tmp_path):
    from tpms.io import UnsupportedFormatError, load

    path = tmp_path / "geometry.xyz"
    path.write_text("nonsense")

    with pytest.raises(UnsupportedFormatError):
        load(path)


# ------------------------------------------------------------------ voxelise

def test_imported_geometry_is_filled(tmp_path):
    """The full path: mesh on disk, voxelised, filled, exported."""
    from tpms.io import export_mesh

    grid = Grid.from_bounds([-16] * 3, [16] * 3, 64)
    x, y, z = grid.meshgrid()
    source = march_field(grid, sdf.cylinder(x, y, z, 12.0, 24.0))

    path = tmp_path / "source.stl"
    export_mesh(source, path)

    params = base_params(
        source_type=SourceType.FILE,
        source_path=str(path),
        domain_resolution=80,
        grading_params={"cell": 7.0, "wall": 1.3},
    )
    result = generate(params)

    assert not result.mesh.is_empty
    assert manifold_fraction(result.mesh) == 1.0
    # The lattice must sit inside the source, not overflow it.
    lower, upper = result.mesh.bounds
    assert lower[0] >= -13.0 and upper[0] <= 13.0


def _exact_cube(size: float = 20.0) -> Mesh:
    """A cube as 8 vertices and 12 triangles — genuinely sharp edges.

    A marching-cubes cube has slightly bevelled edges and does not exercise the same
    code path.
    """
    h = size / 2
    vertices = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],   # -Z
        [4, 5, 6], [4, 6, 7],   # +Z
        [0, 1, 5], [0, 5, 4],   # -Y
        [2, 3, 7], [2, 7, 6],   # +Y
        [1, 2, 6], [1, 6, 5],   # +X
        [0, 4, 7], [0, 7, 3],   # -X
    ], dtype=np.int64)
    return Mesh(vertices, faces)


def test_sharp_edges_do_not_grow_spikes():
    """Voxelising a hard-edged solid must not bulge at the edges.

    Just outside a convex edge lies a wedge that is behind one of the two faces'
    planes. Signing a voxel from the nearest single face normal calls that wedge
    'inside' and the mesh sprouts spikes along every edge, so sample normals are
    angle-weighted vertex normals instead. This pins that down.
    """
    from tpms.features.voxelize import build_domain_field

    cube = _exact_cube(20.0)
    domain = build_domain_field(cube, resolution=96)

    assert domain.solid_volume() == pytest.approx(8000.0, rel=0.03)

    # No point beyond half a voxel outside the cube may be marked solid.
    gx, gy, gz = domain.grid.meshgrid(dtype=np.float64)
    voxel = float(domain.grid.spacing[0])

    outside = (
        (np.abs(gx) > 10.0 + voxel)
        | (np.abs(gy) > 10.0 + voxel)
        | (np.abs(gz) > 10.0 + voxel)
    )
    assert not (domain.values[outside] < 0).any(), (
        f"{int((domain.values[outside] < 0).sum())} voxels outside the cube were "
        "marked solid — the edge sign test has regressed"
    )


def test_large_triangles_are_sampled_finely_enough():
    """A CAD block is a few huge triangles; sampling must still seal the shell.

    Capping subdivision made samples land further apart than a voxel, the exterior
    flood fill poured through the gaps, and a perfectly watertight part came back with
    no interior at all.
    """
    from tpms.features.voxelize import build_domain_field

    # One 200 mm cube: 12 triangles with 280 mm diagonals, at a sub-millimetre voxel.
    cube = _exact_cube(200.0)
    domain = build_domain_field(cube, resolution=192)

    assert domain.solid_volume() == pytest.approx(200.0 ** 3, rel=0.03)


def test_non_watertight_mesh_is_reported():
    from tpms.features.voxelize import NotWatertightError, build_domain_field

    grid = Grid.from_bounds([-12] * 3, [12] * 3, 48)
    x, y, z = grid.meshgrid()
    closed = march_field(grid, sdf.sphere(x, y, z, 8.0))

    torn = Mesh(closed.vertices, closed.faces[: len(closed.faces) // 2])

    with pytest.raises(NotWatertightError):
        build_domain_field(torn, resolution=56)


# ------------------------------------------------------------------- params

def test_params_round_trip_through_json(tmp_path):
    params = base_params(
        pattern="neovius", mode=SolidMode.SOLID,
        grading="expression",
        grading_params={"cell_expression": "5+3*u", "wall_expression": "1.1"},
    )

    path = tmp_path / "settings.json"
    params.save(path)
    restored = GenerationParams.load(path)

    assert restored.pattern == "neovius"
    assert restored.mode is SolidMode.SOLID
    assert restored.grading_params["cell_expression"] == "5+3*u"
    assert restored.to_dict() == params.to_dict()


def test_params_validation_catches_bad_names():
    assert generate_fails(base_params(pattern="does_not_exist"))
    assert generate_fails(base_params(grading="does_not_exist"))


def generate_fails(params) -> bool:
    try:
        generate(params)
    except (ValueError, KeyError):
        return True
    return False


def test_resolution_is_clamped():
    # Clamped at construction.
    assert GenerationParams(resolution=99999).resolution == 512

    # And again after mutation, since the UI and CLI both build then assign.
    params = base_params(resolution=99999, slab_depth=0, volume_fraction=5.0)
    params.normalise()

    assert params.resolution == 512
    assert params.slab_depth == 2
    assert params.volume_fraction == 0.99
