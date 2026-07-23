"""Isosurface extraction, streamed one slab at a time.

The whole point of this module is to never hold a full-resolution field in memory. The
caller supplies a *field function* evaluated per slab; this module marches each slab,
converts vertices to world coordinates, and accumulates only triangles.

Field convention matches :mod:`tpms.core.sdf`: negative inside, surface at zero.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from tpms.core.grid import Grid, Slab
from tpms.core.mesh import Mesh

# Signature of the per-slab evaluator: (grid, slab) -> field array shaped
# (nx, ny, slab.depth), negative inside.
FieldFunction = Callable[[Grid, Slab], np.ndarray]

# Signature of the progress callback: (fraction 0..1, message) -> keep going?
# Returning False cancels the run.
ProgressCallback = Callable[[float, str], bool]


class GenerationCancelled(RuntimeError):
    """Raised when a progress callback asks for cancellation."""


def _marching_cubes(volume: np.ndarray, spacing: Sequence[float], level: float = 0.0):
    """Thin wrapper over scikit-image, isolating the one API detail we depend on."""
    from skimage.measure import marching_cubes

    # skimage needs a genuine sign change; a slab entirely inside or entirely outside
    # has no surface and raises rather than returning empty.
    vmin = float(volume.min())
    vmax = float(volume.max())
    if not (vmin <= level <= vmax) or vmin == vmax:
        return None, None

    try:
        verts, faces, _normals, _values = marching_cubes(
            volume, level=level, spacing=tuple(spacing), allow_degenerate=False
        )
    except (ValueError, RuntimeError):
        # Degenerate slab (e.g. the level touches but never crosses). No surface here.
        return None, None

    if len(faces) == 0:
        return None, None
    return verts, faces


def march_slabs(
    grid: Grid,
    field_fn: FieldFunction,
    slab_depth: int = 16,
    level: float = 0.0,
    progress: ProgressCallback | None = None,
    weld_tolerance: float | None = None,
    min_component_faces: int = 12,
) -> Mesh:
    """Extract the ``level`` isosurface of a field over ``grid``, slab by slab.

    ``field_fn`` is called once per slab and must return an array shaped
    ``(nx, ny, slab.depth)``. Consecutive slabs overlap by one plane, so the surface is
    continuous across seams and the duplicated seam vertices weld away at the end.

    ``weld_tolerance`` defaults to a thousandth of a voxel: far below the spacing
    between genuinely distinct marching-cubes vertices, far above the float32 noise in
    the two copies of a seam vertex.

    Raises :class:`GenerationCancelled` if ``progress`` returns ``False``.
    """
    if weld_tolerance is None:
        weld_tolerance = float(grid.spacing.min()) * 1e-3

    pieces: list[Mesh] = []
    #: Indices, in the concatenated mesh, of vertices sitting on a slab seam. Only
    #: these can have duplicates, so only these are offered to the weld.
    seam_indices: list[np.ndarray] = []
    vertex_offset = 0

    slabs = list(grid.slabs(slab_depth))
    total = len(slabs)

    for slab in slabs:
        if progress is not None:
            fraction = 0.05 + 0.85 * (slab.index / max(total, 1))
            if not progress(fraction, f"Meshing slab {slab.index + 1} of {total}"):
                raise GenerationCancelled("cancelled during meshing")

        volume = field_fn(grid, slab)

        expected = (grid.shape[0], grid.shape[1], slab.depth)
        if volume.shape != expected:
            raise ValueError(
                f"field function returned {volume.shape}, expected {expected}"
            )

        verts, faces = _marching_cubes(volume, grid.spacing, level)
        del volume  # release the slab before the next allocation

        if verts is None:
            continue

        # skimage returns coordinates relative to the sub-volume origin.
        verts = verts + grid.slab_origin(slab)

        # Only two kinds of vertex can ever have a duplicate, and both are cheap to
        # identify — which is what keeps the weld off the other 98 % of the mesh.
        #
        # 1. Seam vertices, on the plane this slab shares with its neighbour.
        # 2. Vertices sitting exactly on a grid node. scikit-image keys each vertex by
        #    the cell edge that produced it, so a field value of exactly zero at a node
        #    yields one coincident vertex per incident edge — up to six copies of the
        #    same point. Sheet mode almost never lands on an exact zero, but a solid
        #    network at 50 % volume fraction marches the level the pattern is
        #    symmetric about, and hits them regularly.
        local_z = verts[:, 2]
        z_low = grid.z[slab.z_start]
        z_high = grid.z[slab.z_stop - 1]
        on_seam = (np.abs(local_z - z_low) <= weld_tolerance) | (
            np.abs(local_z - z_high) <= weld_tolerance
        )

        node_fraction = (verts - grid.origin) / grid.spacing
        on_node = np.all(
            np.abs(node_fraction - np.round(node_fraction)) < 1e-4, axis=1
        )

        candidates = np.flatnonzero(on_seam | on_node)
        seam_indices.append(candidates + vertex_offset)
        vertex_offset += len(verts)

        pieces.append(Mesh(verts, faces))

    if progress is not None and not progress(0.92, "Welding seams"):
        raise GenerationCancelled("cancelled during welding")

    if not pieces:
        return Mesh.empty()

    combined = Mesh.concatenate(pieces)
    pieces.clear()

    candidates = (
        np.concatenate(seam_indices) if seam_indices else np.zeros(0, dtype=np.int64)
    )
    seam_indices.clear()

    # scikit-image already winds faces so normals point out of the negative region,
    # which is the outward direction under this project's negative-inside convention.
    # No flip needed; adding one inverts every exported solid.
    result = (
        combined.weld_subset(candidates, weld_tolerance)
        .remove_unreferenced_vertices()
        .remove_small_components(min_component_faces)
    )

    if progress is not None and not progress(1.0, "Done"):
        raise GenerationCancelled("cancelled after meshing")

    return result


def march_field(
    grid: Grid,
    field: np.ndarray,
    level: float = 0.0,
    weld_tolerance: float | None = None,
    min_component_faces: int = 12,
) -> Mesh:
    """Extract an isosurface from a field that is already fully materialised.

    Convenience for small grids and tests. Prefer :func:`march_slabs` for anything at
    working resolution — a 512³ float32 field is 537 MB.
    """
    if field.shape != grid.shape:
        raise ValueError(f"field shape {field.shape} != grid shape {grid.shape}")

    if weld_tolerance is None:
        weld_tolerance = float(grid.spacing.min()) * 1e-3

    verts, faces = _marching_cubes(field, grid.spacing, level)
    if verts is None:
        return Mesh.empty()

    mesh = Mesh(verts + grid.origin, faces)
    return mesh.clean(tolerance=weld_tolerance, min_faces=min_component_faces)
