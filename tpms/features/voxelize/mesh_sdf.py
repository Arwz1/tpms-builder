"""Turn an imported triangle mesh into a signed distance field.

Four steps, each chosen because the obvious alternative is too slow at working
resolution:

1. **Rasterise the surface.** Every triangle is subdivided until its sample spacing is
   under half a voxel, and the samples mark a one-voxel-thick shell. Guaranteed gap-free
   coverage, unlike random surface sampling, which can leave a hole that the flood fill
   then leaks through.

2. **Flood fill from outside.** Label the complement of the shell and take the component
   touching the grid border as exterior. This is what determines inside from outside,
   and it costs one pass instead of a ray test or winding number per voxel — millions
   of point-in-mesh queries would dominate the whole run.

3. **Distance by exact EDT.** A Euclidean distance transform of the shell, run once each
   way and combined with the sign from step 2.

4. **Refine the narrow band.** The EDT measures distance to the nearest shell *voxel
   centre*, so it is quantised to roughly half a voxel. Within two voxels of the
   surface — the only place the value steers anything — the distance is recomputed
   against the actual surface samples with a KD-tree. That band holds a few hundred
   thousand voxels rather than the millions in the full grid, so the accurate method is
   affordable exactly where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tpms.core.grid import Grid, sample_trilinear
from tpms.core.mesh import Mesh

# Sample spacing as a fraction of voxel size. Half a voxel guarantees that consecutive
# samples on a triangle cannot straddle a voxel without marking it.
SAMPLE_SPACING_RATIO = 0.5

# Refine the distance within this many voxels of the surface.
NARROW_BAND_VOXELS = 2.5

# Safety valve on subdivision, hit only by a single triangle spanning more than ~4000
# voxels. Exceeding it raises rather than under-sampling: a sample spacing wider than a
# voxel leaves gaps in the shell, the exterior flood fill pours through them, and the
# part comes back with no interior at all. A CAD export makes this easy to trigger —
# a plain block is a handful of very large triangles.
MAX_SUBDIVISION_LEVEL = 12

# Points generated per batch. Total sample count is bounded by surface area over
# spacing squared regardless of triangle count, but one group of large triangles can
# spike, so each group is emitted in batches of about this size.
SAMPLE_BATCH_POINTS = 4_000_000


class NotWatertightError(ValueError):
    """The flood fill found no interior — the mesh is open or inside-out."""


def angle_weighted_vertex_normals(mesh: Mesh) -> np.ndarray:
    """Per-vertex normals, each incident face weighted by its corner angle.

    Plain face normals are not enough to decide which side of the surface a point is
    on. Just outside a convex edge there is a wedge of space that lies *behind* one of
    the two faces' planes; testing against that face alone reports "inside", and the
    voxeliser grows a spike along every hard edge in the model. Faces and smooth
    regions are unaffected, which is why the error appears only on machined-looking
    parts and only along their edges.

    Angle weighting is what makes the averaged normal correct rather than merely
    smooth: it is the construction (Bærentzen & Aanæs) for which the sign test is
    provably right everywhere on a closed mesh, including at edges and corners, because
    the weighted normal bisects the wedge instead of favouring whichever triangle
    happened to be nearest.
    """
    triangles = mesh.vertices[mesh.faces].astype(np.float64)

    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(
        face_normals, lengths, out=np.zeros_like(face_normals), where=lengths > 0
    )

    normals = np.zeros_like(mesh.vertices, dtype=np.float64)

    for corner in range(3):
        a = triangles[:, corner]
        b = triangles[:, (corner + 1) % 3]
        c = triangles[:, (corner + 2) % 3]

        u = b - a
        v = c - a
        u_len = np.linalg.norm(u, axis=1, keepdims=True)
        v_len = np.linalg.norm(v, axis=1, keepdims=True)

        u = np.divide(u, u_len, out=np.zeros_like(u), where=u_len > 0)
        v = np.divide(v, v_len, out=np.zeros_like(v), where=v_len > 0)

        angle = np.arccos(np.clip(np.einsum("ij,ij->i", u, v), -1.0, 1.0))

        # Scatter-add: a vertex collects a contribution from every incident face.
        np.add.at(normals, mesh.faces[:, corner], face_normals * angle[:, None])

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(
        normals, lengths, out=np.zeros_like(normals), where=lengths > 0
    )


def _barycentric_grid(n: int) -> np.ndarray:
    """Barycentric coordinates for a triangle subdivided ``n`` times per edge."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = (i + j) <= n
    i = i[keep]
    j = j[keep]
    k = n - i - j
    return np.stack([i, j, k], axis=1).astype(np.float64) / max(n, 1)


def surface_samples(
    mesh: Mesh, spacing: float, with_normals: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Points covering the mesh surface no more than ``spacing`` apart.

    Triangles are grouped by the subdivision level they need, so one barycentric
    template serves every triangle in a group and the whole thing stays vectorised.

    With ``with_normals``, also returns the outward normal of the triangle each sample
    came from. That is what lets a voxel straddling the surface be signed by which side
    of the surface its centre is on.
    """
    triangles = mesh.vertices[mesh.faces]  # (F, 3, 3)

    edges = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    longest = edges.max(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        needed = np.ceil(np.log2(np.maximum(longest / spacing, 1.0)))
    needed = np.nan_to_num(needed, nan=0.0, posinf=MAX_SUBDIVISION_LEVEL + 1)

    if needed.max() > MAX_SUBDIVISION_LEVEL:
        raise ValueError(
            f"a triangle spans {longest.max():.1f} mm, more than "
            f"{2 ** MAX_SUBDIVISION_LEVEL} sampling steps of {spacing:.4g} mm. "
            "Lower the voxelisation resolution, or subdivide the input mesh."
        )

    levels = np.clip(needed, 0, MAX_SUBDIVISION_LEVEL).astype(np.int64)

    # The n=1 barycentric template is exactly the three corners, so every original
    # vertex is already covered and needs no separate block.
    blocks: list[np.ndarray] = []
    normal_blocks: list[np.ndarray] = []

    # Vertex normals, interpolated across each triangle, rather than one flat normal
    # per face — see :func:`angle_weighted_vertex_normals` for why the flat version
    # gets the sign wrong at edges.
    vertex_normals = angle_weighted_vertex_normals(mesh) if with_normals else None

    for level in np.unique(levels):
        mask = levels == level
        selected = triangles[mask]
        n = int(2 ** int(level))
        weights = _barycentric_grid(n)  # (P, 3)

        # Emit in batches. A group of large triangles produces points_per_triangle in
        # the tens of thousands, and doing the whole group at once is a transient
        # allocation far larger than the result it contributes.
        per_triangle = max(len(weights), 1)
        batch = max(1, SAMPLE_BATCH_POINTS // per_triangle)

        selected_normals = (
            vertex_normals[mesh.faces[mask]] if with_normals else None
        )

        for start in range(0, len(selected), batch):
            chunk = selected[start : start + batch]

            # (T, 3, 3) x (P, 3) -> (T, P, 3)
            points = np.einsum("pj,tjc->tpc", weights, chunk)
            blocks.append(points.reshape(-1, 3))

            if with_normals:
                # Same barycentric weights as the positions, so each sample carries the
                # normal of the point it actually sits at.
                block = np.einsum(
                    "pj,tjc->tpc", weights, selected_normals[start : start + batch]
                ).reshape(-1, 3)

                lengths = np.linalg.norm(block, axis=1, keepdims=True)
                np.divide(block, lengths, out=block, where=lengths > 0)
                normal_blocks.append(block)

    points = np.concatenate(blocks, axis=0)

    if not with_normals:
        return points

    return points, np.concatenate(normal_blocks, axis=0)


def _shell_mask(grid: Grid, points: np.ndarray) -> np.ndarray:
    """Mark every voxel containing a surface sample."""
    index = np.floor((points - grid.origin) / grid.spacing).astype(np.int64)
    np.clip(index, 0, np.asarray(grid.shape) - 1, out=index)

    shell = np.zeros(grid.shape, dtype=bool)
    shell[index[:, 0], index[:, 1], index[:, 2]] = True
    return shell


def _interior_mask(shell: np.ndarray) -> np.ndarray:
    """Everything the exterior flood fill cannot reach, excluding the shell itself."""
    from scipy.ndimage import label

    free = ~shell
    labels, count = label(free)

    if count == 0:
        raise NotWatertightError(
            "the mesh fills the entire grid — it may be larger than its bounding box "
            "suggests, or the resolution is too low to resolve it"
        )

    # Any label present on the six faces of the grid is exterior.
    border = np.concatenate([
        labels[0, :, :].ravel(), labels[-1, :, :].ravel(),
        labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ])
    exterior_labels = np.unique(border)
    exterior_labels = exterior_labels[exterior_labels != 0]

    exterior = np.isin(labels, exterior_labels)
    return free & ~exterior


def mesh_to_sdf(
    mesh: Mesh,
    grid: Grid,
    refine: bool = True,
    strict: bool = True,
) -> np.ndarray:
    """Sample a signed distance field for ``mesh`` over ``grid``.

    Negative inside, matching :mod:`tpms.core.sdf`.

    ``strict`` raises :class:`NotWatertightError` when no interior is found. With
    ``strict=False`` the same situation returns an all-positive field, which yields an
    empty result rather than an exception.
    """
    from scipy.ndimage import distance_transform_edt

    spacing = float(grid.spacing.min())

    points, normals = surface_samples(
        mesh, spacing * SAMPLE_SPACING_RATIO, with_normals=True
    )
    shell = _shell_mask(grid, points)

    if not shell.any():
        raise NotWatertightError(
            "no part of the mesh falls inside the sampling grid"
        )

    interior = _interior_mask(shell)

    if not interior.any():
        if strict:
            raise NotWatertightError(
                "no enclosed interior was found. The mesh is probably not watertight — "
                "it has holes, or flipped faces that let the fill escape. Repair it, or "
                "raise the resolution if the walls are thinner than one voxel."
            )
        return np.full(grid.shape, spacing, dtype=np.float32)

    # Solid = interior plus the shell voxels themselves.
    solid = interior | shell

    # Distance to the shell, measured outwards and inwards, in world units.
    outside_distance = distance_transform_edt(~shell, sampling=grid.spacing)
    distance = outside_distance.astype(np.float32)

    field = np.where(solid, -distance, distance).astype(np.float32)

    if refine:
        field = _refine_narrow_band(field, grid, points, normals, solid, shell, spacing)

    return field


def _refine_narrow_band(
    field: np.ndarray,
    grid: Grid,
    points: np.ndarray,
    normals: np.ndarray,
    solid: np.ndarray,
    shell: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Recompute distance and sign near the surface against the true surface samples.

    Two separate corrections, because the EDT gets two separate things wrong.

    *Distance.* The EDT answers "how far to the nearest marked voxel centre", quantised
    to about half a voxel, which shows as terracing on a curved boundary. A KD-tree
    query against the real surface samples replaces it.

    *Sign.* A shell voxel is one that merely *contains* a piece of surface, so its
    centre may sit on either side — measurement shows about half of them fall outside
    the solid. Marking them all solid wraps the part in a one-voxel skin, inflating
    volume by 4-10 % and putting the boundary in the wrong place by up to three voxels.
    So shell voxels are signed by which side of the surface they are on, using the
    normal of the nearest sample. Voxels away from the shell keep the flood-fill sign,
    which is unambiguous there and immune to the normal test's weakness at sharp edges.
    """
    from scipy.spatial import cKDTree

    band = np.abs(field) <= NARROW_BAND_VOXELS * spacing
    if not band.any():
        return field

    indices = np.nonzero(band)
    coords = np.stack(
        [
            grid.origin[axis] + indices[axis] * grid.spacing[axis]
            for axis in range(3)
        ],
        axis=1,
    )

    tree = cKDTree(points)
    distance, nearest = tree.query(coords, k=1, workers=-1)

    sign = np.where(solid[indices], -1.0, 1.0)

    # Where the flood fill cannot decide — voxels straddling the surface — ask the
    # geometry instead: which side of the nearest triangle's plane is this centre on?
    straddling = shell[indices]
    if straddling.any():
        offset = coords[straddling] - points[nearest[straddling]]
        projection = np.einsum("ij,ij->i", offset, normals[nearest[straddling]])
        # A zero projection means the centre lies exactly on the surface; treat it as
        # solid so the boundary stays closed. This biases solid for flat faces that
        # happen to land on voxel centres — an axis-aligned box gains up to half a
        # voxel of skin, about 2 % of its volume at 110³. Closing the boundary is worth
        # more than the bias: the alternative opens a pinhole that the flood fill
        # escapes through, which loses the whole interior rather than a thin shell.
        sign[straddling] = np.where(projection > 0.0, 1.0, -1.0)

    field[indices] = (distance * sign).astype(np.float32)

    return field


@dataclass
class DomainField:
    """A sampled SDF plus the grid it lives on, sampled at arbitrary points.

    The pipeline builds this once at a modest resolution and reads it at full marching
    resolution. Grading laws receive it as ``GradingContext.domain_distance``, which is
    how ``surface_distance`` grading reads the skin depth without importing this module.
    """

    grid: Grid
    values: np.ndarray

    def __call__(self, x, y, z) -> np.ndarray:
        return sample_trilinear(self.grid, self.values, x, y, z)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.grid.lower, self.grid.upper

    def solid_volume(self) -> float:
        """Volume enclosed by the zero level, from the voxel count."""
        voxel = float(np.prod(self.grid.spacing))
        return float((self.values < 0).sum()) * voxel


def build_domain_field(
    mesh: Mesh,
    resolution: int = 192,
    padding_voxels: float = 3.0,
    refine: bool = True,
    strict: bool = True,
) -> DomainField:
    """Voxelise ``mesh`` into a :class:`DomainField`.

    ``padding_voxels`` of empty space is added on every side. The flood fill needs a
    connected exterior to start from, so a mesh flush against the grid wall would have
    nowhere to leak from and would come out inverted.
    """
    lower, upper = mesh.bounds
    extent = upper - lower

    if np.any(extent <= 0):
        raise ValueError("the mesh is flat or empty; nothing to voxelise")

    # Estimate the voxel size before the grid exists, to size the padding in world units.
    approximate_spacing = float(extent.max()) / max(resolution - 1, 1)
    padding = approximate_spacing * float(padding_voxels)

    grid = Grid.from_bounds(lower, upper, resolution, padding=padding)
    values = mesh_to_sdf(mesh, grid, refine=refine, strict=strict)

    return DomainField(grid=grid, values=values)
