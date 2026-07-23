"""Regular sampling grid over an axis-aligned box, with slab iteration.

The grid is the shared coordinate system for the whole pipeline. Fields are never
materialised at full size: callers walk :meth:`Grid.slabs` and evaluate one slab at a
time, which turns field memory from O(n^3) into O(n^2).

A 512^3 float32 field is 537 MB. The same field walked in 32-deep slabs holds 34 MB at
a time, and in the default 16-deep slabs, 17 MB.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Iterator, Sequence

import numpy as np

# A hard ceiling. 512^3 is already ~1 GB of output mesh for a fine lattice; beyond that
# the triangle count, not the field, is what breaks the process.
MAX_RESOLUTION = 512

# Above this the quality panel warns the user before running.
WARN_RESOLUTION = 384


@dataclass(frozen=True)
class Slab:
    """A contiguous run of z-planes within a :class:`Grid`.

    Slabs overlap by exactly one plane so that marching cubes reproduces the same
    surface across the seam and the duplicate vertices weld cleanly.
    """

    index: int
    z_start: int
    z_stop: int          # exclusive
    total_slabs: int

    @property
    def depth(self) -> int:
        return self.z_stop - self.z_start

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.total_slabs - 1


@dataclass(frozen=True)
class Grid:
    """Uniform grid of sample points over ``[origin, origin + (shape-1)*spacing]``.

    Parameters
    ----------
    origin:
        World coordinate of sample ``(0, 0, 0)``.
    spacing:
        Distance between adjacent samples on each axis. Usually isotropic.
    shape:
        Number of samples on each axis.
    """

    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, int, int]

    # Cached coordinate vectors, built in __post_init__.
    _axes: tuple[np.ndarray, np.ndarray, np.ndarray] = _field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        spacing = np.asarray(self.spacing, dtype=np.float64).reshape(3)
        shape = tuple(int(s) for s in self.shape)

        if any(s < 2 for s in shape):
            raise ValueError(f"grid needs at least 2 samples per axis, got {shape}")
        if np.any(spacing <= 0):
            raise ValueError(f"grid spacing must be positive, got {spacing}")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", shape)

        axes = tuple(
            origin[i] + np.arange(shape[i], dtype=np.float64) * spacing[i]
            for i in range(3)
        )
        object.__setattr__(self, "_axes", axes)

    # ------------------------------------------------------------------ builders

    @classmethod
    def from_bounds(
        cls,
        lower: Sequence[float],
        upper: Sequence[float],
        resolution: int,
        padding: float = 0.0,
    ) -> "Grid":
        """Build an isotropic grid covering a box.

        ``resolution`` is the sample count along the *longest* axis; shorter axes get
        proportionally fewer samples so the voxels stay cubic. Cubic voxels matter
        because the lattice is isotropic and anisotropic sampling would make walls
        thinner in one direction than another.

        ``padding`` expands the box on every side, in world units. A little padding
        keeps the isosurface from being clipped flat against the domain wall.
        """
        lower = np.asarray(lower, dtype=np.float64).reshape(3) - padding
        upper = np.asarray(upper, dtype=np.float64).reshape(3) + padding

        extent = upper - lower
        if np.any(extent <= 0):
            raise ValueError(f"degenerate bounds: {lower} .. {upper}")

        resolution = int(np.clip(resolution, 8, MAX_RESOLUTION))

        longest = float(extent.max())
        step = longest / (resolution - 1)

        shape = tuple(max(2, int(np.ceil(e / step)) + 1) for e in extent)
        spacing = np.full(3, step, dtype=np.float64)

        return cls(origin=lower, spacing=spacing, shape=shape)

    def with_resolution(self, resolution: int) -> "Grid":
        """Return a grid over the same box at a different resolution."""
        return Grid.from_bounds(self.lower, self.upper, resolution)

    # ---------------------------------------------------------------- properties

    @property
    def lower(self) -> np.ndarray:
        return self.origin.copy()

    @property
    def upper(self) -> np.ndarray:
        return self.origin + (np.asarray(self.shape) - 1) * self.spacing

    @property
    def extent(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def num_points(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def voxel_diagonal(self) -> float:
        """Length of a voxel's body diagonal — the resolution limit of any feature."""
        return float(np.linalg.norm(self.spacing))

    @property
    def x(self) -> np.ndarray:
        return self._axes[0]

    @property
    def y(self) -> np.ndarray:
        return self._axes[1]

    @property
    def z(self) -> np.ndarray:
        return self._axes[2]

    # ------------------------------------------------------------------- sampling

    def meshgrid(self, dtype: type = np.float32) -> tuple[np.ndarray, ...]:
        """Full-size coordinate arrays. Only for small grids — allocates 3 * n^3."""
        return np.meshgrid(
            self.x.astype(dtype), self.y.astype(dtype), self.z.astype(dtype),
            indexing="ij",
        )

    def slab_meshgrid(
        self, slab: Slab, dtype: type = np.float32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coordinate arrays for one slab, shaped ``(nx, ny, slab.depth)``."""
        zs = self.z[slab.z_start : slab.z_stop]
        return np.meshgrid(
            self.x.astype(dtype), self.y.astype(dtype), zs.astype(dtype),
            indexing="ij",
        )

    def slabs(self, depth: int = 16) -> Iterator[Slab]:
        """Walk the grid in overlapping z-slabs.

        Consecutive slabs share one plane. Without that shared plane marching cubes
        would leave a one-voxel gap at every seam.
        """
        nz = self.shape[2]
        depth = max(2, int(depth))

        if depth >= nz:
            yield Slab(index=0, z_start=0, z_stop=nz, total_slabs=1)
            return

        # Each slab advances by (depth - 1) planes and re-reads the last one.
        stride = depth - 1
        starts = list(range(0, nz - 1, stride))
        total = len(starts)

        for i, z0 in enumerate(starts):
            z1 = min(z0 + depth, nz)
            yield Slab(index=i, z_start=z0, z_stop=z1, total_slabs=total)

    def slab_origin(self, slab: Slab) -> np.ndarray:
        """World coordinate of the slab's ``(0, 0, 0)`` sample."""
        return self.origin + np.array([0.0, 0.0, slab.z_start * self.spacing[2]])

    # ------------------------------------------------------------------ estimates

    def field_bytes(self, slab_depth: int = 16, dtype: type = np.float32) -> int:
        """Resident bytes for one slab of a scalar field."""
        nx, ny, nz = self.shape
        depth = min(max(2, slab_depth), nz)
        return nx * ny * depth * np.dtype(dtype).itemsize

    def full_field_bytes(self, dtype: type = np.float32) -> int:
        """Bytes a full-size field *would* take. Reported to justify slab streaming."""
        return self.num_points * np.dtype(dtype).itemsize

    def estimate_triangles(
        self,
        cell_size: float,
        sheet: bool = True,
        solid_extent: Sequence[float] | None = None,
    ) -> int:
        """Triangle-count estimate for a TPMS fill of this grid.

        A unit cell meshed at this grid's resolution contributes about
        ``9 * (cell_size / voxel)^2`` triangles per surface sheet. The constant is
        fitted to measured gyroid runs and holds to within a few percent from 96³ to
        512³ — at a 60 mm box with 8 mm cells it predicts 947 k / 6.8 M / 15.3 M
        triangles against measured 969 k / 6.8 M / 15.7 M at 96³ / 256³ / 384³.

        ``solid_extent`` is the size of the part being filled. It matters because the
        grid is padded by half a cell on every side, and counting cells across the
        padding rather than the part over-predicts by 45 % on a 60 mm box — the lattice
        is clipped to the part and never occupies the padding.
        """
        if cell_size <= 0:
            return 0

        extent = np.asarray(
            solid_extent if solid_extent is not None else self.extent, dtype=np.float64
        )

        voxels_per_cell = cell_size / float(self.spacing.mean())
        cells = float(np.prod(np.maximum(extent, 0.0) / cell_size))
        per_cell = 9.0 * max(voxels_per_cell, 1.0) ** 2
        if sheet:
            per_cell *= 2.0  # a sheet presents two faces to the mesher
        return int(max(0.0, cells * per_cell))

    def estimate_peak_bytes(self, triangles: int) -> int:
        """Peak process memory for a run producing ``triangles`` triangles.

        A straight line fitted to measured peak RSS, because reasoning from the stored
        size alone under-predicts by more than half — the transient copies during
        concatenate, weld and component removal cost more than the final arrays.

        ==========  ==========  ==========
        triangles   measured    predicted
        ==========  ==========  ==========
        1.7 M         339 MB      338 MB
        6.9 M        1154 MB     1150 MB
        15.7 M       2511 MB     2515 MB
        27.7 M       4392 MB     4393 MB
        ==========  ==========  ==========
        """
        return int(triangles * 156.0 + 70e6)

    def describe(self) -> str:
        nx, ny, nz = self.shape
        return (
            f"{nx}x{ny}x{nz} samples, {self.spacing[0]:.4g} mm voxels, "
            f"{self.extent[0]:.4g} x {self.extent[1]:.4g} x {self.extent[2]:.4g} mm"
        )


def sample_trilinear(
    grid: Grid,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    outside: float | None = None,
) -> np.ndarray:
    """Trilinearly sample ``values`` (defined on ``grid``) at arbitrary points.

    This is what decouples the domain SDF resolution from the lattice resolution: the
    SDF is built once on a coarse grid and read here at full marching resolution. The
    part boundary is smooth and low-frequency, so interpolating it costs nothing
    visible while saving the cube of the resolution ratio in memory and time.

    Points outside the grid are clamped to the edge, or set to ``outside`` when given.
    Clamping is the right default for an SDF: past the domain box the field simply
    continues to grow outwards.
    """
    if values.shape != grid.shape:
        raise ValueError(
            f"values shape {values.shape} does not match grid shape {grid.shape}"
        )

    shape = np.asarray(grid.shape)

    # Continuous index coordinates.
    fx = (x - grid.origin[0]) / grid.spacing[0]
    fy = (y - grid.origin[1]) / grid.spacing[1]
    fz = (z - grid.origin[2]) / grid.spacing[2]

    if outside is not None:
        beyond = (
            (fx < 0) | (fx > shape[0] - 1)
            | (fy < 0) | (fy > shape[1] - 1)
            | (fz < 0) | (fz > shape[2] - 1)
        )

    fx = np.clip(fx, 0.0, shape[0] - 1.0)
    fy = np.clip(fy, 0.0, shape[1] - 1.0)
    fz = np.clip(fz, 0.0, shape[2] - 1.0)

    ix = np.clip(np.floor(fx).astype(np.intp), 0, shape[0] - 2)
    iy = np.clip(np.floor(fy).astype(np.intp), 0, shape[1] - 2)
    iz = np.clip(np.floor(fz).astype(np.intp), 0, shape[2] - 2)

    tx = (fx - ix).astype(values.dtype, copy=False)
    ty = (fy - iy).astype(values.dtype, copy=False)
    tz = (fz - iz).astype(values.dtype, copy=False)

    c000 = values[ix, iy, iz]
    c100 = values[ix + 1, iy, iz]
    c010 = values[ix, iy + 1, iz]
    c110 = values[ix + 1, iy + 1, iz]
    c001 = values[ix, iy, iz + 1]
    c101 = values[ix + 1, iy, iz + 1]
    c011 = values[ix, iy + 1, iz + 1]
    c111 = values[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    result = c0 * (1 - tz) + c1 * tz

    if outside is not None:
        result = np.where(beyond, np.asarray(outside, dtype=result.dtype), result)

    return result
