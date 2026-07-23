"""Constructive solid geometry on sampled signed-distance fields.

Convention throughout the project: **negative is inside**, zero is the surface.

Every function is elementwise and shape-agnostic, so the same calls work on a full
field, a single slab, or a scalar.

A note on exactness: ``min``/``max`` composition is exact on the *surface* but only
bounds the distance away from it. That is all the pipeline needs, because marching cubes
reads the zero crossing and nothing else. The smooth variants trade even that for a
blended join.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | float


# --------------------------------------------------------------------- booleans

def union(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Everything inside ``a`` or inside ``b``."""
    return np.minimum(a, b)


def intersect(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Only what is inside both — the operation that clips a lattice to a part."""
    return np.maximum(a, b)


def subtract(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Inside ``a`` but not inside ``b``."""
    return np.maximum(a, -np.asarray(b))


def invert(a: ArrayLike) -> np.ndarray:
    """Swap inside and outside."""
    return -np.asarray(a)


def union_all(fields: list[ArrayLike]) -> np.ndarray:
    if not fields:
        raise ValueError("union_all needs at least one field")
    out = np.asarray(fields[0])
    for f in fields[1:]:
        out = np.minimum(out, f)
    return out


def intersect_all(fields: list[ArrayLike]) -> np.ndarray:
    if not fields:
        raise ValueError("intersect_all needs at least one field")
    out = np.asarray(fields[0])
    for f in fields[1:]:
        out = np.maximum(out, f)
    return out


# ------------------------------------------------------------- smooth booleans

def smooth_union(a: ArrayLike, b: ArrayLike, k: float) -> np.ndarray:
    """Polynomial smooth minimum — blends the join over a width of ``k``.

    Used to fillet the lattice into the solid skin so the junction is not a stress
    riser. With ``k <= 0`` this is a plain union.
    """
    if k <= 0:
        return union(a, b)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1.0 - h)


def smooth_intersect(a: ArrayLike, b: ArrayLike, k: float) -> np.ndarray:
    if k <= 0:
        return intersect(a, b)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h + k * h * (1.0 - h)


# ---------------------------------------------------------------- modifications

def offset(a: ArrayLike, distance: float) -> np.ndarray:
    """Grow (positive) or shrink (negative) the solid by ``distance``."""
    return np.asarray(a) - distance


def shell(a: ArrayLike, thickness: float) -> np.ndarray:
    """Hollow the solid into a wall of ``thickness``, centred on the surface.

    This is exactly the sheet-TPMS operation, and the reason both live on the same
    convention: ``|d| - t/2`` is negative only within ``t/2`` of the zero level.
    """
    return np.abs(np.asarray(a)) - thickness * 0.5


def inner_shell(a: ArrayLike, thickness: float) -> np.ndarray:
    """A wall of ``thickness`` lying entirely *inside* the surface.

    What you want for a solid skin on a printed part: the outer dimension stays exactly
    as modelled and the wall grows inwards.
    """
    a = np.asarray(a)
    return np.maximum(a, -(a + thickness))


def clamp_field(a: ArrayLike, limit: float) -> np.ndarray:
    """Clamp magnitudes to ``limit``.

    Marching cubes only reads the sign change, and clamping keeps the far field from
    dominating a float32 range when several fields are combined.
    """
    return np.clip(np.asarray(a), -limit, limit)


# ------------------------------------------------------------------- primitives
# Analytic SDFs used by base shapes and by tests. All exact.

def sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float,
           centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    cx, cy, cz = centre
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius


def box(x: np.ndarray, y: np.ndarray, z: np.ndarray,
        half_extents: tuple[float, float, float],
        centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Exact box SDF, correct both inside and outside."""
    hx, hy, hz = half_extents
    cx, cy, cz = centre

    dx = np.abs(x - cx) - hx
    dy = np.abs(y - cy) - hy
    dz = np.abs(z - cz) - hz

    outside = np.sqrt(
        np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2
    )
    inside = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0.0)
    return outside + inside


def cylinder(x: np.ndarray, y: np.ndarray, z: np.ndarray,
             radius: float, height: float, axis: int = 2,
             centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Capped cylinder along ``axis`` (0=X, 1=Y, 2=Z)."""
    coords = [x - centre[0], y - centre[1], z - centre[2]]
    along = coords.pop(axis)
    a, b = coords

    radial = np.sqrt(a ** 2 + b ** 2) - radius
    axial = np.abs(along) - height * 0.5

    outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside


def torus(x: np.ndarray, y: np.ndarray, z: np.ndarray,
          major_radius: float, minor_radius: float, axis: int = 2,
          centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    coords = [x - centre[0], y - centre[1], z - centre[2]]
    along = coords.pop(axis)
    a, b = coords

    q = np.sqrt(a ** 2 + b ** 2) - major_radius
    return np.sqrt(q ** 2 + along ** 2) - minor_radius


def cone(x: np.ndarray, y: np.ndarray, z: np.ndarray,
         radius_bottom: float, radius_top: float, height: float, axis: int = 2,
         centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Capped cone / truncated cone. ``radius_top`` of 0 gives a point."""
    coords = [x - centre[0], y - centre[1], z - centre[2]]
    along = coords.pop(axis)
    a, b = coords

    r = np.sqrt(a ** 2 + b ** 2)
    half_h = height * 0.5

    # Interpolate the radius at this height, then treat it as a slanted plane.
    t = np.clip((along + half_h) / height, 0.0, 1.0)
    radius_at = radius_bottom + (radius_top - radius_bottom) * t

    slope = (radius_top - radius_bottom) / height
    normalise = 1.0 / np.sqrt(1.0 + slope ** 2)

    radial = (r - radius_at) * normalise
    axial = np.abs(along) - half_h

    outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside


def capsule(x: np.ndarray, y: np.ndarray, z: np.ndarray,
            radius: float, height: float, axis: int = 2,
            centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Cylinder with hemispherical caps. ``height`` is the cylindrical run."""
    coords = [x - centre[0], y - centre[1], z - centre[2]]
    along = coords.pop(axis)
    a, b = coords

    clamped = np.clip(along, -height * 0.5, height * 0.5)
    return np.sqrt(a ** 2 + b ** 2 + (along - clamped) ** 2) - radius


def plane(x: np.ndarray, y: np.ndarray, z: np.ndarray,
          normal: tuple[float, float, float], offset_distance: float = 0.0) -> np.ndarray:
    """Half-space. Negative on the side the normal points away from."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return x * n[0] + y * n[1] + z * n[2] - offset_distance
