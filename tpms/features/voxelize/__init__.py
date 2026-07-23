"""Voxelisation feature — imported mesh to signed distance field.

Standalone: needs numpy and scipy.

    >>> from tpms.core import Grid, sdf, march_field
    >>> from tpms.features.voxelize import build_domain_field
    >>> g = Grid.from_bounds([-12] * 3, [12] * 3, 48)
    >>> X, Y, Z = g.meshgrid()
    >>> sphere = march_field(g, sdf.sphere(X, Y, Z, 8.0))
    >>> domain = build_domain_field(sphere, resolution=64)
    >>> round(domain.solid_volume() / (4 / 3 * 3.14159 * 8 ** 3), 2)
    1.0
"""

from tpms.features.voxelize.mesh_sdf import (
    DomainField,
    NotWatertightError,
    build_domain_field,
    mesh_to_sdf,
    surface_samples,
)

__all__ = [
    "DomainField",
    "NotWatertightError",
    "build_domain_field",
    "mesh_to_sdf",
    "surface_samples",
]
