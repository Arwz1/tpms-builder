"""Grading driven by distance to the part's own surface.

Fine cells near the skin where bending stress peaks, coarse cells in the core where they
carry little — the standard weight-saving grade, and the one that needs no manual setup
because the driving field is the domain SDF the pipeline already built.

Falls back to uniform behaviour when no domain distance is available (for instance while
previewing a pattern before any geometry is loaded), rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpms.features.grading.base import (
    MIN_CELL_SIZE,
    Grading,
    GradingContext,
    register_grading,
)


@register_grading
@dataclass
class SurfaceDistanceGrading(Grading):
    name = "surface_distance"
    label = "Distance to surface"
    description = (
        "Cell size and wall thickness interpolate with depth below the part surface. "
        "Dense skin, open core — the usual way to take weight out of a part."
    )

    #: Depth below the surface, in world units, at which the outer values are reached.
    depth: float = 10.0
    cell_surface: float = 4.0
    cell_core: float = 10.0
    wall_surface: float = 1.2
    wall_core: float = 0.8

    def _fraction(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        """0 at the skin, 1 at ``depth`` below it and beyond."""
        if ctx.domain_distance is None:
            return np.zeros(np.shape(x), dtype=np.float32)

        # Domain SDF is negative inside, so depth below the surface is -distance.
        below = -np.asarray(ctx.domain_distance(x, y, z), dtype=np.float32)

        span = max(float(self.depth), 1e-6)
        return np.clip(below / span, 0.0, 1.0).astype(np.float32, copy=False)

    def cell_size(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        t = self._fraction(x, y, z, ctx)
        a0 = max(float(self.cell_surface), MIN_CELL_SIZE)
        a1 = max(float(self.cell_core), MIN_CELL_SIZE)
        return a0 + (a1 - a0) * t

    def thickness(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        t = self._fraction(x, y, z, ctx)
        return float(self.wall_surface) + (
            float(self.wall_core) - float(self.wall_surface)
        ) * t
