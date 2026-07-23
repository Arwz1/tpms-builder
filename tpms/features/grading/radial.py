"""Cell size and thickness graded with distance from a point or an axis.

Spherical mode grades from a centre point outwards — dense core, open shell, or the
reverse. Cylindrical mode grades from a line, which is what a shaft boss or a pressure
vessel wall wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpms.features.grading.base import (
    MIN_CELL_SIZE,
    Grading,
    GradingContext,
    register_grading,
)


@register_grading
@dataclass
class RadialGrading(Grading):
    name = "radial"
    label = "Radial (from a point or axis)"
    description = (
        "Cell size and wall thickness interpolate with distance from a centre point "
        "(spherical) or a centre line (cylindrical)."
    )

    #: 0, 1, 2 to measure distance from the X, Y or Z axis; None for spherical.
    axis: int | None = None
    #: Centre in world units. None uses the domain centre.
    centre: tuple[float, float, float] | None = None
    radius_inner: float = 0.0
    radius_outer: float = 50.0
    cell_inner: float = 4.0
    cell_outer: float = 10.0
    wall_inner: float = 1.0
    wall_outer: float = 1.0

    def _radius(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        centre = np.asarray(
            self.centre if self.centre is not None else ctx.centre, dtype=np.float32
        )

        dx = x - centre[0]
        dy = y - centre[1]
        dz = z - centre[2]

        if self.axis is None:
            return np.sqrt(dx * dx + dy * dy + dz * dz)

        # Distance from a line: drop the component along the axis.
        components = [dx, dy, dz]
        components.pop(int(self.axis))
        a, b = components
        return np.sqrt(a * a + b * b)

    def _fraction(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        r = self._radius(x, y, z, ctx)
        inner = float(self.radius_inner)
        outer = float(self.radius_outer)
        span = outer - inner
        if abs(span) < 1e-9:
            return np.zeros_like(r, dtype=np.float32)
        return np.clip((r - inner) / span, 0.0, 1.0).astype(np.float32, copy=False)

    def cell_size(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        t = self._fraction(x, y, z, ctx)
        a0 = max(float(self.cell_inner), MIN_CELL_SIZE)
        a1 = max(float(self.cell_outer), MIN_CELL_SIZE)
        return a0 + (a1 - a0) * t

    def thickness(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        t = self._fraction(x, y, z, ctx)
        return float(self.wall_inner) + (float(self.wall_outer) - float(self.wall_inner)) * t

    def default_radius_outer(self, ctx: GradingContext) -> float:
        """Half the domain diagonal — a sensible default when the UI first opens."""
        return float(np.linalg.norm(ctx.extent) * 0.5)
