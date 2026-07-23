"""Cell size and thickness graded linearly along one axis.

The classic functional grade: fine at one end, coarse at the other. This is the one law
whose phase integral has a closed form, so it is exact — no cell distortion at all,
however steep the grade.

With a linear cell size ``a(t) = a₀ + (a₁ - a₀)t`` across a span of length ``L``:

    u(t) = ∫₀ᵗ 2πL/a(s) ds = (2πL / (a₁ - a₀)) · ln(a(t) / a₀)

falling back to ``2πLt/a₀`` when the two ends are equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpms.features.grading.base import (
    MIN_CELL_SIZE,
    TWO_PI,
    Grading,
    GradingContext,
    register_grading,
)

AXIS_NAMES = ("X", "Y", "Z")


@register_grading
@dataclass
class AxialGrading(Grading):
    name = "axial"
    label = "Axial (along X / Y / Z)"
    description = (
        "Cell size and wall thickness interpolate linearly along one axis. "
        "Phase is integrated exactly, so cells stay undistorted at any grade."
    )

    axis: int = 2
    cell_start: float = 4.0
    cell_end: float = 10.0
    wall_start: float = 1.0
    wall_end: float = 1.0

    def _fraction(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        """Position along the graded axis as 0..1."""
        coord = (x, y, z)[self.axis]
        lower = ctx.lower[self.axis]
        span = ctx.extent[self.axis]
        return np.clip((coord - lower) / span, 0.0, 1.0).astype(np.float32, copy=False)

    def cell_size(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        t = self._fraction(x, y, z, ctx)
        a0 = max(float(self.cell_start), MIN_CELL_SIZE)
        a1 = max(float(self.cell_end), MIN_CELL_SIZE)
        return a0 + (a1 - a0) * t

    def thickness(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        t = self._fraction(x, y, z, ctx)
        return float(self.wall_start) + (float(self.wall_end) - float(self.wall_start)) * t

    def phase(self, x, y, z, ctx: GradingContext):
        """Exact phase along the graded axis, local scaling across the other two."""
        axis = int(self.axis)
        a0 = max(float(self.cell_start), MIN_CELL_SIZE)
        a1 = max(float(self.cell_end), MIN_CELL_SIZE)
        span = float(ctx.extent[axis])
        lower = float(ctx.lower[axis])

        coord = (x, y, z)[axis]
        t = np.clip((coord - lower) / span, 0.0, 1.0).astype(np.float32, copy=False)

        if abs(a1 - a0) < 1e-9:
            along = np.float32(TWO_PI * span / a0) * t
        else:
            local = a0 + (a1 - a0) * t
            along = np.float32(TWO_PI * span / (a1 - a0)) * np.log(
                local / np.float32(a0)
            )

        # The other two axes are scaled by the local cell size. The cell size does not
        # vary along them, so there is nothing to integrate — but it does vary *across*
        # them via the graded axis, which is the residual shear inherent to any graded
        # lattice.
        cell = np.maximum(
            np.asarray(self.cell_size(x, y, z, ctx), dtype=np.float32), MIN_CELL_SIZE
        )
        k = np.float32(TWO_PI) / cell

        others = [c * k for i, c in enumerate((x, y, z)) if i != axis]
        result = [None, None, None]
        result[axis] = along.astype(np.float32, copy=False)
        for i in (i for i in range(3) if i != axis):
            result[i] = others.pop(0).astype(np.float32, copy=False)

        return tuple(result)
