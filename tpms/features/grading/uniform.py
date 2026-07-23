"""Constant cell size and wall thickness — the baseline."""

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
class UniformGrading(Grading):
    name = "uniform"
    label = "Uniform"
    description = "One cell size and one wall thickness across the whole part."

    cell: float = 6.0
    wall: float = 1.0

    def cell_size(self, x, y, z, ctx: GradingContext) -> float:
        return max(float(self.cell), MIN_CELL_SIZE)

    def thickness(self, x, y, z, ctx: GradingContext) -> float:
        return float(self.wall)

    def phase(self, x, y, z, ctx: GradingContext):
        """Exact: with a constant cell the integral is just a scale factor."""
        k = np.float32(2.0 * np.pi / max(float(self.cell), MIN_CELL_SIZE))
        return x * k, y * k, z * k
