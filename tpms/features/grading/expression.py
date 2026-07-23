"""Grading driven by user-typed maths.

The escape hatch: anything the fixed laws cannot express, typed as ``f(x, y, z)``.
Expressions are compiled through :mod:`tpms.core.expressions`, which gates them against
an AST allowlist — the string comes from a text box, so it never reaches bare ``eval``.

This is also the seam step 2 will use. A stress field solved on the lattice can be fed
straight back in as a thickness expression, turning "thicken where it is loaded" into a
parameter change rather than new plumbing.

Available variables:

======  ====================================================
``x``   world X                ``u``  normalised X, 0..1
``y``   world Y                ``v``  normalised Y, 0..1
``z``   world Z                ``w``  normalised Z, 0..1
``r``   distance from the domain centre
``d``   distance to the part surface, negative inside
======  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tpms.core.expressions import (
    ExpressionError,
    compile_expression,
    validate_expression,
)
from tpms.features.grading.base import (
    MIN_CELL_SIZE,
    Grading,
    GradingContext,
    register_grading,
)

EXPRESSION_VARIABLES = ("x", "y", "z", "r", "d", "u", "v", "w")


@register_grading
@dataclass
class ExpressionGrading(Grading):
    name = "expression"
    label = "Expression f(x, y, z)"
    description = (
        "Cell size and wall thickness from typed maths. Variables: x y z (world), "
        "u v w (0..1 across the box), r (from centre), d (to surface)."
    )

    cell_expression: str = "6"
    wall_expression: str = "1"

    _cell_fn: Callable[..., np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _wall_fn: Callable[..., np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _compiled_for: tuple[str, str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _compile(self) -> None:
        """Compile both expressions, reusing the result while the source is unchanged.

        The pipeline calls this once per slab; recompiling each time would parse the
        same string dozens of times per run.
        """
        key = (self.cell_expression, self.wall_expression)
        if self._compiled_for == key and self._cell_fn is not None:
            return

        self._cell_fn = compile_expression(self.cell_expression, EXPRESSION_VARIABLES)
        self._wall_fn = compile_expression(self.wall_expression, EXPRESSION_VARIABLES)
        self._compiled_for = key

    def _variables(self, x, y, z, ctx: GradingContext) -> dict[str, Any]:
        u, v, w = ctx.normalised(x, y, z)

        centre = ctx.centre
        dx = x - centre[0]
        dy = y - centre[1]
        dz = z - centre[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)

        if ctx.domain_distance is not None:
            d = np.asarray(ctx.domain_distance(x, y, z), dtype=np.float32)
        else:
            d = np.zeros(np.shape(x), dtype=np.float32)

        return {"x": x, "y": y, "z": z, "r": r, "d": d, "u": u, "v": v, "w": w}

    def cell_size(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        self._compile()
        value = self._cell_fn(**self._variables(x, y, z, ctx))
        value = np.broadcast_to(value, np.shape(x)).astype(np.float32, copy=True)

        # An expression can produce anything at all. A non-positive or non-finite cell
        # size would send the phase to infinity and fill the slab with NaN, so it is
        # clamped here rather than being allowed to poison the mesh.
        return np.nan_to_num(
            value, nan=MIN_CELL_SIZE, posinf=1e6, neginf=MIN_CELL_SIZE
        ).clip(MIN_CELL_SIZE, 1e6)

    def thickness(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        self._compile()
        value = self._wall_fn(**self._variables(x, y, z, ctx))
        value = np.broadcast_to(value, np.shape(x)).astype(np.float32, copy=True)
        return np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=0.0).clip(0.0, 1e6)

    def validate(self) -> str | None:
        """Return an error message, or ``None`` when both expressions are usable.

        Compiling is not enough to know an expression works: ``1/0`` and ``log(0)``
        parse and compile perfectly and only fail when evaluated. Both are probed on a
        sample point, so the UI rejects them as they are typed rather than at
        generation time.
        """
        for label, source in (
            ("cell size", self.cell_expression),
            ("wall thickness", self.wall_expression),
        ):
            message = validate_expression(source, EXPRESSION_VARIABLES)
            if message is not None:
                return f"{label}: {message}"
        return None
