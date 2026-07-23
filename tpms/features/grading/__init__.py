"""Grading feature — spatially varying cell size and wall thickness.

Standalone: needs only numpy.

    >>> import numpy as np
    >>> from tpms.features.grading import get_grading, GradingContext
    >>> ctx = GradingContext(lower=[0, 0, 0], upper=[60, 60, 60])
    >>> g = get_grading("axial", axis=2, cell_start=3.0, cell_end=9.0)
    >>> z = np.linspace(0, 60, 5, dtype=np.float32)
    >>> np.round(g.cell_size(z, z, z, ctx), 2)
    array([3. , 4.5, 6. , 7.5, 9. ], dtype=float32)
"""

from tpms.features.grading.base import (
    MIN_CELL_SIZE,
    Grading,
    GradingContext,
    get_grading,
    grading_class,
    grading_names,
    list_gradings,
    register_grading,
)

# Imported for the registration side effect. Order sets the order in the UI dropdown.
from tpms.features.grading.uniform import UniformGrading
from tpms.features.grading.axial import AxialGrading
from tpms.features.grading.radial import RadialGrading
from tpms.features.grading.surface_distance import SurfaceDistanceGrading
from tpms.features.grading.expression import ExpressionGrading

DEFAULT_GRADING = "uniform"

__all__ = [
    "MIN_CELL_SIZE",
    "Grading",
    "GradingContext",
    "get_grading",
    "grading_class",
    "grading_names",
    "list_gradings",
    "register_grading",
    "UniformGrading",
    "AxialGrading",
    "RadialGrading",
    "SurfaceDistanceGrading",
    "ExpressionGrading",
    "DEFAULT_GRADING",
]
