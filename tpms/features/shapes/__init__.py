"""Base-shape feature — analytic solids to fill without importing anything.

Standalone: needs only numpy.

    >>> import numpy as np
    >>> from tpms.features.shapes import get_shape
    >>> s = get_shape("cylinder", radius=10.0, height=20.0)
    >>> float(s.sdf(np.float64(0), np.float64(0), np.float64(0)))
    -10.0
    >>> lower, upper = s.bounds()
    >>> upper.tolist()
    [10.0, 10.0, 10.0]
"""

from tpms.features.shapes.base import (
    BaseShape,
    get_shape,
    list_shapes,
    register_shape,
    shape_class,
    shape_names,
)
from tpms.features.shapes.primitives import (
    BoxShape,
    CapsuleShape,
    ConeShape,
    CylinderShape,
    SphereShape,
    TorusShape,
)

DEFAULT_SHAPE = "box"

__all__ = [
    "BaseShape",
    "get_shape",
    "list_shapes",
    "register_shape",
    "shape_class",
    "shape_names",
    "BoxShape",
    "SphereShape",
    "CylinderShape",
    "ConeShape",
    "TorusShape",
    "CapsuleShape",
    "DEFAULT_SHAPE",
]
