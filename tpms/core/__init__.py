"""Core contracts shared by every feature.

Nothing in this package imports from elsewhere in the project — only numpy and scipy.
That constraint is what keeps the features standalone.
"""

from tpms.core.grid import Grid, Slab
from tpms.core.mesh import Mesh
from tpms.core import sdf
from tpms.core.marching import march_field, march_slabs
from tpms.core.expressions import ExpressionError, compile_expression, evaluate

__all__ = [
    "Grid",
    "Slab",
    "Mesh",
    "sdf",
    "march_field",
    "march_slabs",
    "ExpressionError",
    "compile_expression",
    "evaluate",
]
