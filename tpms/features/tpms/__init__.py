"""TPMS pattern feature — periodic scalar fields and their solidification.

Standalone: needs only numpy.

    >>> import numpy as np
    >>> from tpms.features.tpms import get_pattern, solidify, SolidMode
    >>> g = get_pattern("gyroid")
    >>> u = np.linspace(0, 2 * np.pi, 8, dtype=np.float32)
    >>> uu, vv, ww = np.meshgrid(u, u, u, indexing="ij")
    >>> field = solidify(g, uu, vv, ww, phase_scale=2 * np.pi / 5.0, thickness=1.0)
    >>> field.shape
    (8, 8, 8)
"""

from tpms.features.tpms.base import (
    MIN_GRADIENT,
    TPMSPattern,
    get_pattern,
    list_patterns,
    pattern_names,
    register_pattern,
)
from tpms.features.tpms.modes import (
    SolidMode,
    estimate_volume_fraction,
    solidify,
    wall_thickness_limits,
)

# Imported for the registration side effect. Order sets the order in the UI dropdown.
from tpms.features.tpms.gyroid import Gyroid
from tpms.features.tpms.schwarz_p import SchwarzP
from tpms.features.tpms.schwarz_d import SchwarzD
from tpms.features.tpms.neovius import Neovius

DEFAULT_PATTERN = "gyroid"

__all__ = [
    "MIN_GRADIENT",
    "TPMSPattern",
    "get_pattern",
    "list_patterns",
    "pattern_names",
    "register_pattern",
    "SolidMode",
    "solidify",
    "estimate_volume_fraction",
    "wall_thickness_limits",
    "Gyroid",
    "SchwarzP",
    "SchwarzD",
    "Neovius",
    "DEFAULT_PATTERN",
]
