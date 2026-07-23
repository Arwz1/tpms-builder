"""Generation feature — orchestrates domain, lattice, clipping and meshing.

Standalone: needs numpy, scipy and scikit-image. No Qt.

    >>> from tpms.features.generate import GenerationParams, generate
    >>> p = GenerationParams(shape_name="box", resolution=48)
    >>> p.shape_params = {"width": 30.0, "depth": 30.0, "height": 30.0}
    >>> result = generate(p)
    >>> result.mesh.num_faces > 1000
    True
"""

from tpms.features.generate.params import GenerationParams, SourceType
from tpms.features.generate.pipeline import (
    DomainSource,
    GenerationResult,
    build_domain,
    generate,
    make_field_function,
    preview_params,
)

__all__ = [
    "GenerationParams",
    "SourceType",
    "DomainSource",
    "GenerationResult",
    "build_domain",
    "generate",
    "make_field_function",
    "preview_params",
]
