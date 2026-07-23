"""The grading contract and its registry.

Grading makes cell size and wall thickness vary across the part — denser where the load
is, coarser where it is not. That is most of what separates a designed lattice from a
uniform infill.

**Why grading needs more than a scale factor.** The obvious implementation is to divide
world position by a local cell size: ``u = 2π x / a(x)``. That is wrong in a way that
shows up immediately. Phase must advance continuously, and under local scaling it does
not — as ``a`` changes along the axis, ``2π x / a(x)`` jumps, so cells tear and walls
end mid-air. The correct phase is the accumulated integral

    u(x) = ∫₀ˣ 2π / a(s) ds

which is continuous by construction and reduces to the naive form when ``a`` is
constant. :class:`Grading` therefore exposes :meth:`Grading.phase` rather than only a
cell size, so a law that can integrate itself analytically does, and one that cannot
falls back to local scaling with the distortion confined to where the gradient is
gentle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

TWO_PI = 2.0 * np.pi

# Cell sizes below this are treated as degenerate; without a floor an expression that
# dips to zero produces an infinite phase and fills the slab with NaN.
MIN_CELL_SIZE = 1e-4


@dataclass
class GradingContext:
    """Everything a grading law may need about the part it is filling.

    Passed to every call so that no grading module has to import the voxeliser or the
    pipeline — which is what keeps the feature standalone.
    """

    lower: np.ndarray
    upper: np.ndarray
    base_cell_size: float = 6.0
    base_thickness: float = 1.0
    #: ``(x, y, z) -> signed distance to the part surface``, negative inside.
    #: ``None`` when the domain is analytic or not yet known.
    domain_distance: Callable[..., np.ndarray] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64).reshape(3)
        self.upper = np.asarray(self.upper, dtype=np.float64).reshape(3)

    @property
    def extent(self) -> np.ndarray:
        return np.maximum(self.upper - self.lower, 1e-9)

    @property
    def centre(self) -> np.ndarray:
        return (self.lower + self.upper) * 0.5

    def normalised(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position as 0..1 across the bounding box, for expression variables."""
        e = self.extent
        return (
            (x - self.lower[0]) / e[0],
            (y - self.lower[1]) / e[1],
            (z - self.lower[2]) / e[2],
        )


class Grading(ABC):
    """A spatially varying cell size and wall thickness."""

    name: str = ""
    label: str = ""
    description: str = ""

    @abstractmethod
    def cell_size(self, x, y, z, ctx: GradingContext) -> np.ndarray | float:
        """Cell size in world units at each point. Must stay positive."""

    @abstractmethod
    def thickness(self, x, y, z, ctx: GradingContext) -> np.ndarray | float:
        """Wall thickness in world units at each point."""

    # ------------------------------------------------------------------- phase

    def phase(
        self, x, y, z, ctx: GradingContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phase coordinates for the TPMS field.

        Default: local scaling. Exact wherever the cell size is constant, and an
        approximation elsewhere — cells distort slightly but stay connected as long as
        the size varies gently. Subclasses that can integrate exactly override this.
        """
        cell = np.maximum(
            np.asarray(self.cell_size(x, y, z, ctx), dtype=np.float32), MIN_CELL_SIZE
        )
        k = np.float32(TWO_PI) / cell
        return x * k, y * k, z * k

    def phase_scale(self, x, y, z, ctx: GradingContext) -> np.ndarray:
        """``2π / cell_size`` — converts a world length into phase.

        Needed alongside :meth:`phase` so that wall thickness stays a world dimension
        even where the cell size is varying.
        """
        cell = np.maximum(
            np.asarray(self.cell_size(x, y, z, ctx), dtype=np.float32), MIN_CELL_SIZE
        )
        return (np.float32(TWO_PI) / cell).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ params

    def parameters(self) -> dict[str, Any]:
        """Serialisable parameters, for saving a setup and for the CLI."""
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in vars(self).items()
            if not k.startswith("_")
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        inner = ", ".join(f"{k}={v!r}" for k, v in self.parameters().items())
        return f"{type(self).__name__}({inner})"


_GRADINGS: dict[str, type[Grading]] = {}


def register_grading(cls: type[Grading]) -> type[Grading]:
    """Class decorator adding a grading law to the registry."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name")
    _GRADINGS[cls.name] = cls
    return cls


def get_grading(name: str, **params: Any) -> Grading:
    """Construct a grading law by name."""
    try:
        cls = _GRADINGS[name]
    except KeyError:
        raise KeyError(
            f"unknown grading '{name}'. Available: {', '.join(sorted(_GRADINGS))}"
        ) from None
    return cls(**params)


def grading_class(name: str) -> type[Grading]:
    try:
        return _GRADINGS[name]
    except KeyError:
        raise KeyError(
            f"unknown grading '{name}'. Available: {', '.join(sorted(_GRADINGS))}"
        ) from None


def list_gradings() -> tuple[type[Grading], ...]:
    return tuple(_GRADINGS.values())


def grading_names() -> tuple[str, ...]:
    return tuple(_GRADINGS)
