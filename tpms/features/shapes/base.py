"""The base-shape contract and its registry.

Base shapes let the app produce a TPMS part with no import at all — pick a cylinder,
set a cell size, generate. They are analytic signed distance functions, so unlike an
imported mesh they need no voxelisation and carry no discretisation error: the boundary
is exact at every resolution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseShape(ABC):
    """An analytic solid, defined by its signed distance function."""

    name: str = ""
    label: str = ""
    description: str = ""

    @abstractmethod
    def sdf(self, x, y, z) -> np.ndarray:
        """Signed distance to the surface. Negative inside."""

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as ``(lower, upper)``."""

    def parameters(self) -> dict[str, Any]:
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in vars(self).items()
            if not k.startswith("_")
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        inner = ", ".join(f"{k}={v!r}" for k, v in self.parameters().items())
        return f"{type(self).__name__}({inner})"


_SHAPES: dict[str, type[BaseShape]] = {}


def register_shape(cls: type[BaseShape]) -> type[BaseShape]:
    """Class decorator adding a shape to the registry."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name")
    _SHAPES[cls.name] = cls
    return cls


def get_shape(name: str, **params: Any) -> BaseShape:
    """Construct a shape by name."""
    try:
        cls = _SHAPES[name]
    except KeyError:
        raise KeyError(
            f"unknown shape '{name}'. Available: {', '.join(sorted(_SHAPES))}"
        ) from None
    return cls(**params)


def shape_class(name: str) -> type[BaseShape]:
    try:
        return _SHAPES[name]
    except KeyError:
        raise KeyError(
            f"unknown shape '{name}'. Available: {', '.join(sorted(_SHAPES))}"
        ) from None


def list_shapes() -> tuple[type[BaseShape], ...]:
    return tuple(_SHAPES.values())


def shape_names() -> tuple[str, ...]:
    return tuple(_SHAPES)
