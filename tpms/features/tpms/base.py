"""The TPMS pattern contract and its registry.

Every pattern is a 2π-periodic scalar field evaluated in **phase coordinates**. The
pipeline converts world coordinates to phase with ``u = 2π x / cell_size``, so a pattern
never needs to know the cell size, and a spatially varying cell size costs it nothing.

Adding a pattern means writing one file that subclasses :class:`TPMSPattern` and calling
:func:`register_pattern`. No other file changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Below this the gradient is treated as degenerate. |grad| vanishes at the isolated
# critical points of these fields; dividing by it there would produce infinities that
# poison the whole slab.
MIN_GRADIENT = 1e-6


class TPMSPattern(ABC):
    """A triply periodic scalar field.

    Subclasses implement :meth:`field` and :meth:`gradient_norm`, both in phase
    coordinates where the period is 2π on every axis.
    """

    #: Stable identifier used in saved parameters and on the CLI.
    name: str = ""
    #: Human-readable name for the UI.
    label: str = ""
    #: One-line explanation shown as a tooltip.
    description: str = ""

    @abstractmethod
    def field(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Evaluate the field. Zero is the minimal surface itself."""

    @abstractmethod
    def gradient(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate ``∇field`` in phase coordinates, as three components.

        The gradient is what turns a dimensionless field value into a distance: near
        the surface, ``d ≈ F / |∇F|``. That is what lets the UI offer wall thickness as
        a real millimetre dimension instead of an opaque field threshold.

        The direction matters as well as the magnitude — :func:`~tpms.features.tpms.
        modes.solidify` steps along the normal to refine the distance estimate, because
        one Taylor step alone is only accurate for thin walls.
        """

    def gradient_norm(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """``|∇field|`` in phase coordinates. Derived from :meth:`gradient`."""
        du, dv, dw = self.gradient(u, v, w)
        return np.sqrt(du * du + dv * dv + dw * dw)

    # ------------------------------------------------------------------ helpers

    def signed_distance(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        phase_scale: np.ndarray | float,
    ) -> np.ndarray:
        """Approximate signed distance to the surface, in world units.

        ``phase_scale`` is ``2π / cell_size`` — the factor converting a world length
        into phase. The phase-space gradient is scaled by it to become a world-space
        gradient, and the field is divided through.
        """
        value = self.field(u, v, w)
        grad = self.gradient_norm(u, v, w) * np.asarray(phase_scale, dtype=np.float32)
        return value / np.maximum(grad, MIN_GRADIENT)

    def volume_fraction_offset(self, fraction: float) -> float:
        """Field level giving roughly ``fraction`` solid, for solid-network mode.

        The level-to-fraction relation has no closed form, so this is the linear
        approximation that holds near the 50 % point. Subclasses with a known better
        fit may override it.
        """
        fraction = float(np.clip(fraction, 0.01, 0.99))
        return (fraction - 0.5) * 2.0 * self.typical_amplitude

    #: Rough peak magnitude of the field, used by the fraction approximation above.
    typical_amplitude: float = 1.5

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} '{self.name}'>"


_PATTERNS: dict[str, TPMSPattern] = {}


def register_pattern(pattern: TPMSPattern) -> TPMSPattern:
    """Add a pattern to the registry, keyed on its ``name``."""
    if not pattern.name:
        raise ValueError(f"{type(pattern).__name__} has no name")
    _PATTERNS[pattern.name] = pattern
    return pattern


def get_pattern(name: str) -> TPMSPattern:
    """Look up a pattern by name."""
    try:
        return _PATTERNS[name]
    except KeyError:
        raise KeyError(
            f"unknown TPMS pattern '{name}'. Available: {', '.join(sorted(_PATTERNS))}"
        ) from None


def list_patterns() -> tuple[TPMSPattern, ...]:
    """Every registered pattern, in registration order."""
    return tuple(_PATTERNS.values())


def pattern_names() -> tuple[str, ...]:
    return tuple(_PATTERNS)
