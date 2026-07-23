"""Turning a periodic scalar field into a solid.

A TPMS is a *surface*; a printable part needs a *volume*. There are two ways to get one,
and they produce physically different structures:

**Sheet** — thicken the surface itself into a wall. Both labyrinths stay open and
connected, so the part is a shell network. Best stiffness per unit mass, and the two
void spaces can carry different fluids without mixing.

**Solid network** — fill one labyrinth completely and leave the other empty. Chunkier
struts, more mass at equal cell size, but a single continuous void that is far easier to
clear of trapped powder or resin.

**Inverse network** — fill the other labyrinth. For most patterns this is congruent to
the solid network; where it is not, it gives a genuinely different structure.

All three return a signed distance in world units, negative inside, matching
:mod:`tpms.core.sdf` so the result composes with every CSG operation.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from tpms.features.tpms.base import MIN_GRADIENT, TPMSPattern


class SolidMode(str, Enum):
    """How to give the surface a volume."""

    SHEET = "sheet"
    SOLID = "solid"
    INVERSE = "inverse"

    @property
    def label(self) -> str:
        return {
            SolidMode.SHEET: "Sheet (thickened surface)",
            SolidMode.SOLID: "Solid network",
            SolidMode.INVERSE: "Inverse network",
        }[self]

    @property
    def description(self) -> str:
        return {
            SolidMode.SHEET: (
                "Thickens the minimal surface into a wall. Two separate open void "
                "networks. Best stiffness for the mass."
            ),
            SolidMode.SOLID: (
                "Fills one of the two labyrinths. Heavier at equal cell size, but a "
                "single connected void that de-powders easily."
            ),
            SolidMode.INVERSE: (
                "Fills the other labyrinth. Congruent to the solid network for most "
                "patterns, distinct for the asymmetric ones."
            ),
        }[self]

    @property
    def uses_thickness(self) -> bool:
        """Sheet mode is driven by wall thickness, the networks by volume fraction."""
        return self is SolidMode.SHEET


def signed_distance(
    pattern: TPMSPattern,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    phase_scale: np.ndarray | float,
    level: float = 0.0,
    refine_steps: int = 2,
) -> np.ndarray:
    """Distance from each point to the ``level`` isosurface, in world units.

    The first-order estimate ``d = (F - level) / |∇F|`` is only accurate very close to
    the surface, and a TPMS wall is not thin in those terms. Measured against true wall
    thickness at a 6 mm cell, one Taylor step gives:

    ==========  ============  ============  ============
    pattern     t/cell 0.07   t/cell 0.13   t/cell 0.20
    ==========  ============  ============  ============
    gyroid          -1.6 %        -6.0 %       -11.9 %
    schwarz_p       -0.6 %        -2.1 %        -4.5 %
    schwarz_d       -2.3 %        -8.0 %       -15.3 %
    neovius         +0.3 %        +2.3 %       +11.3 %
    ==========  ============  ============  ============

    Refining fixes it. Each pass steps to the predicted surface point and re-measures
    the residual there, costing one field and one gradient evaluation. Two passes bring
    every pattern to better than 0.8 %, except Neovius at 2.3 %, whose gradient varies
    fivefold across its own surface.

    The probe step is clamped to a quarter cell so a pass cannot overshoot onto the
    neighbouring wall, where the field is small again for the wrong reason. Only values
    near the zero level steer marching cubes, so clamping the far field costs nothing.
    """
    phase_scale = np.asarray(phase_scale, dtype=np.float32)

    value = pattern.field(u, v, w) - level
    du, dv, dw = pattern.gradient(u, v, w)

    grad_sq = du * du + dv * dv + dw * dw
    grad_norm = np.maximum(np.sqrt(grad_sq), MIN_GRADIENT)

    # Phase-space gradient scaled into world space.
    distance = value / (grad_norm * phase_scale)

    # Quarter of a cell, in phase units.
    max_step = np.float32(np.pi * 0.5)

    for _ in range(max(0, int(refine_steps))):
        # Unit normal in phase space, which is also the world-space normal direction.
        inv = 1.0 / grad_norm
        step = np.clip(distance * phase_scale, -max_step, max_step)

        pu = u - step * (du * inv)
        pv = v - step * (dv * inv)
        pw = w - step * (dw * inv)

        value = pattern.field(pu, pv, pw) - level
        du, dv, dw = pattern.gradient(pu, pv, pw)
        grad_norm = np.maximum(
            np.sqrt(du * du + dv * dv + dw * dw), MIN_GRADIENT
        )

        # Total distance is how far the probe actually travelled plus the residual
        # measured there. Adding the residual to the *unclamped* estimate instead
        # would double-count whenever the clamp bites.
        distance = step / phase_scale + value / (grad_norm * phase_scale)

    return distance.astype(np.float32, copy=False)


def solidify(
    pattern: TPMSPattern,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    phase_scale: np.ndarray | float,
    mode: SolidMode | str = SolidMode.SHEET,
    thickness: np.ndarray | float = 1.0,
    volume_fraction: float = 0.5,
    refine_steps: int = 2,
) -> np.ndarray:
    """Evaluate a TPMS as a signed distance field.

    Parameters
    ----------
    pattern:
        The periodic field to solidify.
    u, v, w:
        Phase coordinates — world position times ``phase_scale``.
    phase_scale:
        ``2π / cell_size``. May be an array for graded cell size, in which case it must
        broadcast against ``u``.
    mode:
        One of :class:`SolidMode`.
    thickness:
        Wall thickness in world units, for sheet mode. May be an array for graded
        thickness.
    volume_fraction:
        Target solid fraction for the network modes, between 0 and 1.
    refine_steps:
        Newton passes used to sharpen the distance estimate. See
        :func:`signed_distance`.

    Returns
    -------
    Signed distance in world units, negative inside the solid.
    """
    mode = SolidMode(mode)

    if mode is SolidMode.SHEET:
        distance = signed_distance(
            pattern, u, v, w, phase_scale, level=0.0, refine_steps=refine_steps
        )
        thickness = np.asarray(thickness, dtype=np.float32)
        return (np.abs(distance) - thickness * 0.5).astype(np.float32, copy=False)

    level = pattern.volume_fraction_offset(volume_fraction)
    distance = signed_distance(
        pattern, u, v, w, phase_scale, level=level, refine_steps=refine_steps
    )

    if mode is SolidMode.SOLID:
        return distance
    return (-distance).astype(np.float32, copy=False)


def wall_thickness_limits(cell_size: float) -> tuple[float, float]:
    """Sensible thickness bounds for a given cell size.

    Below about 3 % of the cell the wall is thinner than most printers resolve; above
    about 35 % neighbouring walls merge and the void closes up, which turns a lattice
    into a solid block with voids in it. The UI clamps sliders to this range so the
    common way of producing an unprintable part is simply unavailable.
    """
    return cell_size * 0.03, cell_size * 0.35


def estimate_volume_fraction(
    pattern: TPMSPattern,
    mode: SolidMode | str,
    cell_size: float,
    thickness: float,
    samples: int = 32,
) -> float:
    """Estimate the solid fraction of one unit cell by direct sampling.

    Used by the UI to show a live mass estimate. A 32³ sample of one cell is enough for
    a percent-level answer and costs under a millisecond.
    """
    mode = SolidMode(mode)

    axis = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float32)
    u, v, w = np.meshgrid(axis, axis, axis, indexing="ij")

    phase_scale = np.float32(2.0 * np.pi / cell_size)
    field = solidify(
        pattern, u, v, w, phase_scale, mode=mode, thickness=thickness
    )
    return float((field < 0).mean())
