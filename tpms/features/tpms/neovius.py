"""Neovius surface.

A relative of the Schwarz P with a much larger surface area per unit volume and thicker
nodal junctions. The high area makes it the pick for heat transfer and catalysis; the
same property makes it heavier than the gyroid at equal wall thickness.

    F = 3(cos(u) + cos(v) + cos(w)) + 4cos(u)cos(v)cos(w)
"""

from __future__ import annotations

import numpy as np

from tpms.features.tpms.base import TPMSPattern, register_pattern


class Neovius(TPMSPattern):
    name = "neovius"
    label = "Neovius"
    description = (
        "Neovius — very high surface area per volume and chunky junctions. "
        "Suits heat transfer; heavier than a gyroid at the same wall thickness."
    )
    typical_amplitude = 7.0

    def field(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        cu, cv, cw = np.cos(u), np.cos(v), np.cos(w)
        return 3.0 * (cu + cv + cw) + 4.0 * cu * cv * cw

    def gradient(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)

        return (
            -3.0 * su - 4.0 * su * cv * cw,
            -3.0 * sv - 4.0 * cu * sv * cw,
            -3.0 * sw - 4.0 * cu * cv * sw,
        )


register_pattern(Neovius())
