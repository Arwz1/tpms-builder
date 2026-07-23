"""Schwarz Diamond ("D") surface.

Splits space into two interlocking diamond-lattice labyrinths. The stiffest of the four
per unit mass and the most nearly isotropic, at the cost of steeper overhangs than the
gyroid — worth checking print orientation.

    F = sin(u)sin(v)sin(w) + sin(u)cos(v)cos(w)
      + cos(u)sin(v)cos(w) + cos(u)cos(v)sin(w)
"""

from __future__ import annotations

import numpy as np

from tpms.features.tpms.base import TPMSPattern, register_pattern


class SchwarzD(TPMSPattern):
    name = "schwarz_d"
    label = "Schwarz D (Diamond)"
    description = (
        "Schwarz diamond — two interlocking labyrinths. Highest stiffness-to-mass "
        "of the four, steeper overhangs than the gyroid."
    )
    typical_amplitude = 1.0

    def field(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)

        return su * sv * sw + su * cv * cw + cu * sv * cw + cu * cv * sw

    def gradient(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)

        return (
            cu * sv * sw + cu * cv * cw - su * sv * cw - su * cv * sw,
            su * cv * sw - su * sv * cw + cu * cv * cw - cu * sv * sw,
            su * sv * cw - su * cv * sw - cu * sv * sw + cu * cv * cw,
        )


register_pattern(SchwarzD())
