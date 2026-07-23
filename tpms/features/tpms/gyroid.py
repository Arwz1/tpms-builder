"""Schoen Gyroid.

The default for lattice work: no straight channels, no self-intersections, and every
strut meets its neighbours at a smooth junction, so it prints without support in any
orientation and has near-isotropic stiffness.

    F = sin(u)cos(v) + sin(v)cos(w) + sin(w)cos(u)
"""

from __future__ import annotations

import numpy as np

from tpms.features.tpms.base import TPMSPattern, register_pattern


class Gyroid(TPMSPattern):
    name = "gyroid"
    label = "Gyroid"
    description = (
        "Schoen gyroid — self-supporting, near-isotropic, no straight channels. "
        "The usual first choice for printed lattices."
    )
    typical_amplitude = 1.5

    def field(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        return (
            np.sin(u) * np.cos(v)
            + np.sin(v) * np.cos(w)
            + np.sin(w) * np.cos(u)
        )

    def gradient(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)

        return (
            cu * cv - sw * su,
            cv * cw - su * sv,
            cw * cu - sv * sw,
        )


register_pattern(Gyroid())
