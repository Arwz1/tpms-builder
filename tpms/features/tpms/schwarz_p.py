"""Schwarz Primitive ("P") surface.

Cubic symmetry with straight open channels along all three axes — the most permeable of
the four and the easiest to clear of powder or resin, which is why it dominates heat
exchangers and flow media. Stiffer along the axes than diagonally.

    F = cos(u) + cos(v) + cos(w)
"""

from __future__ import annotations

import numpy as np

from tpms.features.tpms.base import TPMSPattern, register_pattern


class SchwarzP(TPMSPattern):
    name = "schwarz_p"
    label = "Schwarz P"
    description = (
        "Schwarz primitive — straight channels on all three axes, highest "
        "permeability, easiest to de-powder. Anisotropic stiffness."
    )
    typical_amplitude = 3.0

    def field(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.cos(u) + np.cos(v) + np.cos(w)

    def gradient(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return -np.sin(u), -np.sin(v), -np.sin(w)


register_pattern(SchwarzP())
