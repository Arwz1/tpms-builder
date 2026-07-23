"""Analytic base shapes.

Each wraps a primitive from :mod:`tpms.core.sdf` with parameters and a bounding box.
All are centred on the origin, which keeps the grading defaults (radial from centre,
axial across the box) meaningful without any setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpms.core import sdf
from tpms.features.shapes.base import BaseShape, register_shape


@register_shape
@dataclass
class BoxShape(BaseShape):
    name = "box"
    label = "Box"
    description = "Rectangular block."

    width: float = 60.0
    depth: float = 60.0
    height: float = 60.0

    def sdf(self, x, y, z) -> np.ndarray:
        return sdf.box(
            x, y, z,
            (self.width * 0.5, self.depth * 0.5, self.height * 0.5),
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = np.array([self.width, self.depth, self.height], dtype=np.float64) * 0.5
        return -half, half


@register_shape
@dataclass
class SphereShape(BaseShape):
    name = "sphere"
    label = "Sphere"
    description = "Solid sphere."

    radius: float = 30.0

    def sdf(self, x, y, z) -> np.ndarray:
        return sdf.sphere(x, y, z, self.radius)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = np.full(3, float(self.radius))
        return -half, half


@register_shape
@dataclass
class CylinderShape(BaseShape):
    name = "cylinder"
    label = "Cylinder"
    description = "Capped cylinder along the chosen axis."

    radius: float = 25.0
    height: float = 60.0
    axis: int = 2

    def sdf(self, x, y, z) -> np.ndarray:
        return sdf.cylinder(x, y, z, self.radius, self.height, axis=self.axis)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = np.full(3, float(self.radius))
        half[int(self.axis)] = float(self.height) * 0.5
        return -half, half


@register_shape
@dataclass
class ConeShape(BaseShape):
    name = "cone"
    label = "Cone / frustum"
    description = "Truncated cone. A top radius of zero gives a point."

    radius_bottom: float = 30.0
    radius_top: float = 0.0
    height: float = 60.0
    axis: int = 2

    def sdf(self, x, y, z) -> np.ndarray:
        return sdf.cone(
            x, y, z,
            self.radius_bottom, self.radius_top, self.height, axis=self.axis,
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        widest = max(float(self.radius_bottom), float(self.radius_top))
        half = np.full(3, widest)
        half[int(self.axis)] = float(self.height) * 0.5
        return -half, half


@register_shape
@dataclass
class TorusShape(BaseShape):
    name = "torus"
    label = "Torus"
    description = "Ring. Major radius to the tube centre, minor radius of the tube."

    major_radius: float = 30.0
    minor_radius: float = 10.0
    axis: int = 2

    def sdf(self, x, y, z) -> np.ndarray:
        return sdf.torus(
            x, y, z, self.major_radius, self.minor_radius, axis=self.axis
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        outer = float(self.major_radius) + float(self.minor_radius)
        half = np.full(3, outer)
        half[int(self.axis)] = float(self.minor_radius)
        return -half, half


@register_shape
@dataclass
class CapsuleShape(BaseShape):
    name = "capsule"
    label = "Capsule"
    description = "Cylinder with hemispherical ends. Height is the cylindrical run."

    radius: float = 20.0
    height: float = 40.0
    axis: int = 2

    def sdf(self, x, y, z) -> np.ndarray:
        return sdf.capsule(x, y, z, self.radius, self.height, axis=self.axis)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = np.full(3, float(self.radius))
        half[int(self.axis)] = float(self.height) * 0.5 + float(self.radius)
        return -half, half
