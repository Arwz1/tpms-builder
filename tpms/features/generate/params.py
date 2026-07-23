"""Generation parameters.

One serialisable dataclass describing a complete setup. Everything the pipeline needs
arrives here, which is what lets a run be saved, reloaded, sent to the CLI, or handed to
a worker thread without dragging live objects across.

Deliberately holds *names and values*, not constructed objects — a `dict` round-trips
through JSON, a compiled expression does not.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from tpms.core.grid import MAX_RESOLUTION, WARN_RESOLUTION
from tpms.features.tpms.modes import SolidMode


class SourceType(str, Enum):
    """Where the domain geometry comes from."""

    SHAPE = "shape"
    FILE = "file"


@dataclass
class GenerationParams:
    """A complete, serialisable description of one lattice generation."""

    # ---- domain ------------------------------------------------------------
    source_type: SourceType = SourceType.SHAPE
    shape_name: str = "box"
    shape_params: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    # ---- lattice -----------------------------------------------------------
    pattern: str = "gyroid"
    mode: SolidMode = SolidMode.SHEET
    volume_fraction: float = 0.5
    refine_steps: int = 2

    # ---- grading -----------------------------------------------------------
    grading: str = "uniform"
    grading_params: dict[str, Any] = field(default_factory=lambda: {"cell": 6.0, "wall": 1.0})

    # ---- shell -------------------------------------------------------------
    #: Solid skin thickness on the part surface, 0 for none.
    skin_thickness: float = 0.0
    #: Blend radius where the lattice meets the skin. 0 gives a sharp junction.
    skin_blend: float = 0.0

    # ---- quality -----------------------------------------------------------
    resolution: int = 192
    #: Resolution of the imported-geometry SDF. Independent of `resolution`; see
    #: docs/architecture.md on why the two are decoupled.
    domain_resolution: int = 160
    slab_depth: int = 16
    #: Drop connected components smaller than this, to clear clipping slivers.
    min_component_faces: int = 24

    # ---- output ------------------------------------------------------------
    export_path: str = ""

    # ------------------------------------------------------------- validation

    def __post_init__(self) -> None:
        self.normalise()

    def normalise(self) -> "GenerationParams":
        """Coerce types and clamp every value into its valid range.

        Called from ``__post_init__``, and again by the pipeline. Fields are plain
        dataclass attributes, so anything built once and then mutated — which is how
        both the UI and the CLI assemble a setup — would otherwise carry whatever was
        assigned. Re-applying the clamps at the point of use makes that harmless.
        """
        self.source_type = SourceType(self.source_type)
        self.mode = SolidMode(self.mode)
        self.resolution = int(max(8, min(int(self.resolution), MAX_RESOLUTION)))
        self.domain_resolution = int(max(8, min(int(self.domain_resolution), 256)))
        self.slab_depth = int(max(2, self.slab_depth))
        self.volume_fraction = float(min(max(self.volume_fraction, 0.01), 0.99))
        self.refine_steps = int(max(0, min(int(self.refine_steps), 6)))
        self.min_component_faces = int(max(0, self.min_component_faces))
        return self

    def validate(self) -> list[str]:
        """Return a list of problems, empty when the setup is usable."""
        problems: list[str] = []

        if self.source_type is SourceType.FILE and not self.source_path:
            problems.append("No geometry file chosen.")

        from tpms.features.grading import grading_names
        from tpms.features.shapes import shape_names
        from tpms.features.tpms import pattern_names

        if self.pattern not in pattern_names():
            problems.append(f"Unknown pattern '{self.pattern}'.")
        if self.grading not in grading_names():
            problems.append(f"Unknown grading '{self.grading}'.")
        if self.source_type is SourceType.SHAPE and self.shape_name not in shape_names():
            problems.append(f"Unknown base shape '{self.shape_name}'.")

        if self.grading == "expression":
            from tpms.features.grading import get_grading

            message = get_grading("expression", **self.grading_params).validate()
            if message:
                problems.append(message)

        return problems

    @property
    def is_high_resolution(self) -> bool:
        """True when the run warrants the quality panel's warning."""
        return self.resolution > WARN_RESOLUTION

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = self.source_type.value
        data["mode"] = self.mode.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "GenerationParams":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "GenerationParams":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_json(handle.read())

    # ------------------------------------------------------------- convenience

    def nominal_cell_size(self) -> float:
        """A representative cell size, for estimates and slider defaults."""
        p = self.grading_params
        for key in ("cell", "cell_start", "cell_inner", "cell_surface"):
            if key in p:
                return float(p[key])
        return 6.0

    def nominal_thickness(self) -> float:
        p = self.grading_params
        for key in ("wall", "wall_start", "wall_inner", "wall_surface"):
            if key in p:
                return float(p[key])
        return 1.0
