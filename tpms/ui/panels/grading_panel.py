"""Grading panel — how cell size and wall thickness vary across the part.

One stacked page per grading law. Each page owns its own controls and knows how to read
and write its slice of ``grading_params``, so adding a law means adding a page here and
nothing else.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLineEdit, QStackedWidget, QWidget

from tpms.core.expressions import VARIABLE_HELP
from tpms.features.generate import GenerationParams
from tpms.features.grading import get_grading, list_gradings
from tpms.ui.panels.widgets import (
    BasePanel,
    InfoLabel,
    axis_choice,
    choice,
    current_choice,
    muted,
    number,
    section,
    set_choice,
)


class _Page(QWidget):
    """Base for one grading law's controls."""

    grading_name = ""

    def values(self) -> dict:
        raise NotImplementedError

    def load(self, params: dict) -> None:
        raise NotImplementedError

    def representative_cell(self) -> float:
        return 6.0

    def representative_wall(self) -> float:
        return 1.0


class _UniformPage(_Page):
    grading_name = "uniform"

    def __init__(self, panel: BasePanel) -> None:
        super().__init__()
        box, form = section("Uniform")
        self.cell = number(6.0, 0.05, 1000.0, step=0.5,
                           tooltip="Length of one repeating unit cell.")
        self.wall = number(1.0, 0.01, 500.0, step=0.1,
                           tooltip="Thickness of the lattice wall.")
        form.addRow("Cell size", self.cell)
        form.addRow("Wall thickness", self.wall)

        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        panel.wire(self.cell, self.wall)

    def values(self) -> dict:
        return {"cell": self.cell.value(), "wall": self.wall.value()}

    def load(self, params: dict) -> None:
        self.cell.setValue(float(params.get("cell", 6.0)))
        self.wall.setValue(float(params.get("wall", 1.0)))

    def representative_cell(self) -> float:
        return float(self.cell.value())

    def representative_wall(self) -> float:
        return float(self.wall.value())


class _AxialPage(_Page):
    grading_name = "axial"

    def __init__(self, panel: BasePanel) -> None:
        super().__init__()
        box, form = section("Axial grade")
        self.axis = axis_choice(2)
        self.cell_start = number(4.0, 0.05, 1000.0, step=0.5)
        self.cell_end = number(10.0, 0.05, 1000.0, step=0.5)
        self.wall_start = number(1.2, 0.01, 500.0, step=0.1)
        self.wall_end = number(0.8, 0.01, 500.0, step=0.1)

        form.addRow("Axis", self.axis)
        form.addRow("Cell at start", self.cell_start)
        form.addRow("Cell at end", self.cell_end)
        form.addRow("Wall at start", self.wall_start)
        form.addRow("Wall at end", self.wall_end)
        form.addRow(muted(
            "Phase is integrated exactly along this axis, so cells stay undistorted "
            "however steep the grade."
        ))

        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        panel.wire(self.axis, self.cell_start, self.cell_end,
                   self.wall_start, self.wall_end)

    def values(self) -> dict:
        return {
            "axis": int(self.axis.currentData()),
            "cell_start": self.cell_start.value(),
            "cell_end": self.cell_end.value(),
            "wall_start": self.wall_start.value(),
            "wall_end": self.wall_end.value(),
        }

    def load(self, params: dict) -> None:
        self.axis.setCurrentIndex(int(params.get("axis", 2)))
        self.cell_start.setValue(float(params.get("cell_start", 4.0)))
        self.cell_end.setValue(float(params.get("cell_end", 10.0)))
        self.wall_start.setValue(float(params.get("wall_start", 1.2)))
        self.wall_end.setValue(float(params.get("wall_end", 0.8)))

    def representative_cell(self) -> float:
        return (self.cell_start.value() + self.cell_end.value()) * 0.5

    def representative_wall(self) -> float:
        return (self.wall_start.value() + self.wall_end.value()) * 0.5


class _RadialPage(_Page):
    grading_name = "radial"

    def __init__(self, panel: BasePanel) -> None:
        super().__init__()
        box, form = section("Radial grade")
        self.kind = choice(
            [("sphere", "From a point (spherical)", ""),
             ("axis", "From a line (cylindrical)", "")],
            "sphere",
        )
        self.axis = axis_choice(2)
        self.radius_inner = number(0.0, 0.0, 5000.0, step=1.0)
        self.radius_outer = number(30.0, 0.01, 5000.0, step=1.0)
        self.cell_inner = number(4.0, 0.05, 1000.0, step=0.5)
        self.cell_outer = number(10.0, 0.05, 1000.0, step=0.5)
        self.wall_inner = number(1.2, 0.01, 500.0, step=0.1)
        self.wall_outer = number(0.8, 0.01, 500.0, step=0.1)

        form.addRow("Measured", self.kind)
        form.addRow("Axis", self.axis)
        form.addRow("Inner radius", self.radius_inner)
        form.addRow("Outer radius", self.radius_outer)
        form.addRow("Cell inner", self.cell_inner)
        form.addRow("Cell outer", self.cell_outer)
        form.addRow("Wall inner", self.wall_inner)
        form.addRow("Wall outer", self.wall_outer)

        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.kind.currentIndexChanged.connect(self._sync)
        panel.wire(self.kind, self.axis, self.radius_inner, self.radius_outer,
                   self.cell_inner, self.cell_outer, self.wall_inner, self.wall_outer)
        self._sync()

    def _sync(self) -> None:
        self.axis.setEnabled(current_choice(self.kind) == "axis")

    def values(self) -> dict:
        return {
            "axis": int(self.axis.currentData())
                    if current_choice(self.kind) == "axis" else None,
            "radius_inner": self.radius_inner.value(),
            "radius_outer": self.radius_outer.value(),
            "cell_inner": self.cell_inner.value(),
            "cell_outer": self.cell_outer.value(),
            "wall_inner": self.wall_inner.value(),
            "wall_outer": self.wall_outer.value(),
        }

    def load(self, params: dict) -> None:
        axis = params.get("axis")
        set_choice(self.kind, "sphere" if axis is None else "axis")
        if axis is not None:
            self.axis.setCurrentIndex(int(axis))
        self.radius_inner.setValue(float(params.get("radius_inner", 0.0)))
        self.radius_outer.setValue(float(params.get("radius_outer", 30.0)))
        self.cell_inner.setValue(float(params.get("cell_inner", 4.0)))
        self.cell_outer.setValue(float(params.get("cell_outer", 10.0)))
        self.wall_inner.setValue(float(params.get("wall_inner", 1.2)))
        self.wall_outer.setValue(float(params.get("wall_outer", 0.8)))
        self._sync()

    def representative_cell(self) -> float:
        return (self.cell_inner.value() + self.cell_outer.value()) * 0.5

    def representative_wall(self) -> float:
        return (self.wall_inner.value() + self.wall_outer.value()) * 0.5


class _SurfacePage(_Page):
    grading_name = "surface_distance"

    def __init__(self, panel: BasePanel) -> None:
        super().__init__()
        box, form = section("Distance to surface")
        self.depth = number(10.0, 0.01, 5000.0, step=1.0,
                            tooltip="Depth below the skin at which the core values "
                                    "are reached.")
        self.cell_surface = number(4.0, 0.05, 1000.0, step=0.5)
        self.cell_core = number(10.0, 0.05, 1000.0, step=0.5)
        self.wall_surface = number(1.2, 0.01, 500.0, step=0.1)
        self.wall_core = number(0.8, 0.01, 500.0, step=0.1)

        form.addRow("Transition depth", self.depth)
        form.addRow("Cell at surface", self.cell_surface)
        form.addRow("Cell at core", self.cell_core)
        form.addRow("Wall at surface", self.wall_surface)
        form.addRow("Wall at core", self.wall_core)
        form.addRow(muted(
            "Dense skin, open core — the usual way to take weight out of a part."
        ))

        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        panel.wire(self.depth, self.cell_surface, self.cell_core,
                   self.wall_surface, self.wall_core)

    def values(self) -> dict:
        return {
            "depth": self.depth.value(),
            "cell_surface": self.cell_surface.value(),
            "cell_core": self.cell_core.value(),
            "wall_surface": self.wall_surface.value(),
            "wall_core": self.wall_core.value(),
        }

    def load(self, params: dict) -> None:
        self.depth.setValue(float(params.get("depth", 10.0)))
        self.cell_surface.setValue(float(params.get("cell_surface", 4.0)))
        self.cell_core.setValue(float(params.get("cell_core", 10.0)))
        self.wall_surface.setValue(float(params.get("wall_surface", 1.2)))
        self.wall_core.setValue(float(params.get("wall_core", 0.8)))

    def representative_cell(self) -> float:
        return (self.cell_surface.value() + self.cell_core.value()) * 0.5

    def representative_wall(self) -> float:
        return (self.wall_surface.value() + self.wall_core.value()) * 0.5


class _ExpressionPage(_Page):
    grading_name = "expression"

    def __init__(self, panel: BasePanel) -> None:
        super().__init__()
        self._panel = panel

        box, form = section("Expression")
        self.cell_edit = QLineEdit("5 + 4*w")
        self.wall_edit = QLineEdit("1.0")
        form.addRow("Cell size =", self.cell_edit)
        form.addRow("Wall thickness =", self.wall_edit)

        self.status = InfoLabel("")
        form.addRow(self.status)

        help_text = "  ".join(f"{k} = {v}" for k, v in VARIABLE_HELP.items())
        form.addRow(muted("Variables: " + help_text))
        form.addRow(muted(
            "Functions: sin cos tan exp log sqrt abs min max clamp lerp "
            "smoothstep hypot mod where"
        ))

        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.cell_edit.textChanged.connect(self._validate)
        self.wall_edit.textChanged.connect(self._validate)
        self._validate()

    def _validate(self) -> None:
        grading = get_grading(
            "expression",
            cell_expression=self.cell_edit.text(),
            wall_expression=self.wall_edit.text(),
        )
        message = grading.validate()

        invalid = message is not None
        for edit in (self.cell_edit, self.wall_edit):
            edit.setProperty("invalid", invalid)
            edit.style().unpolish(edit)
            edit.style().polish(edit)

        if invalid:
            self.status.error(message)
        else:
            self.status.success("Both expressions are valid.")
            self._panel.emit_changed()

    @property
    def is_valid(self) -> bool:
        return get_grading(
            "expression",
            cell_expression=self.cell_edit.text(),
            wall_expression=self.wall_edit.text(),
        ).validate() is None

    def values(self) -> dict:
        return {
            "cell_expression": self.cell_edit.text(),
            "wall_expression": self.wall_edit.text(),
        }

    def load(self, params: dict) -> None:
        self.cell_edit.setText(str(params.get("cell_expression", "5 + 4*w")))
        self.wall_edit.setText(str(params.get("wall_expression", "1.0")))


PAGE_TYPES = (_UniformPage, _AxialPage, _RadialPage, _SurfacePage, _ExpressionPage)


class GradingPanel(BasePanel):
    """Chooses a grading law and edits its parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        box, form = section("Grading")
        self.grading_combo = choice(
            [(g.name, g.label, g.description) for g in list_gradings()], "uniform"
        )
        form.addRow("Vary by", self.grading_combo)
        self.note = muted("")
        form.addRow(self.note)
        self.add(box)

        self.stack = QStackedWidget()
        self._pages: dict[str, _Page] = {}
        for page_type in PAGE_TYPES:
            page = page_type(self)
            self._pages[page_type.grading_name] = page
            self.stack.addWidget(page)
        self.add(self.stack)
        self.add_stretch()

        self.grading_combo.currentIndexChanged.connect(self._on_grading_changed)
        self.wire(self.grading_combo)
        self._on_grading_changed()

    def _on_grading_changed(self) -> None:
        name = current_choice(self.grading_combo)
        page = self._pages.get(name)
        if page is not None:
            self.stack.setCurrentWidget(page)
        for grading in list_gradings():
            if grading.name == name:
                self.note.setText(grading.description)
                break

    @property
    def current_page(self) -> _Page:
        return self._pages[current_choice(self.grading_combo)]

    @property
    def is_valid(self) -> bool:
        page = self.current_page
        return getattr(page, "is_valid", True)

    def representative_cell(self) -> float:
        return self.current_page.representative_cell()

    def representative_wall(self) -> float:
        return self.current_page.representative_wall()

    def suggest_bounds(self, lower, upper) -> None:
        """Set radius and depth defaults from the loaded geometry's size."""
        import numpy as np

        extent = np.asarray(upper) - np.asarray(lower)
        half_diagonal = float(np.linalg.norm(extent)) * 0.5

        with self.loading():
            radial = self._pages["radial"]
            if radial.radius_outer.value() <= 30.0001:
                radial.radius_outer.setValue(round(half_diagonal, 1))

            surface = self._pages["surface_distance"]
            if surface.depth.value() <= 10.0001:
                surface.depth.setValue(round(float(extent.min()) * 0.25, 1))

    # --------------------------------------------------------------- parameters

    def apply_to(self, params: GenerationParams) -> None:
        params.grading = current_choice(self.grading_combo)
        params.grading_params = self.current_page.values()

    def load_from(self, params: GenerationParams) -> None:
        with self.loading():
            set_choice(self.grading_combo, params.grading)
            page = self._pages.get(params.grading)
            if page is not None:
                page.load(params.grading_params)
        self._on_grading_changed()
