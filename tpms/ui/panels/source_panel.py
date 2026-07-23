"""Source panel — import a file, or pick a base shape.

Base shapes are what let the app do something useful before any geometry exists, which
matters when the point is to try patterns and cell sizes quickly.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QWidget,
)

from tpms.features.generate import GenerationParams, SourceType
from tpms.features.shapes import list_shapes, shape_class
from tpms.ui.panels.widgets import (
    BasePanel,
    InfoLabel,
    axis_choice,
    choice,
    current_choice,
    number,
    section,
    set_choice,
)

# Fields each shape exposes, with sensible ranges. Kept beside the panel rather than on
# the shape, so the feature stays free of UI concerns.
SHAPE_FIELDS: dict[str, list[tuple[str, str, float, float, float]]] = {
    "box":      [("width", "Width", 60.0, 0.1, 5000.0),
                 ("depth", "Depth", 60.0, 0.1, 5000.0),
                 ("height", "Height", 60.0, 0.1, 5000.0)],
    "sphere":   [("radius", "Radius", 30.0, 0.1, 5000.0)],
    "cylinder": [("radius", "Radius", 25.0, 0.1, 5000.0),
                 ("height", "Height", 60.0, 0.1, 5000.0)],
    "cone":     [("radius_bottom", "Bottom radius", 30.0, 0.0, 5000.0),
                 ("radius_top", "Top radius", 0.0, 0.0, 5000.0),
                 ("height", "Height", 60.0, 0.1, 5000.0)],
    "torus":    [("major_radius", "Major radius", 30.0, 0.1, 5000.0),
                 ("minor_radius", "Minor radius", 10.0, 0.1, 5000.0)],
    "capsule":  [("radius", "Radius", 20.0, 0.1, 5000.0),
                 ("height", "Cylinder height", 40.0, 0.0, 5000.0)],
}

#: Shapes with a meaningful axis.
AXIAL_SHAPES = {"cylinder", "cone", "torus", "capsule"}


class SourcePanel(BasePanel):
    """Chooses the solid to be filled."""

    file_requested = Signal(str)
    source_type_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._shape_widgets: dict[str, dict[str, QWidget]] = {}

        # ---- mode -----------------------------------------------------------
        mode_box, mode_form = section("Source")
        self.shape_radio = QRadioButton("Base shape")
        self.file_radio = QRadioButton("Imported geometry")
        self.shape_radio.setChecked(True)

        row = QHBoxLayout()
        row.addWidget(self.shape_radio)
        row.addWidget(self.file_radio)
        row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        mode_form.addRow(holder)
        self.add(mode_box)

        self.shape_radio.toggled.connect(self._on_mode_changed)

        # ---- the two alternatives -------------------------------------------
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_shape_page())
        self.stack.addWidget(self._build_file_page())
        self.add(self.stack)

        self.status = InfoLabel("")
        self.add(self.status)
        self.add_stretch()

        self._on_shape_changed()

    # -------------------------------------------------------------------- pages

    def _build_shape_page(self) -> QWidget:
        page = QWidget()
        layout = self._page_layout(page)

        box, form = section("Base shape")
        self.shape_combo = choice(
            [(s.name, s.label, s.description) for s in list_shapes()], "box"
        )
        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        form.addRow("Shape", self.shape_combo)

        # One spin box per parameter of every shape; only the active shape's are shown.
        for name, fields in SHAPE_FIELDS.items():
            self._shape_widgets[name] = {}
            for key, label, default, low, high in fields:
                spin = number(default, low, high, step=1.0)
                spin.valueChanged.connect(self.emit_changed)
                self._shape_widgets[name][key] = spin
                form.addRow(label, spin)
                spin.setVisible(False)
                form.labelForField(spin).setVisible(False)

        self.axis_combo = axis_choice(2)
        self.axis_combo.currentIndexChanged.connect(self.emit_changed)
        form.addRow("Axis", self.axis_combo)

        self._shape_form = form
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _build_file_page(self) -> QWidget:
        page = QWidget()
        layout = self._page_layout(page)

        box, form = section("Imported geometry")

        self.browse_button = QPushButton("Choose file…")
        self.browse_button.clicked.connect(self._browse)
        form.addRow(self.browse_button)

        self.file_label = InfoLabel("No file chosen.")
        form.addRow(self.file_label)

        self.import_status = InfoLabel("")
        form.addRow(self.import_status)

        layout.addWidget(box)
        layout.addWidget(self._formats_note())
        layout.addStretch(1)
        return page

    def _formats_note(self) -> InfoLabel:
        from tpms.io import registered_loaders

        lines = []
        for info in registered_loaders():
            extensions = ", ".join(f".{e}" for e in info.extensions)
            reason = info.unavailable_reason()
            if reason is None:
                lines.append(f"{info.name}: {extensions}")
            else:
                lines.append(f"{info.name}: {extensions} — unavailable. {reason}")

        label = InfoLabel("\n".join(lines))
        return label

    def _page_layout(self, page: QWidget):
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        return layout

    # ----------------------------------------------------------------- handlers

    def _on_mode_changed(self) -> None:
        self.stack.setCurrentIndex(0 if self.shape_radio.isChecked() else 1)
        self.source_type_changed.emit()
        self.emit_changed()

    def _on_shape_changed(self) -> None:
        name = current_choice(self.shape_combo)

        for shape_name, widgets in self._shape_widgets.items():
            visible = shape_name == name
            for spin in widgets.values():
                spin.setVisible(visible)
                label = self._shape_form.labelForField(spin)
                if label is not None:
                    label.setVisible(visible)

        show_axis = name in AXIAL_SHAPES
        self.axis_combo.setVisible(show_axis)
        axis_label = self._shape_form.labelForField(self.axis_combo)
        if axis_label is not None:
            axis_label.setVisible(show_axis)

        self.emit_changed()

    def _browse(self) -> None:
        from tpms.io import qt_file_filter

        path, _ = QFileDialog.getOpenFileName(
            self, "Import geometry", "", qt_file_filter()
        )
        if path:
            self.set_file(path)
            self.file_requested.emit(path)

    # -------------------------------------------------------------------- state

    def set_file(self, path: str) -> None:
        self._path = path
        self.file_label.info(
            f"{os.path.basename(path)}\n{_format_size(path)}"
        )
        self.file_radio.setChecked(True)
        self.emit_changed()

    def report_import(self, message: str, ok: bool = True) -> None:
        if ok:
            self.import_status.success(message)
        else:
            self.import_status.error(message)

    @property
    def path(self) -> str:
        return getattr(self, "_path", "")

    # --------------------------------------------------------------- parameters

    def apply_to(self, params: GenerationParams) -> None:
        if self.shape_radio.isChecked():
            params.source_type = SourceType.SHAPE
            name = current_choice(self.shape_combo)
            params.shape_name = name

            values = {
                key: float(spin.value())
                for key, spin in self._shape_widgets[name].items()
            }
            if name in AXIAL_SHAPES:
                values["axis"] = int(self.axis_combo.currentData())
            params.shape_params = values
        else:
            params.source_type = SourceType.FILE
            params.source_path = self.path

    def load_from(self, params: GenerationParams) -> None:
        with self.loading():
            if params.source_type is SourceType.SHAPE:
                self.shape_radio.setChecked(True)
                set_choice(self.shape_combo, params.shape_name)
                widgets = self._shape_widgets.get(params.shape_name, {})
                for key, value in params.shape_params.items():
                    if key == "axis":
                        self.axis_combo.setCurrentIndex(int(value))
                    elif key in widgets:
                        widgets[key].setValue(float(value))
            else:
                self.file_radio.setChecked(True)
                if params.source_path:
                    self._path = params.source_path
                    self.file_label.info(os.path.basename(params.source_path))
        self._on_shape_changed()


def _format_size(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
