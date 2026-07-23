"""Small reusable controls shared by the panels.

Kept here so each panel stays about its own feature rather than about Qt boilerplate.
"""

from __future__ import annotations

from typing import Callable, Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def section(title: str) -> tuple[QGroupBox, QFormLayout]:
    """A titled group box with a form layout inside."""
    box = QGroupBox(title)
    form = QFormLayout(box)
    form.setContentsMargins(10, 6, 10, 8)
    form.setSpacing(7)
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return box, form


def number(
    value: float,
    minimum: float = 0.0,
    maximum: float = 10_000.0,
    step: float = 0.1,
    decimals: int = 2,
    suffix: str = " mm",
    tooltip: str = "",
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    if tooltip:
        spin.setToolTip(tooltip)
    spin.setKeyboardTracking(False)
    return spin


def integer(
    value: int,
    minimum: int = 0,
    maximum: int = 1_000_000,
    step: int = 1,
    suffix: str = "",
    tooltip: str = "",
) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    if tooltip:
        spin.setToolTip(tooltip)
    spin.setKeyboardTracking(False)
    return spin


def choice(
    items: Iterable[tuple[str, str, str]], current: str = ""
) -> QComboBox:
    """A combo box built from ``(value, label, tooltip)`` triples."""
    combo = QComboBox()
    for value, label, tooltip in items:
        combo.addItem(label, userData=value)
        if tooltip:
            combo.setItemData(combo.count() - 1, tooltip, Qt.ToolTipRole)
    if current:
        set_choice(combo, current)
    return combo


def set_choice(combo: QComboBox, value: str) -> None:
    """Select the entry whose userData is ``value``, if present."""
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def current_choice(combo: QComboBox) -> str:
    data = combo.currentData()
    return str(data) if data is not None else ""


def axis_choice(current: int = 2) -> QComboBox:
    combo = QComboBox()
    for index, name in enumerate(("X", "Y", "Z")):
        combo.addItem(name, userData=index)
    combo.setCurrentIndex(int(current))
    return combo


def muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


class InfoLabel(QLabel):
    """A status line that colours itself by severity."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setWordWrap(True)
        self.setObjectName("muted")

    def _set(self, text: str, name: str) -> None:
        self.setText(text)
        self.setObjectName(name)
        # Re-polish so the new objectName's stylesheet rule takes effect.
        self.style().unpolish(self)
        self.style().polish(self)

    def info(self, text: str) -> None:
        self._set(text, "muted")

    def warn(self, text: str) -> None:
        self._set(text, "warning")

    def error(self, text: str) -> None:
        self._set(text, "error")

    def success(self, text: str) -> None:
        self._set(text, "success")


class BasePanel(QWidget):
    """Common shape for every parameter panel.

    Panels never touch the pipeline. They read and write
    :class:`~tpms.features.generate.params.GenerationParams` and emit
    :attr:`changed` — the main window decides what that means.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(9)
        self._loading = False

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def emit_changed(self, *_args) -> None:
        """Signal a parameter change, unless we are the ones setting the values."""
        if not self._loading:
            self.changed.emit()

    def wire(self, *widgets: QWidget) -> None:
        """Connect the usual value signals of each widget to :meth:`emit_changed`."""
        for widget in widgets:
            for name in ("valueChanged", "currentIndexChanged", "textChanged",
                         "toggled", "stateChanged"):
                signal = getattr(widget, name, None)
                if signal is not None:
                    signal.connect(self.emit_changed)
                    break

    class _Loading:
        def __init__(self, panel: "BasePanel") -> None:
            self.panel = panel

        def __enter__(self):
            self.panel._loading = True
            return self.panel

        def __exit__(self, *exc):
            self.panel._loading = False
            return False

    def loading(self) -> "_Loading":
        """Context manager suppressing :attr:`changed` while values are set."""
        return BasePanel._Loading(self)
