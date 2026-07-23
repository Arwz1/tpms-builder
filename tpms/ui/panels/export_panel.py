"""Export panel — write the generated lattice to disk.

Export always writes the full-resolution mesh, even when the viewport is showing a
decimated preview.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QPushButton, QWidget

from tpms.core.mesh import Mesh
from tpms.io import ExportError, export_extensions, export_mesh, qt_export_filter
from tpms.ui.panels.widgets import (
    BasePanel,
    InfoLabel,
    choice,
    current_choice,
    muted,
    section,
)


class ExportPanel(BasePanel):
    """Chooses a format and writes the mesh."""

    exported = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._mesh: Mesh | None = None

        box, form = section("Export")

        self.format_combo = choice(
            [
                ("stl", "STL — binary", "For slicers and printers. The usual choice."),
                ("obj", "OBJ — Wavefront", "Wide support, text-based, larger files."),
                ("ply", "PLY — Stanford", "Compact binary, good for further processing."),
                ("3mf", "3MF", "Modern printing format, compressed."),
                ("off", "OFF", "Plain text, for academic tooling."),
                ("glb", "GLB — binary glTF", "For visualisation and the web."),
            ],
            "stl",
        )
        form.addRow("Format", self.format_combo)

        self.save_button = QPushButton("Export…")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self._export)
        self.save_button.setEnabled(False)
        form.addRow(self.save_button)

        self.status = InfoLabel("Generate a lattice first.")
        form.addRow(self.status)

        self.add(box)

        stats_box, stats_form = section("Result")
        self.stats = InfoLabel("—")
        stats_form.addRow(self.stats)
        self.add(stats_box)

        self.add(muted(
            "Export always writes the full-resolution mesh, even when the 3D view is "
            "showing a simplified preview."
        ))
        self.add_stretch()

    # -------------------------------------------------------------------- state

    def set_mesh(self, mesh: Mesh | None, summary: str = "") -> None:
        self._mesh = mesh
        has_mesh = mesh is not None and not mesh.is_empty
        self.save_button.setEnabled(has_mesh)

        if not has_mesh:
            self.status.info("Generate a lattice first.")
            self.stats.info("—")
            return

        self.status.info("Ready to export.")
        self.stats.info(summary or mesh.describe())

    # ----------------------------------------------------------------- handlers

    def _export(self) -> None:
        if self._mesh is None or self._mesh.is_empty:
            return

        extension = current_choice(self.format_combo)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export lattice", f"lattice.{extension}", qt_export_filter()
        )
        if not path:
            return

        # Honour the chosen format when the typed name has no extension of its own.
        if not os.path.splitext(path)[1]:
            path = f"{path}.{extension}"

        self.status.info("Writing…")
        self.save_button.setEnabled(False)

        try:
            written = export_mesh(self._mesh, path)
        except ExportError as exc:
            self.status.error(str(exc))
        except Exception as exc:
            self.status.error(f"Export failed: {exc}")
        else:
            size = os.path.getsize(written) / 1e6
            self.status.success(
                f"Wrote {os.path.basename(written)} ({size:.1f} MB, "
                f"{self._mesh.num_faces:,} triangles)."
            )
            self.exported.emit(written)
        finally:
            self.save_button.setEnabled(True)
