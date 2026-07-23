"""Main window — docks, menus, and the generation lifecycle.

The only place that knows about both the UI and the pipeline. Panels describe
parameters; this decides when to run, what to show, and how to recover from failure.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tpms import __version__
from tpms.core.mesh import Mesh
from tpms.features.generate import GenerationParams, GenerationResult, SourceType
from tpms.ui import theme
from tpms.ui.panels import (
    ExportPanel,
    GradingPanel,
    QualityPanel,
    SourcePanel,
    TpmsPanel,
)
from tpms.ui.viewport import Viewport
from tpms.ui.worker import GenerationController

#: Delay before an auto-preview fires, so dragging a spin box does not queue a run per
#: keystroke.
PREVIEW_DELAY_MS = 400


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(f"TPMS Builder {__version__}")
        self.resize(1500, 950)

        self._source_mesh: Mesh | None = None
        self._result: GenerationResult | None = None
        self._auto_preview = False

        self.controller = GenerationController(self)
        self._connect_controller()

        self._build_ui()
        self._build_menus()
        self._connect_panels()

        self._refresh_estimates()

    # ------------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.source_panel = SourcePanel()
        self.tpms_panel = TpmsPanel()
        self.grading_panel = GradingPanel()
        self.quality_panel = QualityPanel()
        self.export_panel = ExportPanel()

        for panel, title in (
            (self.source_panel, "Source"),
            (self.tpms_panel, "Pattern"),
            (self.grading_panel, "Grading"),
            (self.quality_panel, "Quality"),
            (self.export_panel, "Export"),
        ):
            self.tabs.addTab(self._scrollable(panel), title)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(self.tabs, 1)
        side_layout.addWidget(self._build_actions())

        from PySide6.QtWidgets import QDockWidget

        dock = QDockWidget("Parameters", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setWidget(side)
        dock.setMinimumWidth(370)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._dock = dock

        self._build_status_bar()

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        area.setFrameShape(QScrollArea.NoFrame)
        return area

    def _build_actions(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(7)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self.generate)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.controller.cancel)
        self.cancel_button.setVisible(False)

        row.addWidget(self.generate_button, 2)
        row.addWidget(self.cancel_button, 1)
        layout.addLayout(row)

        return holder

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.status_label = QLabel("Ready.")
        bar.addWidget(self.status_label, 1)

        self.result_label = QLabel("")
        self.result_label.setObjectName("muted")
        bar.addPermanentWidget(self.result_label)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Import geometry…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._import_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        load_action = QAction("&Load settings…", self)
        load_action.triggered.connect(self._load_settings)
        file_menu.addAction(load_action)

        save_action = QAction("&Save settings…", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_settings)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ---- generate -------------------------------------------------------
        run_menu = self.menuBar().addMenu("&Generate")

        run_action = QAction("&Generate now", self)
        run_action.setShortcut("Ctrl+Return")
        run_action.triggered.connect(self.generate)
        run_menu.addAction(run_action)

        self.auto_action = QAction("&Auto-preview on change", self)
        self.auto_action.setCheckable(True)
        self.auto_action.toggled.connect(self._set_auto_preview)
        run_menu.addAction(self.auto_action)

        # ---- view -----------------------------------------------------------
        view_menu = self.menuBar().addMenu("&View")

        for label, shortcut, handler in (
            ("&Isometric", "1", self.viewport.view_isometric),
            ("Front (&-Y)", "2", lambda: self.viewport.view_along("y")),
            ("Right (&X)", "3", lambda: self.viewport.view_along("x")),
            ("Top (&Z)", "4", lambda: self.viewport.view_along("z")),
        ):
            action = QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            view_menu.addAction(action)

        view_menu.addSeparator()

        self.show_source_action = QAction("Show &source geometry", self)
        self.show_source_action.setCheckable(True)
        self.show_source_action.setChecked(True)
        self.show_source_action.toggled.connect(self.viewport.set_source_visible)
        view_menu.addAction(self.show_source_action)

        screenshot_action = QAction("Save s&creenshot…", self)
        screenshot_action.triggered.connect(self._screenshot)
        view_menu.addAction(screenshot_action)

        # ---- help -----------------------------------------------------------
        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------ wiring

    def _connect_controller(self) -> None:
        self.controller.started.connect(self._on_started)
        self.controller.progressed.connect(self._on_progress)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.controller.cancelled.connect(self._on_cancelled)
        self.controller.stopped.connect(self._on_stopped)

    def _connect_panels(self) -> None:
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self._auto_generate)

        for panel in (
            self.source_panel, self.tpms_panel,
            self.grading_panel, self.quality_panel,
        ):
            panel.changed.connect(self._on_parameters_changed)

        self.source_panel.file_requested.connect(self._import_file)

    # ---------------------------------------------------------------- parameters

    def current_params(self) -> GenerationParams:
        params = GenerationParams()
        self.source_panel.apply_to(params)
        self.tpms_panel.apply_to(params)
        self.grading_panel.apply_to(params)
        self.quality_panel.apply_to(params)
        return params

    def _on_parameters_changed(self) -> None:
        self._refresh_estimates()
        if self._auto_preview:
            self._preview_timer.start()

    def _refresh_estimates(self) -> None:
        cell = self.grading_panel.representative_cell()
        wall = self.grading_panel.representative_wall()

        params = self.current_params()
        lower, upper = self._domain_bounds(params)

        self.quality_panel.update_estimate(lower, upper, cell, params.mode)
        self.tpms_panel.update_estimate(cell, wall)

        message = self.quality_panel.check_wall_resolution(wall)
        if message:
            self.status_label.setText(message)
        elif not self.controller.is_running:
            self.status_label.setText("Ready.")

    def _domain_bounds(self, params: GenerationParams):
        """Bounds of whatever is currently selected as the source."""
        if params.source_type is SourceType.FILE and self._source_mesh is not None:
            return self._source_mesh.bounds

        if params.source_type is SourceType.SHAPE:
            try:
                from tpms.features.shapes import get_shape

                return get_shape(params.shape_name, **params.shape_params).bounds()
            except Exception:
                pass

        return (-30.0, -30.0, -30.0), (30.0, 30.0, 30.0)

    # -------------------------------------------------------------------- import

    def _import_dialog(self) -> None:
        from tpms.io import qt_file_filter

        path, _ = QFileDialog.getOpenFileName(
            self, "Import geometry", "", qt_file_filter()
        )
        if path:
            self.source_panel.set_file(path)
            self._import_file(path)

    def _import_file(self, path: str) -> None:
        from tpms.io import load

        self.status_label.setText(f"Loading {os.path.basename(path)}…")
        # Let the label paint before a large STEP file blocks the thread.
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        try:
            mesh = load(path)
        except Exception as exc:
            self._source_mesh = None
            self.source_panel.report_import(str(exc), ok=False)
            self.status_label.setText("Import failed.")
            QMessageBox.warning(self, "Import failed", str(exc))
            return

        self._source_mesh = mesh
        lower, upper = mesh.bounds
        extent = upper - lower

        self.source_panel.report_import(
            f"{mesh.num_faces:,} triangles\n"
            f"{extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm"
        )
        self.grading_panel.suggest_bounds(lower, upper)

        self.viewport.show_source(mesh)
        self.viewport.reset_camera()

        self.status_label.setText(f"Loaded {os.path.basename(path)}.")
        self._refresh_estimates()

    # ----------------------------------------------------------------- generate

    def generate(self) -> None:
        if self.controller.is_running:
            return

        params = self.current_params()

        if params.source_type is SourceType.FILE and self._source_mesh is None:
            QMessageBox.information(
                self, "No geometry",
                "Import a geometry file first, or switch the source to a base shape.",
            )
            return

        if not self.grading_panel.is_valid:
            QMessageBox.warning(
                self, "Invalid expression",
                "Fix the grading expression before generating.",
            )
            return

        problems = params.validate()
        if problems:
            QMessageBox.warning(self, "Cannot generate", "\n".join(problems))
            return

        self.controller.start(params, mesh=self._source_mesh)

    def _auto_generate(self) -> None:
        """Auto-preview at reduced resolution, so it stays interactive."""
        if self.controller.is_running:
            # Something is already running; try again once it finishes.
            self._preview_timer.start()
            return

        params = self.current_params()
        if params.source_type is SourceType.FILE and self._source_mesh is None:
            return
        if not self.grading_panel.is_valid or params.validate():
            return

        from tpms.features.generate import preview_params

        self.controller.start(preview_params(params), mesh=self._source_mesh)

    def _set_auto_preview(self, enabled: bool) -> None:
        self._auto_preview = bool(enabled)
        if enabled:
            self._preview_timer.start()

    # ------------------------------------------------------------- run handlers

    def _on_started(self) -> None:
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.generate_button.setVisible(False)
        self.cancel_button.setVisible(True)

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(fraction * 100))
        self.status_label.setText(message)

    def _on_finished(self, result: GenerationResult) -> None:
        self._result = result

        if result.mesh.is_empty:
            self.status_label.setText("The result was empty.")
            self.export_panel.set_mesh(None)
            QMessageBox.information(
                self, "Nothing generated",
                "\n\n".join(result.warnings)
                or "The lattice came out empty. Try a thicker wall or a higher "
                   "resolution.",
            )
            return

        note = self.viewport.show_mesh(result.mesh)
        self.viewport.set_source_visible(self.show_source_action.isChecked())

        summary = result.summary()
        self.export_panel.set_mesh(result.mesh, summary)

        self.result_label.setText(
            f"{result.mesh.num_faces:,} triangles   "
            f"{result.relative_density * 100:.1f}% dense   "
            f"{result.seconds:.1f}s"
        )

        status = f"Done in {result.seconds:.1f} s."
        if note:
            status = f"{status}  {note}"
        if result.warnings:
            status = f"{status}  {result.warnings[0]}"
        self.status_label.setText(status)

    def _on_failed(self, message: str, detail: str) -> None:
        self.status_label.setText("Generation failed.")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Generation failed")
        box.setText(message)
        box.setDetailedText(detail)
        box.exec()

    def _on_cancelled(self) -> None:
        self.status_label.setText("Cancelled.")

    def _on_stopped(self) -> None:
        self.progress.setVisible(False)
        self.generate_button.setVisible(True)
        self.cancel_button.setVisible(False)

    # ----------------------------------------------------------------- settings

    def _save_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save settings", "tpms-settings.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.current_params().save(path)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
        else:
            self.status_label.setText(f"Saved {os.path.basename(path)}.")

    def _load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            params = GenerationParams.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not load", str(exc))
            return

        self.apply_params(params)
        self.status_label.setText(f"Loaded {os.path.basename(path)}.")

    def apply_params(self, params: GenerationParams) -> None:
        """Push a full parameter set into every panel."""
        self.source_panel.load_from(params)
        self.tpms_panel.load_from(params)
        self.grading_panel.load_from(params)
        self.quality_panel.load_from(params)

        if params.source_type is SourceType.FILE and params.source_path:
            if os.path.isfile(params.source_path):
                self._import_file(params.source_path)

        self._refresh_estimates()

    # -------------------------------------------------------------------- misc

    def _screenshot(self) -> None:
        if not self.viewport.is_available:
            QMessageBox.information(
                self, "No 3D view", "The 3D viewport is not available."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", "tpms.png", "PNG (*.png)"
        )
        if path:
            self.viewport.screenshot(path)
            self.status_label.setText(f"Saved {os.path.basename(path)}.")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About TPMS Builder",
            f"<b>TPMS Builder {__version__}</b>"
            "<p>Generates triply periodic minimal surface lattices inside imported "
            "or generated geometry.</p>"
            "<p>Patterns: gyroid, Schwarz P, Schwarz D, Neovius.<br>"
            "Import: STL, OBJ, PLY, 3MF, STEP, IGES.</p>",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.controller.is_running:
            self.controller.cancel()
            self.controller.wait(5000)
        self.viewport.close_viewport()
        super().closeEvent(event)
