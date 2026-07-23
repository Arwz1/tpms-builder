"""3D viewport, backed by PyVista/VTK.

Falls back to a plain message panel when VTK cannot start — a headless session, a
machine with no usable OpenGL, or a remote desktop without 3D. The rest of the
application keeps working in that case, because everything it actually needs (generate,
inspect statistics, export) is independent of the view.
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from tpms.core.mesh import Mesh
from tpms.ui import theme

# Above this, rendering the full lattice makes orbiting stutter. The viewport shows a
# decimated copy and says so; export always uses the full-resolution mesh.
PREVIEW_TRIANGLE_LIMIT = 3_000_000


def _pyvista_available() -> tuple[bool, str]:
    """Decide whether it is safe to build a VTK render window.

    This has to be settled *before* constructing the interactor, not with a try/except
    around it. When VTK cannot obtain a pixel format — a headless session, a VM with no
    3D driver, an RDP connection, or Qt's own offscreen platform — it does not raise:
    it logs "failed to get valid pixel format" and then segfaults inside C++, which no
    Python handler can intercept. A crash on start is a far worse failure than a
    missing preview, so the risky cases are ruled out in advance.
    """
    if os.environ.get("TPMS_NO_3D"):
        return False, "Disabled by the TPMS_NO_3D environment variable."

    platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    if any(name in platform for name in ("offscreen", "minimal", "vnc")):
        return False, (
            f"Qt is running on the '{platform}' platform, which has no OpenGL surface."
        )

    try:
        import pyvista  # noqa: F401
        import pyvistaqt  # noqa: F401
    except ImportError as exc:
        return False, f"PyVista is not installed ({exc})."

    ok, reason = _opengl_usable()
    if not ok:
        return False, reason

    return True, ""


def _opengl_usable() -> tuple[bool, str]:
    """Ask Qt whether a real OpenGL context can be created and made current."""
    try:
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
    except ImportError as exc:  # pragma: no cover - Qt is a hard dependency
        return False, f"Qt OpenGL support is unavailable ({exc})."

    context = QOpenGLContext()
    if not context.create():
        return False, "No OpenGL context could be created on this display."

    surface = QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()

    if not surface.isValid():
        return False, "No usable rendering surface is available."

    if not context.makeCurrent(surface):
        return False, "An OpenGL context was created but could not be made current."

    context.doneCurrent()
    return True, ""


class Viewport(QWidget):
    """Embeds a PyVista render window, or explains why it could not."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plotter = None
        self._mesh_actor = None
        self._source_actor = None
        self._available = False
        self._decimated_note = ""

        available, reason = _pyvista_available()
        if not available:
            layout.addWidget(self._fallback_label(reason))
            return

        try:
            from pyvistaqt import QtInteractor

            self._plotter = QtInteractor(self)
            layout.addWidget(self._plotter.interactor)
            self._available = True
            self._setup_scene()
        except Exception as exc:  # pragma: no cover - depends on the graphics stack
            self._plotter = None
            layout.addWidget(
                self._fallback_label(
                    f"The 3D view could not start ({exc}).\n\n"
                    "Generation and export still work."
                )
            )

    # -------------------------------------------------------------------- setup

    def _fallback_label(self, reason: str) -> QLabel:
        label = QLabel(
            f"3D preview unavailable\n\n{reason}\n\n"
            "Everything else — generating, statistics and export — is unaffected."
        )
        label.setWordWrap(True)
        label.setObjectName("muted")
        label.setStyleSheet("padding: 40px;")
        from PySide6.QtCore import Qt

        label.setAlignment(Qt.AlignCenter)
        return label

    def _setup_scene(self) -> None:
        p = self._plotter
        p.set_background(
            theme.VIEWPORT_BACKGROUND, top=theme.VIEWPORT_BACKGROUND_TOP
        )
        p.add_axes(interactive=False)
        p.enable_anti_aliasing("fxaa")

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------ display

    def _to_polydata(self, mesh: Mesh):
        import pyvista as pv

        # PyVista wants a face array of [3, i, j, k] per triangle.
        faces = np.empty((len(mesh.faces), 4), dtype=np.int64)
        faces[:, 0] = 3
        faces[:, 1:] = mesh.faces

        return pv.PolyData(
            np.ascontiguousarray(mesh.vertices, dtype=np.float32), faces.ravel()
        )

    def show_mesh(self, mesh: Mesh, reset_camera: bool = True) -> str:
        """Display a generated lattice. Returns a note about any decimation applied."""
        if not self._available or mesh.is_empty:
            return ""

        polydata = self._to_polydata(mesh)
        self._decimated_note = ""

        if mesh.num_faces > PREVIEW_TRIANGLE_LIMIT:
            target = 1.0 - PREVIEW_TRIANGLE_LIMIT / mesh.num_faces
            try:
                polydata = polydata.decimate_pro(
                    target, preserve_topology=True, progress_bar=False
                )
                self._decimated_note = (
                    f"Preview decimated to {polydata.n_faces_strict:,} triangles; "
                    f"export writes the full {mesh.num_faces:,}."
                )
            except Exception:
                # Decimation is a convenience. If VTK cannot do it, show everything and
                # accept the slower interaction rather than showing nothing.
                self._decimated_note = ""

        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor, reset_camera=False)

        self._mesh_actor = self._plotter.add_mesh(
            polydata,
            color=theme.MESH_COLOUR,
            smooth_shading=True,
            specular=0.3,
            specular_power=15,
            show_edges=False,
        )

        if reset_camera:
            self._plotter.reset_camera()
        self._plotter.render()

        return self._decimated_note

    def show_source(self, mesh: Mesh | None) -> None:
        """Show the imported geometry as a translucent cage around the lattice."""
        if not self._available:
            return

        if self._source_actor is not None:
            self._plotter.remove_actor(self._source_actor, reset_camera=False)
            self._source_actor = None

        if mesh is None or mesh.is_empty:
            self._plotter.render()
            return

        self._source_actor = self._plotter.add_mesh(
            self._to_polydata(mesh),
            color=theme.SOURCE_COLOUR,
            opacity=0.15,
            smooth_shading=True,
            show_edges=False,
        )
        self._plotter.render()

    def set_source_visible(self, visible: bool) -> None:
        if self._source_actor is not None:
            self._source_actor.SetVisibility(bool(visible))
            self._plotter.render()

    def clear(self) -> None:
        if not self._available:
            return
        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor, reset_camera=False)
            self._mesh_actor = None
        if self._source_actor is not None:
            self._plotter.remove_actor(self._source_actor, reset_camera=False)
            self._source_actor = None
        self._plotter.render()

    # -------------------------------------------------------------------- views

    def reset_camera(self) -> None:
        if self._available:
            self._plotter.reset_camera()
            self._plotter.render()

    def view_isometric(self) -> None:
        if self._available:
            self._plotter.view_isometric()
            self._plotter.render()

    def view_along(self, axis: str) -> None:
        if not self._available:
            return
        {
            "x": self._plotter.view_yz,
            "y": self._plotter.view_xz,
            "z": self._plotter.view_xy,
        }[axis]()
        self._plotter.render()

    def screenshot(self, path: str) -> None:
        if self._available:
            self._plotter.screenshot(path)

    def close_viewport(self) -> None:
        """Release VTK resources. Called on window close.

        Without this, a PyVista interactor can keep the process alive after the window
        has gone.
        """
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
            self._plotter = None
            self._available = False
