"""Background generation on a QThread.

A 512³ run takes the better part of a minute. On the GUI thread that is an unresponsive
window and, on Windows, a "not responding" ghost. So generation runs on a worker and
talks back through signals.

Cancellation is cooperative: the pipeline calls a progress callback between slabs, and
returning ``False`` from it raises
:class:`~tpms.core.marching.GenerationCancelled` inside the worker. That keeps
cancellation at slab granularity — worst case a fraction of a second — without any
thread killing.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal

from tpms.core.marching import GenerationCancelled
from tpms.core.mesh import Mesh
from tpms.features.generate import GenerationParams, GenerationResult, generate


class GenerationWorker(QObject):
    """Runs one generation, then stops. Lives on its own thread."""

    progressed = Signal(float, str)
    finished = Signal(object)      # GenerationResult
    failed = Signal(str, str)      # message, traceback
    cancelled = Signal()

    def __init__(self, params: GenerationParams, mesh: Mesh | None = None) -> None:
        super().__init__()
        self._params = params
        self._mesh = mesh
        self._cancel = False

    def cancel(self) -> None:
        """Ask the run to stop. Safe to call from the GUI thread."""
        self._cancel = True

    def _progress(self, fraction: float, message: str) -> bool:
        if self._cancel:
            return False
        self.progressed.emit(float(fraction), str(message))
        return True

    def run(self) -> None:
        try:
            result = generate(self._params, mesh=self._mesh, progress=self._progress)
        except GenerationCancelled:
            self.cancelled.emit()
        except MemoryError:
            self.failed.emit(
                "Ran out of memory. Lower the resolution, or increase the cell size "
                "so fewer triangles are produced.",
                traceback.format_exc(),
            )
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
        else:
            self.finished.emit(result)


class GenerationController(QObject):
    """Owns the worker and its thread, and guarantees only one run at a time.

    Qt object lifetime across threads is easy to get wrong — a worker garbage-collected
    while its thread is still running is a crash, not an exception. Keeping both
    references here, and only clearing them once ``QThread.finished`` has fired, is what
    makes repeated generation safe.
    """

    progressed = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    started = Signal()
    stopped = Signal()          # any terminal outcome, for re-enabling the UI

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: GenerationWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, params: GenerationParams, mesh: Mesh | None = None) -> bool:
        """Begin a run. Returns ``False`` if one is already going."""
        if self.is_running:
            return False

        thread = QThread()
        worker = GenerationWorker(params, mesh)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)

        worker.progressed.connect(self.progressed)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)

        # Every terminal signal quits the event loop so the thread can wind down.
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)

        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker

        thread.start()
        self.started.emit()
        return True

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def wait(self, milliseconds: int = 30_000) -> None:
        """Block until the run ends. For application shutdown only."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(milliseconds)

    # ----------------------------------------------------------------- handlers

    def _on_finished(self, result: GenerationResult) -> None:
        self.finished.emit(result)

    def _on_failed(self, message: str, detail: str) -> None:
        self.failed.emit(message, detail)

    def _on_cancelled(self) -> None:
        self.cancelled.emit()

    def _on_thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.stopped.emit()
