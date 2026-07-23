"""Application bootstrap."""

from __future__ import annotations

import os
import sys


def _configure_environment() -> None:
    """Environment settings that must be in place before Qt or VTK start."""
    # Qt reads this at QApplication construction, so it is too late in main().
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    # VTK's OpenGL probe writes a warning window on some drivers; suppress it so a
    # driver quirk does not put a stray window in front of the app.
    os.environ.setdefault("VTK_SILENCE_GET_VOID_POINTER_WARNINGS", "1")


def main(argv: list[str] | None = None) -> int:
    """Start the GUI. Returns the process exit code."""
    _configure_environment()

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("TPMS Builder")
    app.setOrganizationName("TPMS Builder")

    from tpms.ui.theme import apply_theme

    apply_theme(app)

    from tpms.ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    # A settings file or geometry file may be passed on the command line.
    arguments = (argv if argv is not None else sys.argv)[1:]
    for argument in arguments:
        if not os.path.isfile(argument):
            continue
        if argument.lower().endswith(".json"):
            from tpms.features.generate import GenerationParams

            try:
                window.apply_params(GenerationParams.load(argument))
            except Exception:
                pass
        else:
            window.source_panel.set_file(argument)
            window._import_file(argument)
        break

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
