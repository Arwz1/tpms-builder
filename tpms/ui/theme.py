"""Dark theme.

A CAD viewport is judged against its background, and a light chrome around a dark 3D
view makes shaded geometry hard to read. One place for the palette so panels stay
consistent.
"""

from __future__ import annotations

BACKGROUND = "#1e1f22"
SURFACE = "#2b2d30"
SURFACE_RAISED = "#35373b"
BORDER = "#43454a"
TEXT = "#dfe1e5"
TEXT_MUTED = "#9da0a8"
ACCENT = "#4a9eff"
ACCENT_PRESSED = "#3d84d6"
WARNING = "#e8a33d"
ERROR = "#e5534b"
SUCCESS = "#57ab5a"

#: Viewport background, as an RGB triple in 0..1 for VTK.
VIEWPORT_BACKGROUND = (0.12, 0.12, 0.14)
VIEWPORT_BACKGROUND_TOP = (0.18, 0.19, 0.22)

#: Default colour for generated lattice geometry.
MESH_COLOUR = "#8ab4f8"
#: Colour for the imported source geometry shown as a translucent cage.
SOURCE_COLOUR = "#6b7280"


STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 13px;
}}

QMainWindow::separator {{
    background-color: {BORDER};
    width: 1px;
    height: 1px;
}}

QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-weight: 600;
}}
QDockWidget::title {{
    background-color: {SURFACE};
    padding: 7px 10px;
    border-bottom: 1px solid {BORDER};
}}

QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {TEXT_MUTED};
}}

QScrollArea {{ border: none; }}

QPushButton {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 14px;
    min-height: 18px;
}}
QPushButton:hover {{ background-color: #3f4247; }}
QPushButton:pressed {{ background-color: #303236; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; background-color: #2a2c2f; }}

QPushButton#primary {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: #08111f;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: #5aa8ff; }}
QPushButton#primary:pressed {{ background-color: {ACCENT_PRESSED}; }}
QPushButton#primary:disabled {{ background-color: #35414f; border-color: #35414f; color: {TEXT_MUTED}; }}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

QLineEdit[invalid="true"] {{ border-color: {ERROR}; }}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}

QProgressBar {{
    background-color: {BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 5px;
    text-align: center;
    height: 18px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 4px; }}

QStatusBar {{
    background-color: {SURFACE};
    border-top: 1px solid {BORDER};
}}
QStatusBar::item {{ border: none; }}

QMenuBar {{ background-color: {SURFACE}; }}
QMenuBar::item:selected {{ background-color: {SURFACE_RAISED}; }}
QMenu {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{ background-color: {ACCENT}; color: #08111f; }}

QLabel#heading {{ font-weight: 600; color: {TEXT}; }}
QLabel#muted {{ color: {TEXT_MUTED}; }}
QLabel#warning {{ color: {WARNING}; }}
QLabel#error {{ color: {ERROR}; }}
QLabel#success {{ color: {SUCCESS}; }}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BACKGROUND};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 5px; }}
QTabBar::tab {{
    background: transparent;
    padding: 7px 14px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ border-bottom-color: {ACCENT}; color: {TEXT}; }}
QTabBar::tab:!selected {{ color: {TEXT_MUTED}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #55575d; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 24px; }}

QToolTip {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px;
}}
"""


def apply_theme(app) -> None:
    """Apply the palette and stylesheet to a ``QApplication``."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(BACKGROUND))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#08111f"))
    palette.setColor(QPalette.ToolTipBase, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    app.setPalette(palette)

    app.setStyleSheet(STYLESHEET)
