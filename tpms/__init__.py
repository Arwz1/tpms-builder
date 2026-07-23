"""TPMS Builder — generate Triply Periodic Minimal Surface lattices inside 3D geometry.

Layering (strictly one-way):

    ui  ->  features  ->  core
            io        ->  core

``core`` imports nothing from this project, which is what allows every feature to be
used from a plain script without pulling in Qt or VTK.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
