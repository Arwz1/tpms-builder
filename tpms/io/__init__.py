"""Geometry import and export.

Importing this package registers every loader. Loaders that need a heavy optional
dependency (OpenCascade, for STEP) register their *metadata* eagerly but import the
dependency only when a file of that type is actually opened, so a missing extra costs
nothing until it is used.
"""

from tpms.io.registry import (
    LoaderInfo,
    UnsupportedFormatError,
    load,
    loader_for,
    qt_file_filter,
    register_loader,
    registered_loaders,
    supported_extensions,
)

# Import for the registration side effect. Order sets the file-dialog filter order.
from tpms.io import mesh_loader as _mesh_loader  # noqa: F401
from tpms.io import step_loader as _step_loader  # noqa: F401

from tpms.io.exporters import (
    ExportError,
    export_mesh,
    export_extensions,
    qt_export_filter,
)

__all__ = [
    "LoaderInfo",
    "UnsupportedFormatError",
    "load",
    "loader_for",
    "qt_file_filter",
    "register_loader",
    "registered_loaders",
    "supported_extensions",
    "ExportError",
    "export_mesh",
    "export_extensions",
    "qt_export_filter",
]
