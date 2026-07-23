"""Mesh export.

STL is the workhorse for print-ready lattices; the rest are there because a lattice
often needs to go back into another tool. Binary STL is written directly rather than
through trimesh, because a multi-million-triangle lattice is exactly where the
round-trip through an intermediate object costs the most memory.
"""

from __future__ import annotations

import os
import struct

import numpy as np

from tpms.core.mesh import Mesh


class ExportError(RuntimeError):
    """Writing the mesh failed."""


EXPORT_EXTENSIONS = ("stl", "obj", "ply", "3mf", "off", "glb")


def export_extensions() -> tuple[str, ...]:
    return EXPORT_EXTENSIONS


def _write_binary_stl(mesh: Mesh, path: str) -> None:
    """Write binary STL without materialising the whole buffer at once.

    The format is 80 bytes of header, a uint32 count, then 50 bytes per triangle:
    12 floats (normal + 3 corners) and a 2-byte attribute word.
    """
    triangles = mesh.vertices[mesh.faces].astype(np.float32)
    normals = mesh.face_normals().astype(np.float32)
    count = len(triangles)

    # One structured array, written in chunks so peak memory stays bounded.
    record = np.dtype(
        [("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")]
    )

    chunk_size = 200_000

    with open(path, "wb") as handle:
        handle.write(b"TPMS Builder binary STL".ljust(80, b"\0"))
        handle.write(struct.pack("<I", count))

        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            block = np.empty(stop - start, dtype=record)
            block["normal"] = normals[start:stop]
            block["corners"] = triangles[start:stop]
            block["attr"] = 0
            handle.write(block.tobytes())


def _write_via_trimesh(mesh: Mesh, path: str, extension: str) -> None:
    try:
        import trimesh
    except ImportError as exc:
        raise ExportError(
            f"writing .{extension} needs trimesh. Run: pip install trimesh"
        ) from exc

    tm = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces, process=False, validate=False
    )
    try:
        tm.export(path)
    except Exception as exc:
        raise ExportError(f"trimesh could not write '{path}': {exc}") from exc


def export_mesh(mesh: Mesh, path: str | os.PathLike[str]) -> str:
    """Write ``mesh`` to ``path``, choosing the writer from the extension.

    Returns the path written. Raises :class:`ExportError` on any failure.
    """
    path = str(path)

    if mesh.is_empty:
        raise ExportError("nothing to export: the mesh is empty")

    extension = os.path.splitext(path)[1].lower().lstrip(".")
    if not extension:
        extension = "stl"
        path = f"{path}.stl"

    if extension not in EXPORT_EXTENSIONS:
        raise ExportError(
            f"cannot write '.{extension}'. Supported: "
            f"{', '.join('.' + e for e in EXPORT_EXTENSIONS)}"
        )

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        if extension == "stl":
            _write_binary_stl(mesh, path)
        else:
            _write_via_trimesh(mesh, path, extension)
    except ExportError:
        raise
    except OSError as exc:
        raise ExportError(f"could not write '{path}': {exc}") from exc

    return path


def qt_export_filter() -> str:
    """File-dialog filter for the export panel."""
    labels = {
        "stl": "STL — binary, for slicers and printers",
        "obj": "OBJ — Wavefront",
        "ply": "PLY — Stanford",
        "3mf": "3MF — 3D Manufacturing Format",
        "off": "OFF — Object File Format",
        "glb": "GLB — binary glTF",
    }
    return ";;".join(f"{labels[e]} (*.{e})" for e in EXPORT_EXTENSIONS)
