"""STL / OBJ / PLY / 3MF import via trimesh.

Scenes with several parts are merged into one mesh: the pipeline fills a single solid
domain, so an assembly is treated as the union of its bodies.
"""

from __future__ import annotations

import os

import numpy as np

from tpms.core.mesh import Mesh
from tpms.io.registry import LoaderInfo, register_loader

MESH_EXTENSIONS = ("stl", "obj", "ply", "3mf", "off", "glb", "gltf")


def _trimesh_missing() -> str | None:
    try:
        import trimesh  # noqa: F401
    except ImportError:
        return "trimesh is not installed. Run: pip install trimesh"
    return None


def _to_mesh(obj) -> Mesh:
    """Flatten a trimesh Trimesh or Scene into our container."""
    import trimesh

    if isinstance(obj, trimesh.Scene):
        geometries = [
            g for g in obj.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)
        ]
        if not geometries:
            return Mesh.empty()
        # dump() has already baked each node's transform into its vertices.
        obj = trimesh.util.concatenate(geometries)

    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"unsupported geometry type: {type(obj).__name__}")

    return Mesh(np.asarray(obj.vertices, dtype=np.float64),
                np.asarray(obj.faces, dtype=np.int64))


def load_mesh(path: str, merge_vertices: bool = True, **_options) -> Mesh:
    """Read a polygon mesh file.

    ``merge_vertices`` welds the split vertices that STL always produces. Without it a
    facetted STL has three unshared vertices per triangle, which breaks the watertight
    check the voxeliser relies on to decide inside from outside.
    """
    import trimesh

    loaded = trimesh.load(
        path,
        force="mesh" if os.path.splitext(path)[1].lower() != ".3mf" else None,
        process=merge_vertices,
        skip_materials=True,
    )

    mesh = _to_mesh(loaded)

    if mesh.is_empty:
        raise ValueError(f"'{os.path.basename(path)}' contained no triangles")

    return mesh


register_loader(
    LoaderInfo(
        name="Polygon mesh",
        extensions=MESH_EXTENSIONS,
        load=load_mesh,
        description="Triangle meshes: STL, OBJ, PLY, 3MF, OFF, glTF",
        availability_check=_trimesh_missing,
    )
)
