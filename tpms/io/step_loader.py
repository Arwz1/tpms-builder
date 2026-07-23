"""STEP / IGES import via OpenCascade.

BREP surfaces have to be tessellated before anything downstream can use them, so this
module reads the shape, meshes it to a deflection tolerance derived from the model size,
and hands back triangles.

OpenCascade is a large optional dependency. It is imported inside the functions, never
at module scope, so the format still appears in the registry — greyed out with an
install hint — when it is absent.
"""

from __future__ import annotations

import os

import numpy as np

from tpms.core.mesh import Mesh
from tpms.io.registry import LoaderInfo, register_loader

STEP_EXTENSIONS = ("step", "stp", "iges", "igs")

# Tessellation deflection as a fraction of the model's bounding-box diagonal. The
# lattice pipeline resamples onto a voxel grid anyway, so chasing a finer tessellation
# than the grid can represent only costs time.
DEFAULT_DEFLECTION_RATIO = 1.0 / 2000.0
DEFAULT_ANGULAR_DEFLECTION = 0.35  # radians


def _ocp_missing() -> str | None:
    try:
        import OCP  # noqa: F401
        from OCP.STEPControl import STEPControl_Reader  # noqa: F401
    except ImportError:
        return (
            "OpenCascade is not installed. Run: pip install -r requirements-step.txt "
            "(or: pip install cadquery-ocp)"
        )
    return None


def _read_shape(path: str):
    """Read a STEP or IGES file into a single TopoDS_Shape."""
    from OCP.IFSelect import IFSelect_ReturnStatus

    extension = os.path.splitext(path)[1].lower().lstrip(".")

    if extension in ("step", "stp"):
        from OCP.STEPControl import STEPControl_Reader
        reader = STEPControl_Reader()
        label = "STEP"
    else:
        from OCP.IGESControl import IGESControl_Reader
        reader = IGESControl_Reader()
        label = "IGES"

    status = reader.ReadFile(path)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(
            f"OpenCascade could not read '{os.path.basename(path)}' as {label} "
            f"(status {status})"
        )

    reader.TransferRoots()

    if reader.NbShapes() == 0:
        raise ValueError(f"'{os.path.basename(path)}' contained no solids or surfaces")

    return reader.OneShape()


def _bbox_diagonal(shape) -> float:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return 1.0

    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return float(
        np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    ) or 1.0


def _triangulate(shape, linear_deflection: float, angular_deflection: float) -> Mesh:
    """Tessellate every face and concatenate the triangles."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_Orientation
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(
        shape, linear_deflection, False, angular_deflection, True
    )

    vertex_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)

        if triangulation is None:
            explorer.Next()
            continue

        transform = location.Transformation()

        n_nodes = triangulation.NbNodes()
        nodes = np.empty((n_nodes, 3), dtype=np.float64)
        for i in range(1, n_nodes + 1):
            point = triangulation.Node(i).Transformed(transform)
            nodes[i - 1] = (point.X(), point.Y(), point.Z())

        n_tris = triangulation.NbTriangles()
        tris = np.empty((n_tris, 3), dtype=np.int64)
        for i in range(1, n_tris + 1):
            a, b, c = triangulation.Triangle(i).Get()
            tris[i - 1] = (a - 1, b - 1, c - 1)  # OCCT indices are 1-based

        # A reversed face stores its triangles in the surface's own winding, so the
        # outward normal is the other way round. Left unflipped, the imported solid
        # comes out inside-out and the voxeliser fills the wrong region.
        if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
            tris = tris[:, ::-1]

        vertex_blocks.append(nodes)
        face_blocks.append(tris + offset)
        offset += n_nodes

        explorer.Next()

    if not vertex_blocks:
        return Mesh.empty()

    return Mesh(np.concatenate(vertex_blocks), np.concatenate(face_blocks))


def load_step(
    path: str,
    linear_deflection: float | None = None,
    angular_deflection: float = DEFAULT_ANGULAR_DEFLECTION,
    **_options,
) -> Mesh:
    """Read a STEP or IGES file and return its tessellation.

    ``linear_deflection`` is the maximum gap between the true surface and its
    tessellation, in model units. It defaults to a fraction of the bounding-box
    diagonal, which keeps the triangle count sane on large models and the curvature
    faithful on small ones.
    """
    shape = _read_shape(path)

    if linear_deflection is None:
        linear_deflection = _bbox_diagonal(shape) * DEFAULT_DEFLECTION_RATIO

    mesh = _triangulate(shape, linear_deflection, angular_deflection)

    if mesh.is_empty:
        raise ValueError(
            f"'{os.path.basename(path)}' produced no triangles. The file may contain "
            "only curves or reference geometry."
        )

    # OCCT emits per-face vertex blocks, so shared edges arrive duplicated. Welding is
    # what makes the result watertight enough for the voxeliser to sign correctly.
    return mesh.weld(tolerance=max(linear_deflection * 0.1, 1e-9))


register_loader(
    LoaderInfo(
        name="CAD solid",
        extensions=STEP_EXTENSIONS,
        load=load_step,
        description="Boundary-representation CAD: STEP, IGES",
        availability_check=_ocp_missing,
    )
)
