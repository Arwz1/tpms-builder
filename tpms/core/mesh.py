"""Minimal triangle mesh container.

Deliberately thin: vertices, faces, and the operations the pipeline actually needs.
Interop with ``trimesh`` happens at the I/O boundary only, so nothing in ``core`` or
``features`` requires trimesh to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


#: Vertex and index dtypes. A fine lattice at 512³ reaches ~28 M triangles, where the
#: mesh — not the field — is what exhausts memory: float64 vertices plus int64 indices
#: cost about 1 GB before any working copy. float32 holds ~1e-5 mm at part scale, far
#: finer than any printer, and int32 indexes 2.1 billion vertices. Halving both is what
#: makes the top of the resolution range usable.
VERTEX_DTYPE = np.float32
INDEX_DTYPE = np.int32


@dataclass
class Mesh:
    """An indexed triangle mesh.

    ``vertices`` is ``(V, 3)`` float32; ``faces`` is ``(F, 3)`` int32 indices.
    """

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=VERTEX_DTYPE)
        f = np.asarray(self.faces, dtype=INDEX_DTYPE)

        if v.size == 0:
            v = np.zeros((0, 3), dtype=VERTEX_DTYPE)
        if f.size == 0:
            f = np.zeros((0, 3), dtype=INDEX_DTYPE)

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3), got {f.shape}")
        if f.size and (f.max() >= len(v) or f.min() < 0):
            raise ValueError("face indices out of range")

        self.vertices = v
        self.faces = f

    # ------------------------------------------------------------------ factories

    @classmethod
    def empty(cls) -> "Mesh":
        return cls(
            np.zeros((0, 3), dtype=VERTEX_DTYPE),
            np.zeros((0, 3), dtype=INDEX_DTYPE),
        )

    @classmethod
    def concatenate(cls, meshes: Iterable["Mesh"]) -> "Mesh":
        """Join meshes, offsetting face indices. Does not weld — call :meth:`weld`.

        Preallocates and fills in place. ``np.concatenate`` on a list would hold the
        inputs and the output at once, doubling peak memory at exactly the point where
        the mesh is already the largest thing in the process.
        """
        meshes = [m for m in meshes if not m.is_empty]
        if not meshes:
            return cls.empty()
        if len(meshes) == 1:
            return meshes[0]

        total_vertices = sum(len(m.vertices) for m in meshes)
        total_faces = sum(len(m.faces) for m in meshes)

        vertices = np.empty((total_vertices, 3), dtype=VERTEX_DTYPE)
        faces = np.empty((total_faces, 3), dtype=INDEX_DTYPE)

        v_at = f_at = 0
        for m in meshes:
            nv, nf = len(m.vertices), len(m.faces)
            vertices[v_at : v_at + nv] = m.vertices
            np.add(m.faces, v_at, out=faces[f_at : f_at + nf], casting="unsafe")
            v_at += nv
            f_at += nf

        return cls(vertices, faces)

    # ----------------------------------------------------------------- properties

    @property
    def is_empty(self) -> bool:
        return len(self.faces) == 0

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_empty and len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def extent(self) -> np.ndarray:
        lower, upper = self.bounds
        return upper - lower

    @property
    def centroid(self) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.zeros(3)
        return self.vertices.mean(axis=0)

    # ------------------------------------------------------------------ measures

    def face_normals(self) -> np.ndarray:
        """Unnormalised-then-normalised per-face normals; zero for degenerates."""
        if self.is_empty:
            return np.zeros((0, 3))
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, length, out=np.zeros_like(n), where=length > 0)

    def _chunks(self, chunk_size: int = 1_000_000):
        """Iterate faces in blocks, upcast to float64.

        Both measures below sum millions of small terms, which float32 accumulation
        turns into visible drift. Upcasting the whole vertex array instead would
        allocate gigabytes on a fine lattice, so it is done a block at a time.
        """
        for start in range(0, len(self.faces), chunk_size):
            block = self.faces[start : start + chunk_size]
            yield self.vertices[block].astype(np.float64, copy=False)

    def area(self) -> float:
        if self.is_empty:
            return 0.0
        total = 0.0
        for tri in self._chunks():
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            total += float(np.linalg.norm(n, axis=1).sum())
        return 0.5 * total

    def volume(self) -> float:
        """Signed volume via the divergence theorem. Meaningful only if watertight."""
        if self.is_empty:
            return 0.0
        total = 0.0
        for tri in self._chunks():
            total += float(
                np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
            )
        return total / 6.0

    # ------------------------------------------------------------------- cleanup

    def weld(self, tolerance: float = 1e-6) -> "Mesh":
        """Merge vertices that coincide within ``tolerance``.

        Marching a grid in slabs produces a duplicate vertex ring at every seam. Welding
        turns those back into a single connected surface, which is what makes the export
        watertight.

        Proximity welding, not quantisation. Rounding coordinates onto a lattice looks
        cheaper but silently fails: scikit-image returns float32 vertices, and the two
        copies of a seam vertex differ in the last bits, so any pair straddling a
        rounding boundary survives as two vertices and leaves a crack. A KD-tree asks
        the question that was actually meant — which vertices are within tolerance of
        each other — and has no boundary to straddle.
        """
        return self.weld_subset(None, tolerance)

    def weld_subset(
        self, candidates: np.ndarray | None, tolerance: float = 1e-6
    ) -> "Mesh":
        """Weld only the vertices in ``candidates`` (an index array), or all if ``None``.

        Slab marching duplicates vertices *only* on the seam planes, and those are a
        thin O(n²) subset of an O(n³) mesh — a few hundred thousand out of fourteen
        million at 512³. Building the KD-tree over everything asks a question whose
        answer is already known for 98 % of the input, and it is the single largest
        allocation in a high-resolution run.
        """
        if len(self.vertices) == 0:
            return self.copy()

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        n = len(self.vertices)

        if candidates is None:
            subset = np.arange(n, dtype=INDEX_DTYPE)
            points = self.vertices
        else:
            subset = np.asarray(candidates, dtype=INDEX_DTYPE)
            if len(subset) == 0:
                return self.remove_degenerate_faces()
            points = self.vertices[subset]

        tree = cKDTree(points)
        local_pairs = tree.query_pairs(r=tolerance, output_type="ndarray")

        if len(local_pairs) == 0:
            return self.remove_degenerate_faces()

        # Map the subset-local indices produced by the tree back to mesh indices.
        pairs = subset[local_pairs]

        # Group mutually-close vertices; a cluster may hold more than two members
        # where several slab corners meet.
        graph = coo_matrix(
            (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
            shape=(n, n),
        )
        _, labels = connected_components(graph, directed=False)

        # Keep the first vertex of each cluster as its representative.
        order = np.argsort(labels, kind="stable")
        first_of_label = np.empty(labels.max() + 1, dtype=INDEX_DTYPE)
        sorted_labels = labels[order]
        boundaries = np.r_[True, sorted_labels[1:] != sorted_labels[:-1]]
        first_of_label[sorted_labels[boundaries]] = order[boundaries]

        keep = np.sort(first_of_label)
        remap = np.empty(n, dtype=INDEX_DTYPE)
        remap[keep] = np.arange(len(keep), dtype=INDEX_DTYPE)

        vertices = self.vertices[keep]
        faces = remap[first_of_label[labels]][self.faces]

        return Mesh(vertices, faces).remove_degenerate_faces()

    def remove_degenerate_faces(self) -> "Mesh":
        """Drop faces with a repeated vertex or zero area."""
        if self.is_empty:
            return self.copy()

        f = self.faces
        distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])

        tri = self.vertices[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        has_area = np.linalg.norm(n, axis=1) > 1e-14

        keep = distinct & has_area
        if keep.all():
            return self.copy()
        return Mesh(self.vertices, f[keep])

    def remove_unreferenced_vertices(self) -> "Mesh":
        """Drop vertices no face points at, reindexing what remains."""
        if self.is_empty:
            return Mesh.empty()

        used = np.zeros(len(self.vertices), dtype=bool)
        used[self.faces.ravel()] = True

        remap = np.full(len(self.vertices), -1, dtype=INDEX_DTYPE)
        remap[used] = np.arange(int(used.sum()), dtype=INDEX_DTYPE)

        return Mesh(self.vertices[used], remap[self.faces])

    def remove_small_components(self, min_faces: int = 12) -> "Mesh":
        """Drop connected components below ``min_faces``.

        A TPMS clipped against a part boundary always sheds a few slivers where a wall
        grazes the surface. They are unprintable and they upset slicers, so they go.

        Connectivity comes from a sparse vertex adjacency graph. A Python union-find
        would need three iterations per face — several million on a fine lattice — so
        the whole thing is handed to scipy instead.
        """
        if self.is_empty or min_faces <= 1:
            return self.copy()

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n = len(self.vertices)
        f = self.faces

        # Two edges per face are enough to connect its three vertices, so the third is
        # redundant work and a third more memory in the graph.
        rows = np.concatenate([f[:, 0], f[:, 1]])
        cols = np.concatenate([f[:, 1], f[:, 2]])

        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
        _, roots = connected_components(adjacency, directed=False)

        face_root = roots[f[:, 0]]

        labels, counts = np.unique(face_root, return_counts=True)
        keep_labels = labels[counts >= min_faces]
        if len(keep_labels) == 0:
            # Everything is small — keep the largest rather than returning nothing.
            keep_labels = labels[[int(np.argmax(counts))]]

        keep = np.isin(face_root, keep_labels)
        if keep.all():
            return self.copy()

        return Mesh(self.vertices, self.faces[keep]).remove_unreferenced_vertices()

    def clean(self, tolerance: float = 1e-6, min_faces: int = 12) -> "Mesh":
        """The full post-march cleanup, in the order that matters."""
        return (
            self.weld(tolerance)
            .remove_unreferenced_vertices()
            .remove_small_components(min_faces)
        )

    # ---------------------------------------------------------------- transforms

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    def translated(self, offset: Sequence[float]) -> "Mesh":
        return Mesh(self.vertices + np.asarray(offset, dtype=np.float64), self.faces)

    def scaled(self, factor: float | Sequence[float]) -> "Mesh":
        return Mesh(self.vertices * np.asarray(factor, dtype=np.float64), self.faces)

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        """Apply a 4x4 homogeneous transform."""
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"expected a 4x4 matrix, got {m.shape}")
        v = self.vertices @ m[:3, :3].T + m[:3, 3]
        return Mesh(v, self.faces)

    def centred(self) -> "Mesh":
        """Move the bounding-box centre to the origin."""
        lower, upper = self.bounds
        return self.translated(-(lower + upper) * 0.5)

    def flipped(self) -> "Mesh":
        """Reverse winding, flipping the normals."""
        return Mesh(self.vertices, self.faces[:, ::-1].copy())

    # ------------------------------------------------------------------- reports

    def stats(self) -> dict[str, object]:
        lower, upper = self.bounds
        return {
            "vertices": self.num_vertices,
            "faces": self.num_faces,
            "area": self.area(),
            "volume": self.volume(),
            "bounds_min": lower.tolist(),
            "bounds_max": upper.tolist(),
            "extent": (upper - lower).tolist(),
        }

    def describe(self) -> str:
        if self.is_empty:
            return "empty mesh"
        e = self.extent
        return (
            f"{self.num_faces:,} triangles, {self.num_vertices:,} vertices, "
            f"{e[0]:.4g} x {e[1]:.4g} x {e[2]:.4g} mm"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Mesh {self.num_vertices} verts, {self.num_faces} faces>"
