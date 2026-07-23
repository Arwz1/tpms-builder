"""The generation pipeline — the one place that knows the whole sequence.

Everything else is a part; this assembles them:

    domain  ->  grid  ->  per-slab lattice field  ->  clip  ->  march  ->  mesh

Runs headless. The UI wraps it in a thread and passes a progress callback, but nothing
here imports Qt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tpms.core import sdf
from tpms.core.grid import Grid, Slab
from tpms.core.marching import GenerationCancelled, march_slabs
from tpms.core.mesh import Mesh
from tpms.features.generate.params import GenerationParams, SourceType
from tpms.features.grading import GradingContext, get_grading
from tpms.features.shapes import get_shape
from tpms.features.tpms import get_pattern, solidify
from tpms.features.voxelize import DomainField, build_domain_field

# (fraction 0..1, message) -> keep going?
ProgressCallback = Callable[[float, str], bool]


@dataclass
class GenerationResult:
    """What a run produced, plus enough context to report on it."""

    mesh: Mesh
    grid: Grid
    params: GenerationParams
    seconds: float
    domain_volume: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def lattice_volume(self) -> float:
        return self.mesh.volume()

    @property
    def relative_density(self) -> float:
        """Lattice volume as a fraction of the solid part it fills."""
        if self.domain_volume <= 0:
            return 0.0
        return self.lattice_volume / self.domain_volume

    def summary(self) -> str:
        lines = [
            f"{self.mesh.num_faces:,} triangles, {self.mesh.num_vertices:,} vertices",
            f"grid {self.grid.describe()}",
            f"generated in {self.seconds:.1f} s",
        ]
        if self.domain_volume > 0:
            lines.append(
                f"relative density {self.relative_density * 100:.1f}% "
                f"({self.lattice_volume:.0f} of {self.domain_volume:.0f} mm³)"
            )
        return "\n".join(lines)


def _noop_progress(fraction: float, message: str) -> bool:
    return True


class DomainSource:
    """The solid being filled, however it was obtained.

    Wraps either an analytic shape or a voxelised import behind one call, so the field
    builder does not care which it got.
    """

    def __init__(
        self,
        distance: Callable[..., np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        volume: float = 0.0,
        domain_field: DomainField | None = None,
    ) -> None:
        self.distance = distance
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.volume = volume
        self.domain_field = domain_field

    def __call__(self, x, y, z) -> np.ndarray:
        return self.distance(x, y, z)


def build_domain(
    params: GenerationParams,
    mesh: Mesh | None = None,
    progress: ProgressCallback = _noop_progress,
) -> DomainSource:
    """Produce the solid to be filled.

    An analytic shape is used directly — exact at every resolution, no voxelisation.
    An imported mesh is voxelised once into an SDF and read back by interpolation.
    """
    if params.source_type is SourceType.SHAPE:
        shape = get_shape(params.shape_name, **params.shape_params)
        lower, upper = shape.bounds()

        # Analytic volume by sampling, only for the density report.
        return DomainSource(
            distance=shape.sdf, lower=lower, upper=upper, volume=0.0
        )

    if mesh is None:
        from tpms.io import load

        if not progress(0.02, "Loading geometry"):
            raise GenerationCancelled("cancelled while loading")
        mesh = load(params.source_path)

    if not progress(0.05, "Voxelising geometry"):
        raise GenerationCancelled("cancelled while voxelising")

    domain_field = build_domain_field(mesh, resolution=params.domain_resolution)
    lower, upper = mesh.bounds

    return DomainSource(
        distance=domain_field,
        lower=lower,
        upper=upper,
        volume=domain_field.solid_volume(),
        domain_field=domain_field,
    )


def make_field_function(
    params: GenerationParams,
    domain: DomainSource,
    context: GradingContext,
) -> Callable[[Grid, Slab], np.ndarray]:
    """Build the per-slab field evaluator.

    Returned as a closure so the pattern, grading law and context are resolved once
    rather than per slab.
    """
    pattern = get_pattern(params.pattern)
    grading = get_grading(params.grading, **params.grading_params)

    skin = float(params.skin_thickness)
    blend = float(params.skin_blend)

    def field_fn(grid: Grid, slab: Slab) -> np.ndarray:
        x, y, z = grid.slab_meshgrid(slab, dtype=np.float32)

        # Phase and scale come from the grading law, which owns the relationship
        # between world position and cell size.
        u, v, w = grading.phase(x, y, z, context)
        phase_scale = grading.phase_scale(x, y, z, context)
        thickness = grading.thickness(x, y, z, context)

        lattice = solidify(
            pattern, u, v, w, phase_scale,
            mode=params.mode,
            thickness=thickness,
            volume_fraction=params.volume_fraction,
            refine_steps=params.refine_steps,
        )

        boundary = np.asarray(domain(x, y, z), dtype=np.float32)

        # Clip the lattice to the part.
        result = sdf.intersect(lattice, boundary)

        if skin > 0.0:
            # A wall of `skin` lying entirely inside the surface, so the part keeps its
            # modelled outer dimensions.
            skin_field = sdf.inner_shell(boundary, skin)
            if blend > 0.0:
                result = sdf.smooth_union(result, skin_field, blend)
            else:
                result = sdf.union(result, skin_field)

        return np.ascontiguousarray(result, dtype=np.float32)

    return field_fn


def generate(
    params: GenerationParams,
    mesh: Mesh | None = None,
    progress: ProgressCallback | None = None,
) -> GenerationResult:
    """Run the full pipeline.

    ``mesh`` lets a caller pass geometry that is already loaded, so the UI does not
    re-read the file on every parameter change.

    Raises :class:`~tpms.core.marching.GenerationCancelled` if ``progress`` returns
    ``False``, and ``ValueError`` if the parameters do not validate.
    """
    progress = progress or _noop_progress
    started = time.perf_counter()

    # Params are a mutable dataclass that callers build and then adjust, so the
    # constructor's clamps may be stale by now.
    params.normalise()

    problems = params.validate()
    if problems:
        raise ValueError("; ".join(problems))

    domain = build_domain(params, mesh=mesh, progress=progress)

    # Pad by one cell so the lattice is not clipped flat against the domain wall.
    padding = params.nominal_cell_size() * 0.5
    grid = Grid.from_bounds(domain.lower, domain.upper, params.resolution, padding=padding)

    context = GradingContext(
        lower=domain.lower,
        upper=domain.upper,
        base_cell_size=params.nominal_cell_size(),
        base_thickness=params.nominal_thickness(),
        domain_distance=domain.distance,
    )

    field_fn = make_field_function(params, domain, context)

    if not progress(0.08, f"Meshing at {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]}"):
        raise GenerationCancelled("cancelled before meshing")

    lattice = march_slabs(
        grid,
        field_fn,
        slab_depth=params.slab_depth,
        progress=progress,
        min_component_faces=params.min_component_faces,
    )

    warnings: list[str] = []
    if lattice.is_empty:
        warnings.append(
            "The result is empty. The wall may be thinner than one voxel — raise the "
            "resolution or the wall thickness."
        )
    elif lattice.num_faces < 100:
        warnings.append(
            "Very few triangles were produced. Check that the cell size is smaller "
            "than the part."
        )

    domain_volume = domain.volume
    if domain_volume <= 0 and not lattice.is_empty:
        domain_volume = _estimate_domain_volume(domain, grid)

    return GenerationResult(
        mesh=lattice,
        grid=grid,
        params=params,
        seconds=time.perf_counter() - started,
        domain_volume=domain_volume,
        warnings=warnings,
    )


def _estimate_domain_volume(domain: DomainSource, grid: Grid, samples: int = 64) -> float:
    """Monte-Carlo-free volume estimate for an analytic domain, by coarse sampling."""
    coarse = grid.with_resolution(samples)
    total = 0
    for slab in coarse.slabs(16):
        x, y, z = coarse.slab_meshgrid(slab, dtype=np.float32)
        # Slabs overlap by a plane; drop the duplicate so it is not counted twice.
        inside = np.asarray(domain(x, y, z)) < 0
        if not slab.is_first:
            inside = inside[:, :, 1:]
        total += int(inside.sum())

    return float(total) * float(np.prod(coarse.spacing))


def preview_params(params: GenerationParams, resolution: int = 96) -> GenerationParams:
    """A cheaper copy of ``params`` for interactive preview."""
    from dataclasses import replace

    return replace(
        params,
        resolution=min(resolution, params.resolution),
        domain_resolution=min(96, params.domain_resolution),
        refine_steps=1,
        min_component_faces=0,
    )
