"""Quality panel — resolution, and an honest estimate of what it will cost.

The estimates matter. At 512³ a fine lattice reaches ~28 million triangles and about
4.4 GB of peak memory, and there is no way for a user to guess that from a resolution
slider. So the panel predicts triangles and memory before the run and says plainly when
a setting is going to hurt, rather than letting the machine find out.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from tpms.core.grid import MAX_RESOLUTION, WARN_RESOLUTION, Grid
from tpms.features.generate import GenerationParams
from tpms.features.tpms import SolidMode
from tpms.ui.panels.widgets import (
    BasePanel,
    InfoLabel,
    integer,
    muted,
    number,
    section,
)

#: Peak memory above which the panel warns outright.
MEMORY_WARN_BYTES = 3.0e9
MEMORY_ALARM_BYTES = 6.0e9


class QualityPanel(BasePanel):
    """Sampling resolution and the estimates that come with it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        box, form = section("Lattice resolution")
        self.resolution = integer(
            192, 32, MAX_RESOLUTION, step=32,
            tooltip="Samples along the longest axis. Doubling it multiplies "
                    "triangles by about four.",
        )
        form.addRow("Grid", self.resolution)

        self.voxel_label = muted("")
        form.addRow("Voxel size", self.voxel_label)
        self.add(box)

        # ---- imported geometry ---------------------------------------------
        domain_box, domain_form = section("Imported geometry")
        self.domain_resolution = integer(
            160, 32, 256, step=32,
            tooltip="Resolution of the signed distance field built from an imported "
                    "mesh. Independent of the lattice resolution.",
        )
        domain_form.addRow("SDF grid", self.domain_resolution)
        domain_form.addRow(muted(
            "The part boundary is smooth and low-frequency; the lattice is not. "
            "Sampling them separately means a fine lattice does not need a fine "
            "boundary field. Only affects imported geometry — base shapes are exact."
        ))
        self.add(domain_box)

        # ---- advanced -------------------------------------------------------
        advanced_box, advanced_form = section("Meshing")
        self.slab_depth = integer(
            16, 2, 256, step=2,
            tooltip="Z-planes held in memory at once. Smaller uses less memory; "
                    "larger is marginally faster.",
        )
        advanced_form.addRow("Slab depth", self.slab_depth)

        self.min_faces = integer(
            24, 0, 100_000, step=4,
            tooltip="Drop disconnected fragments below this many triangles. "
                    "Clears the slivers left where a wall grazes the part surface.",
        )
        advanced_form.addRow("Min fragment", self.min_faces)
        self.add(advanced_box)

        # ---- estimates ------------------------------------------------------
        self.triangles_label = InfoLabel("")
        self.add(self.triangles_label)
        self.memory_label = InfoLabel("")
        self.add(self.memory_label)
        self.add_stretch()

        self.wire(self.resolution, self.domain_resolution,
                  self.slab_depth, self.min_faces)

        self._lower = (-30.0, -30.0, -30.0)
        self._upper = (30.0, 30.0, 30.0)
        self._cell = 6.0
        self._sheet = True

    # ---------------------------------------------------------------- estimates

    def update_estimate(
        self, lower, upper, cell_size: float, mode: SolidMode
    ) -> None:
        """Recompute the predictions. Called by the main window on any change."""
        self._lower = lower
        self._upper = upper
        self._cell = max(float(cell_size), 1e-6)
        self._sheet = SolidMode(mode).uses_thickness
        self._refresh()

    def _refresh(self) -> None:
        try:
            grid = Grid.from_bounds(
                self._lower, self._upper, int(self.resolution.value()),
                padding=self._cell * 0.5,
            )
        except ValueError:
            self.voxel_label.setText("—")
            return

        voxel = float(grid.spacing[0])
        self.voxel_label.setText(
            f"{voxel:.3f} mm   ({grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]})"
        )

        # Cells are counted across the part, not the padded grid — the lattice is
        # clipped to the part and never fills the padding.
        import numpy as np

        solid_extent = np.asarray(self._upper, dtype=float) - np.asarray(
            self._lower, dtype=float
        )

        triangles = grid.estimate_triangles(
            self._cell, sheet=self._sheet, solid_extent=solid_extent
        )
        memory = grid.estimate_peak_bytes(triangles)

        self.triangles_label.info(f"Estimated output: {triangles / 1e6:.1f} M triangles.")

        gigabytes = memory / 1e9
        if memory > MEMORY_ALARM_BYTES:
            self.memory_label.error(
                f"Estimated peak memory {gigabytes:.1f} GB. This will likely exhaust "
                "RAM on most machines. Lower the resolution or increase the cell size."
            )
        elif memory > MEMORY_WARN_BYTES:
            self.memory_label.warn(
                f"Estimated peak memory {gigabytes:.1f} GB, and the run may take a "
                "minute or more. Generation stays cancellable."
            )
        elif int(self.resolution.value()) > WARN_RESOLUTION:
            self.memory_label.warn(
                f"Estimated peak memory {gigabytes:.1f} GB. High resolution — "
                "expect tens of seconds."
            )
        else:
            self.memory_label.info(f"Estimated peak memory {gigabytes:.1f} GB.")

    def check_wall_resolution(self, wall_thickness: float) -> str:
        """Warn when the wall is too thin for the grid to resolve.

        A wall spanning fewer than about two voxels comes out pitted or disappears
        entirely — the single most common reason a run returns nothing.
        """
        try:
            grid = Grid.from_bounds(
                self._lower, self._upper, int(self.resolution.value()),
                padding=self._cell * 0.5,
            )
        except ValueError:
            return ""

        voxels = wall_thickness / float(grid.spacing[0])
        if voxels < 2.0:
            return (
                f"The wall is only {voxels:.1f} voxels thick. Raise the resolution or "
                "the wall thickness, or the lattice will come out broken or empty."
            )
        if voxels < 3.0:
            return (
                f"The wall is {voxels:.1f} voxels thick. It will mesh, but coarsely."
            )
        return ""

    # --------------------------------------------------------------- parameters

    def apply_to(self, params: GenerationParams) -> None:
        params.resolution = int(self.resolution.value())
        params.domain_resolution = int(self.domain_resolution.value())
        params.slab_depth = int(self.slab_depth.value())
        params.min_component_faces = int(self.min_faces.value())

    def load_from(self, params: GenerationParams) -> None:
        with self.loading():
            self.resolution.setValue(params.resolution)
            self.domain_resolution.setValue(params.domain_resolution)
            self.slab_depth.setValue(params.slab_depth)
            self.min_faces.setValue(params.min_component_faces)
        self._refresh()
