"""TPMS panel — pattern, solidification mode, and the solid skin.

Cell size and wall thickness live in the grading panel, because under grading they are
no longer single numbers.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from tpms.features.generate import GenerationParams
from tpms.features.tpms import (
    SolidMode,
    estimate_volume_fraction,
    get_pattern,
    list_patterns,
)
from tpms.ui.panels.widgets import (
    BasePanel,
    InfoLabel,
    choice,
    current_choice,
    integer,
    muted,
    number,
    section,
    set_choice,
)


class TpmsPanel(BasePanel):
    """Which surface, and how it becomes a solid."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ---- pattern --------------------------------------------------------
        box, form = section("Pattern")
        self.pattern_combo = choice(
            [(p.name, p.label, p.description) for p in list_patterns()], "gyroid"
        )
        form.addRow("Surface", self.pattern_combo)

        self.pattern_note = muted("")
        form.addRow(self.pattern_note)

        self.mode_combo = choice(
            [(m.value, m.label, m.description) for m in SolidMode], SolidMode.SHEET.value
        )
        form.addRow("Solid from", self.mode_combo)

        self.mode_note = muted("")
        form.addRow(self.mode_note)

        self.fraction_spin = number(
            50.0, 1.0, 99.0, step=1.0, decimals=0, suffix=" %",
            tooltip="Target solid fraction of the labyrinth that gets filled.",
        )
        form.addRow("Volume fraction", self.fraction_spin)
        self.add(box)

        # ---- skin -----------------------------------------------------------
        skin_box, skin_form = section("Solid skin")
        self.skin_spin = number(
            0.0, 0.0, 100.0, step=0.1,
            tooltip="A solid wall on the part surface. 0 leaves the lattice open.",
        )
        skin_form.addRow("Thickness", self.skin_spin)

        self.blend_spin = number(
            0.0, 0.0, 20.0, step=0.1,
            tooltip="Fillet radius where the lattice meets the skin. "
                    "A blend removes the sharp junction, which is where a loaded "
                    "part tends to crack first.",
        )
        skin_form.addRow("Blend", self.blend_spin)
        skin_form.addRow(muted(
            "The skin grows inwards, so the part keeps its modelled outer size."
        ))
        self.add(skin_box)

        # ---- accuracy -------------------------------------------------------
        quality_box, quality_form = section("Wall accuracy")
        self.refine_spin = integer(
            2, 0, 4,
            tooltip="Newton passes used to convert the pattern field into a real "
                    "distance. 0 is fastest and makes walls up to 15% thin; 2 holds "
                    "them under 1%.",
        )
        quality_form.addRow("Refinement", self.refine_spin)
        quality_form.addRow(muted(
            "2 passes keep wall thickness within about 1% of the value you set. "
            "0 is faster but runs thin, increasingly so for thick walls."
        ))
        self.add(quality_box)

        self.estimate = InfoLabel("")
        self.add(self.estimate)
        self.add_stretch()

        self.wire(
            self.pattern_combo, self.mode_combo, self.fraction_spin,
            self.skin_spin, self.blend_spin, self.refine_spin,
        )
        self.pattern_combo.currentIndexChanged.connect(self._update_notes)
        self.mode_combo.currentIndexChanged.connect(self._update_notes)

        self._update_notes()

    # ----------------------------------------------------------------- handlers

    def _update_notes(self) -> None:
        pattern = get_pattern(current_choice(self.pattern_combo))
        self.pattern_note.setText(pattern.description)

        mode = SolidMode(current_choice(self.mode_combo))
        self.mode_note.setText(mode.description)

        # Volume fraction only means anything for the network modes.
        uses_fraction = not mode.uses_thickness
        self.fraction_spin.setEnabled(uses_fraction)

    def update_estimate(self, cell_size: float, thickness: float) -> None:
        """Show the predicted relative density. Called by the main window."""
        mode = SolidMode(current_choice(self.mode_combo))
        if not mode.uses_thickness:
            self.estimate.info(
                f"Network mode: relative density follows the volume fraction "
                f"({self.fraction_spin.value():.0f}%)."
            )
            return

        try:
            pattern = get_pattern(current_choice(self.pattern_combo))
            fraction = estimate_volume_fraction(pattern, mode, cell_size, thickness)
        except Exception:
            self.estimate.info("")
            return

        ratio = thickness / cell_size if cell_size > 0 else 0.0
        text = f"Estimated relative density {fraction * 100:.0f}%."

        if ratio > 0.30:
            self.estimate.warn(
                text + " The wall is over 30% of the cell — neighbouring walls are "
                "merging and the void is closing up."
            )
        elif ratio < 0.04:
            self.estimate.warn(
                text + " The wall is under 4% of the cell and may be too thin to print."
            )
        else:
            self.estimate.info(text)

    # --------------------------------------------------------------- parameters

    def apply_to(self, params: GenerationParams) -> None:
        params.pattern = current_choice(self.pattern_combo)
        params.mode = SolidMode(current_choice(self.mode_combo))
        params.volume_fraction = float(self.fraction_spin.value()) / 100.0
        params.skin_thickness = float(self.skin_spin.value())
        params.skin_blend = float(self.blend_spin.value())
        params.refine_steps = int(self.refine_spin.value())

    def load_from(self, params: GenerationParams) -> None:
        with self.loading():
            set_choice(self.pattern_combo, params.pattern)
            set_choice(self.mode_combo, params.mode.value)
            self.fraction_spin.setValue(params.volume_fraction * 100.0)
            self.skin_spin.setValue(params.skin_thickness)
            self.blend_spin.setValue(params.skin_blend)
            self.refine_spin.setValue(params.refine_steps)
        self._update_notes()
