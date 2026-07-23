"""Parameter panels, one per feature.

Every panel reads and writes :class:`~tpms.features.generate.params.GenerationParams`
and emits ``changed``. None of them touch the pipeline directly — the main window owns
that decision.
"""

from tpms.ui.panels.export_panel import ExportPanel
from tpms.ui.panels.grading_panel import GradingPanel
from tpms.ui.panels.quality_panel import QualityPanel
from tpms.ui.panels.source_panel import SourcePanel
from tpms.ui.panels.tpms_panel import TpmsPanel
from tpms.ui.panels.widgets import BasePanel, InfoLabel

__all__ = [
    "BasePanel",
    "InfoLabel",
    "SourcePanel",
    "TpmsPanel",
    "GradingPanel",
    "QualityPanel",
    "ExportPanel",
]
