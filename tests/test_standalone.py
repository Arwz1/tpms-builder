"""Each feature must work on its own.

`claude.md` requires that features be usable standalone. That is easy to state and easy
to break — one convenience import of a Qt symbol inside a feature and the whole thing
silently becomes GUI-only. These tests make the rule enforceable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every module that must import and run without Qt.
HEADLESS_PACKAGES = [
    "tpms.core",
    "tpms.io",
    "tpms.features.tpms",
    "tpms.features.grading",
    "tpms.features.shapes",
    "tpms.features.voxelize",
    "tpms.features.generate",
    "tpms.cli",
]


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter, so imports from other tests cannot mask a bug."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )


@pytest.mark.parametrize("package", HEADLESS_PACKAGES)
def test_imports_without_qt(package: str) -> None:
    """Importing a feature must not pull in PySide6.

    Checked in a subprocess: if a previous test already imported Qt, an in-process
    check would pass regardless.
    """
    code = (
        "import sys\n"
        f"import {package}\n"
        "loaded = [m for m in sys.modules if m.startswith('PySide6')]\n"
        "assert not loaded, f'{loaded}'\n"
        "print('ok')\n"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, (
        f"{package} pulled in Qt or failed to import:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_tpms_feature_alone() -> None:
    from tpms.features.tpms import SolidMode, get_pattern, solidify

    pattern = get_pattern("gyroid")
    axis = np.linspace(0, 2 * np.pi, 12, dtype=np.float32)
    u, v, w = np.meshgrid(axis, axis, axis, indexing="ij")

    field = solidify(pattern, u, v, w, np.float32(1.0), SolidMode.SHEET, thickness=0.3)

    assert field.shape == (12, 12, 12)
    assert np.isfinite(field).all()
    assert (field < 0).any() and (field > 0).any()


def test_grading_feature_alone() -> None:
    from tpms.features.grading import GradingContext, get_grading

    ctx = GradingContext(lower=[0, 0, 0], upper=[60, 60, 60])
    grading = get_grading("axial", axis=2, cell_start=3.0, cell_end=9.0)

    z = np.linspace(0, 60, 5, dtype=np.float32)
    cell = grading.cell_size(z, z, z, ctx)

    assert np.allclose(cell, [3.0, 4.5, 6.0, 7.5, 9.0], atol=1e-4)


def test_shapes_feature_alone() -> None:
    from tpms.features.shapes import get_shape, shape_names

    assert "box" in shape_names()

    shape = get_shape("sphere", radius=5.0)
    centre = np.float64(0.0)

    assert float(shape.sdf(centre, centre, centre)) == pytest.approx(-5.0)
    lower, upper = shape.bounds()
    assert np.allclose(upper, [5.0, 5.0, 5.0])


def test_voxelize_feature_alone() -> None:
    from tpms.core import Grid, march_field, sdf
    from tpms.features.voxelize import build_domain_field

    grid = Grid.from_bounds([-12] * 3, [12] * 3, 56)
    x, y, z = grid.meshgrid()
    sphere = march_field(grid, sdf.sphere(x, y, z, 8.0))

    domain = build_domain_field(sphere, resolution=72)
    expected = 4 / 3 * np.pi * 8.0 ** 3

    assert domain.solid_volume() == pytest.approx(expected, rel=0.05)


def test_core_has_no_project_dependencies() -> None:
    """``core`` must not import from ``features``, ``io`` or ``ui``.

    The layering only holds if the bottom layer stays at the bottom.
    """
    core = ROOT / "tpms" / "core"
    forbidden = ("tpms.features", "tpms.io", "tpms.ui")

    for path in core.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in text, f"{path.name} imports {name}"


def test_features_do_not_import_ui() -> None:
    features = ROOT / "tpms" / "features"

    for path in features.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "tpms.ui" not in text, f"{path.relative_to(ROOT)} references tpms.ui"
        assert "PySide6" not in text, f"{path.relative_to(ROOT)} references PySide6"
