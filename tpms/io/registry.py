"""Extension-to-loader dispatch.

A loader is registered with the extensions it handles and a callable that turns a path
into a :class:`~tpms.core.mesh.Mesh`. Adding a format means writing one module and
calling :func:`register_loader` — nothing else in the project changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from tpms.core.mesh import Mesh

# (path, **options) -> Mesh
LoadFunction = Callable[..., Mesh]


class UnsupportedFormatError(ValueError):
    """The file extension has no registered loader."""


class LoaderUnavailableError(RuntimeError):
    """A loader exists for this format but its dependency is not installed."""


@dataclass(frozen=True)
class LoaderInfo:
    """Everything the app needs to know about one import format."""

    name: str
    extensions: tuple[str, ...]
    load: LoadFunction
    description: str = ""
    # Returns None when usable, or a message explaining what is missing.
    availability_check: Callable[[], str | None] | None = field(
        default=None, repr=False
    )

    def unavailable_reason(self) -> str | None:
        if self.availability_check is None:
            return None
        return self.availability_check()

    @property
    def is_available(self) -> bool:
        return self.unavailable_reason() is None


_LOADERS: list[LoaderInfo] = []


def register_loader(info: LoaderInfo) -> LoaderInfo:
    """Add a loader. A later registration wins for any extension it shares."""
    _LOADERS.append(info)
    return info


def registered_loaders() -> tuple[LoaderInfo, ...]:
    return tuple(_LOADERS)


def _normalise(extension: str) -> str:
    return extension.lower().lstrip(".")


def loader_for(path: str | os.PathLike[str]) -> LoaderInfo:
    """Find the loader for a path, newest registration first."""
    extension = _normalise(os.path.splitext(str(path))[1])
    if not extension:
        raise UnsupportedFormatError(f"'{path}' has no file extension")

    for info in reversed(_LOADERS):
        if extension in info.extensions:
            return info

    raise UnsupportedFormatError(
        f"no loader for '.{extension}'. Supported: "
        f"{', '.join('.' + e for e in supported_extensions())}"
    )


def supported_extensions(available_only: bool = False) -> tuple[str, ...]:
    """Every importable extension, sorted."""
    found: set[str] = set()
    for info in _LOADERS:
        if available_only and not info.is_available:
            continue
        found.update(info.extensions)
    return tuple(sorted(found))


def load(path: str | os.PathLike[str], **options) -> Mesh:
    """Import ``path`` as a mesh.

    Raises :class:`UnsupportedFormatError` for an unknown extension and
    :class:`LoaderUnavailableError` when the format's dependency is missing.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")

    info = loader_for(path)

    reason = info.unavailable_reason()
    if reason is not None:
        raise LoaderUnavailableError(
            f"cannot read {info.name} files: {reason}"
        )

    mesh = info.load(path, **options)

    if mesh.is_empty:
        raise ValueError(f"'{os.path.basename(path)}' contained no triangles")

    return mesh


def qt_file_filter(available_only: bool = True) -> str:
    """Build a Qt file-dialog filter string.

    Formats whose dependency is missing are hidden by default rather than offered and
    then failing — the source panel surfaces them separately with an install hint.
    """
    parts: list[str] = []

    every = " ".join(
        f"*.{e}" for e in supported_extensions(available_only=available_only)
    )
    if every:
        parts.append(f"All supported geometry ({every})")

    for info in _LOADERS:
        if available_only and not info.is_available:
            continue
        patterns = " ".join(f"*.{e}" for e in info.extensions)
        parts.append(f"{info.name} ({patterns})")

    parts.append("All files (*)")
    return ";;".join(parts)
