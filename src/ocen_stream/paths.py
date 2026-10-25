"""Shared paths (repository root, data and results directories, the local .env).

OCEN_STREAM_ROOT can override the repository root. Some exploratory tails modules
also use working-directory-relative outputs and external local resources;
this module does not centralise every path in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "project_root",
    "data_dir",
    "raw_dir",
    "interim_dir",
    "processed_dir",
    "provenance_dir",
    "configs_dir",
    "results_dir",
    "ensure_data_tree",
    "load_local_env",
]


def project_root() -> Path:
    """Return the repository root.

    Returns
    -------
    pathlib.Path
        The repository root. Overridden by the ``OCEN_STREAM_ROOT`` environment
        variable when set, otherwise inferred from this file's location
        (``src/ocen_stream/paths.py`` -> two levels up).
    """
    env = os.environ.get("OCEN_STREAM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return ``<root>/data``."""
    return project_root() / "data"


def raw_dir() -> Path:
    """Return ``<root>/data/raw`` (never edited by hand, never committed)."""
    return data_dir() / "raw"


def interim_dir() -> Path:
    """Return ``<root>/data/interim`` (reproducible intermediate products)."""
    return data_dir() / "interim"


def processed_dir() -> Path:
    """Return ``<root>/data/processed`` (analysis-ready products)."""
    return data_dir() / "processed"


def provenance_dir() -> Path:
    """Return ``<root>/provenance``."""
    return project_root() / "provenance"


def configs_dir() -> Path:
    """Return ``<root>/configs``."""
    return project_root() / "configs"


def results_dir() -> Path:
    """Return ``<root>/results``."""
    return project_root() / "results"


def ensure_data_tree() -> None:
    """Create the data/results directory tree if it does not yet exist."""
    for path in (raw_dir(), interim_dir(), processed_dir(), results_dir()):
        path.mkdir(parents=True, exist_ok=True)


def load_local_env(path: Path | None = None) -> dict[str, str]:
    """Set environment variables from the repository's local ``.env`` file.

    The file (gitignored, one ``NAME=value`` per line; ``#`` comments, an optional
    ``export`` prefix and surrounding quotes allowed) holds the machine-specific
    locations of external inputs such as ``OCEN_CATALOGUES``. Values are expanded
    for ``~`` and ``$VAR``. A variable that is already set in the environment is
    never overwritten, so an explicit ``export`` still wins. Called on
    ``import ocen_stream``; returns the variables it set.
    """
    path = Path(path) if path is not None else project_root() / ".env"
    applied: dict[str, str] = {}
    if not path.is_file():
        return applied
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, value = (part.strip() for part in line.split("=", 1))
        if not name.isidentifier() or name in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        value = os.path.expanduser(os.path.expandvars(value))
        os.environ[name] = value
        applied[name] = value
    return applied
