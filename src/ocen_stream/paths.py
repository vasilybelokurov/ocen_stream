"""Shared paths for catalogue preparation and kinematic fits.

OCEN_DM_ROOT can override the repository root. Some exploratory tails modules
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
]


def project_root() -> Path:
    """Return the repository root.

    Returns
    -------
    pathlib.Path
        The repository root. Overridden by the ``OCEN_DM_ROOT`` environment
        variable when set, otherwise inferred from this file's location
        (``src/ocen_dm/paths.py`` -> two levels up).
    """
    env = os.environ.get("OCEN_DM_ROOT")
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
