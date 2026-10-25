"""Omega Centauri tidal stream (STREAMFINDER stream 54) and the Milky Way bar: data extraction, particle-spray models, fits.

Split from oCen_dm on 2026-10-25 (history kept); the former ocen_dm.streams package is this package.
"""

__version__ = "0.1.0"

from .paths import load_local_env

#: machine-specific locations of external inputs, from the gitignored .env
load_local_env()
