"""Shared fixtures. All test data are synthetic and generated in-process."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator."""
    return np.random.default_rng(20260916)


@pytest.fixture
def synthetic_periphery(rng) -> Table:
    """A synthetic Gaia-like periphery catalogue (NOT real Omega Cen data)."""
    n = 64
    table = Table()
    table["source_id"] = np.arange(n, dtype=np.int64) + 6_000_000_000
    table["ra"] = (201.7 + 0.5 * rng.standard_normal(n)) * u.deg
    table["dec"] = (-47.5 + 0.5 * rng.standard_normal(n)) * u.deg
    table["pmra"] = (-3.25 + 0.3 * rng.standard_normal(n)) * u.mas / u.yr
    table["pmdec"] = (-6.75 + 0.3 * rng.standard_normal(n)) * u.mas / u.yr
    table["phot_g_mean_mag"] = (17.0 + rng.uniform(-2, 2, n)) * u.mag
    table["prob"] = rng.uniform(0, 1, n)
    table.meta["synthetic"] = True
    return table


@pytest.fixture
def synthetic_profile(rng) -> Table:
    """A synthetic binned dispersion profile shaped like oMEGACat VI Table 2.

    Bin edges and asymmetric errors, with deliberately non-standard column
    names so that the explicit column map is exercised.
    """
    edges = np.geomspace(1.0, 2000.0, 13)
    lower, upper = edges[:-1], edges[1:]
    median = np.sqrt(lower * upper)
    sigma = 0.8 / np.sqrt(1.0 + (median / 300.0) ** 2)
    table = Table()
    table["R_low"] = lower * u.arcsec
    table["R_med"] = median * u.arcsec
    table["R_up"] = upper * u.arcsec
    table["N"] = np.full(median.size, 500, dtype=np.int64)
    table["sigma_R"] = sigma * u.mas / u.yr
    table["sigma_R_lo"] = 0.04 * sigma * u.mas / u.yr
    table["sigma_R_hi"] = 0.05 * sigma * u.mas / u.yr
    table.meta["synthetic"] = True
    return table


@pytest.fixture
def synthetic_vizier_fimbulthul(rng) -> Table:
    """A synthetic table with the REAL column names of J/other/NatAs/3.667.

    Names, units and dtypes were read from the live VizieR table description on
    2026-09-16; only the values are synthetic.
    """
    n = 12
    table = Table()
    table["recno"] = np.arange(1, n + 1, dtype=np.int32)
    table["RAJ2000"] = (200.0 + rng.uniform(-5, 5, n)) * u.deg
    table["DEJ2000"] = (-27.0 + rng.uniform(-5, 5, n)) * u.deg
    table["pmRA"] = rng.normal(-3.0, 0.5, n) * u.mas / u.yr
    table["e_pmRA"] = np.full(n, 0.1) * u.mas / u.yr
    table["pmDE"] = rng.normal(-6.0, 0.5, n) * u.mas / u.yr
    table["e_pmDE"] = np.full(n, 0.1) * u.mas / u.yr
    table["plx"] = rng.normal(0.1, 0.05, n) * u.mas
    table["e_plx"] = np.full(n, 0.05) * u.mas
    table["G0mag"] = rng.uniform(15, 19, n) * u.mag
    table["__BP-RP_0"] = rng.uniform(0.6, 1.0, n) * u.mag
    table["SimbadName"] = [f"Gaia DR3 {6190000000000000000 + i}" for i in range(n)]
    table.meta["synthetic"] = True
    return table


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """Point the package's path helpers at a temporary project root."""
    monkeypatch.setenv("OCEN_DM_ROOT", str(tmp_path))
    for sub in ("data/raw", "data/interim", "data/processed", "provenance", "configs"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def registry_copy(isolated_root):
    """Copy the real provenance registry into the temporary project root."""
    import shutil
    from pathlib import Path

    import ocen_dm

    repo_root = Path(ocen_dm.__file__).resolve().parents[2]
    shutil.copy(repo_root / "provenance" / "datasets.yaml",
                isolated_root / "provenance" / "datasets.yaml")
    return isolated_root


@pytest.fixture(autouse=True, scope="module")
def _restore_agama_units():
    """Stop one module's global AGAMA units leaking into the next.

    AGAMA units are process-global. Orbit/tail modules set kpc units, while the
    DF branch (``positive_df.agama_pc``) requires pc and refuses mixed units.
    Units in force when a module starts are restored when it ends; if none were
    set, the DF convention (pc, Msun, km/s) is restored. AGAMA objects made
    under a module's own units must not outlive that module.
    """
    try:
        import agama
    except ImportError:
        yield
        return
    before = dict(agama.getUnits())
    yield
    after = dict(agama.getUnits())
    keys = ("length", "mass", "velocity")
    if before and all(np.isclose(after.get(k, np.nan), before[k]) for k in keys):
        return
    if before:
        agama.setUnits(**{k: before[k] for k in keys})
    elif after:
        agama.setUnits(length=0.001, mass=1., velocity=1.)
