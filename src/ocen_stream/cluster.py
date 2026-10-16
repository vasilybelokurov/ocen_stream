"""The cluster's position, motion and distance: the single reference for the whole repository.

Every other module and script imports these values; typing any of them anywhere else is
forbidden and caught by ``tests/test_cluster_centre.py::test_no_cluster_reference_value_is_typed_in``
(results of our own fits, e.g. stream best-fit PMs, are not reference values and are unaffected).

Adopted set (2026-10-16), matching the choices of the oMEGACat papers, all peer-reviewed:

* **Centre**: oMEGACat / Anderson & van der Marel (2010, ApJ 710, 1032; 2010ApJ...710.1032A),
  recovered from the catalogue by :func:`centre` (see below).
* **Systemic proper motion**: Vasiliev & Baumgardt (2021, MNRAS 505, 5978; 2021MNRAS.505.5978V),
  Table A1 (VizieR J/MNRAS/505/5978): (-3.250, -6.746) +- 0.022 mas/yr, correlation 0.01. Used by
  oMEGACat I. The value previously here, (-3.257, -6.730), was labelled VB21 but matches no source.
* **Systemic line-of-sight velocity**: Baumgardt & Hilker (2018, MNRAS 478, 1520;
  2018MNRAS.478.1520B), table 2: 232.7 +- 0.2 km/s (5096 stars). Used by oMEGACat I.
* **Distance**: Baumgardt & Vasiliev (2021, MNRAS 505, 5957; 2021MNRAS.505.5957B):
  5.43 +- 0.05 kpc. Used by oMEGACat I and II. A default only; the DF fits keep it free.

Alternatives on record, not adopted: oMEGACat I MUSE systemic velocity 232.99 +- 0.06 km/s
(Nitschai et al. 2023, central field); oMEGACat VI kinematic distance 5.494 +- 0.061 kpc
(Haeberle et al. 2025); Baumgardt's undated website orbit table PM (-3.236, -6.731) +- 0.011.

The centre: one definition, derived from the data.

Until 2026-09-19 the centre was hard-coded in four separate modules as
``(201.696833, -47.476583)`` with the comment "Baumgardt catalogue centre". That declination
came from nowhere: the directory it cites was never downloaded, and the value is **10.7
arcsec north** of the centre the oMEGACat catalogue itself uses and 9.9 arcsec north of
Vasiliev & Baumgardt's. It was written from memory, given a provenance it did not have, and
copied rather than shared, so there was no single place to correct it. The consequence was
worst where it mattered most: stars labelled as lying 2.6-3.2 arcsec from the centre were
really at 7.8-13.9 arcsec.

The centre is therefore no longer a literal. :func:`centre` reads it back out of the
catalogues that define it, and the module-level constants are those values.

* **oMEGACat**: its astrometric solution is tied to a pixel grid whose reference pixel is
  (15000, 15000); fitting the local solution from the 272 stars within 60 px gives
  ``(201.696833, -47.479569)``.
* **Vasiliev & Baumgardt**: their member table carries centred coordinates ``x, y``, from
  which the origin recovers as ``(201.696838, -47.479339)``.

The two agree to **0.83 arcsec**. We adopt the HST one, because it is set by the densest and
best-measured field in the core, and the residual 0.83 arcsec is far below any bin width.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OCEN_RA", "OCEN_DEC", "OCEN_VSYS_KMS", "OCEN_VSYS_ERR_KMS", "MU_SYS", "MU_SYS_ERR", "MU_SYS_CORR",
           "OCEN_DISTANCE_KPC", "OCEN_DISTANCE_ERR_KPC", "OCEN_DISTANCE_OMEGACAT_KPC", "galactic", "centre"]

#: adopted centre (deg), = ``centre("omegacat")``; verified by ``tests/test_cluster_centre.py``
OCEN_RA = 201.69683333
OCEN_DEC = -47.47956944
#: systemic line-of-sight velocity and its error (km/s), Baumgardt & Hilker 2018
OCEN_VSYS_KMS = 232.7
OCEN_VSYS_ERR_KMS = 0.2
#: systemic proper motion (pmra*, pmdec), error per component and correlation (mas/yr), Vasiliev & Baumgardt 2021
MU_SYS = (-3.250, -6.746)
MU_SYS_ERR = 0.022
MU_SYS_CORR = 0.01
#: heliocentric distance and its error (kpc), Baumgardt & Vasiliev 2021
OCEN_DISTANCE_KPC = 5.43
OCEN_DISTANCE_ERR_KPC = 0.05
#: oMEGACat VI kinematic distance (kpc), Haeberle et al. 2025 (alternative, used only where stated)
OCEN_DISTANCE_OMEGACAT_KPC = 5.494


def galactic() -> tuple[float, float]:
    """Galactic (l, b) of the adopted centre, in degrees."""
    import astropy.coordinates as coord
    import astropy.units as u

    g = coord.SkyCoord(OCEN_RA*u.deg, OCEN_DEC*u.deg, frame="icrs").galactic
    return float(g.l.deg), float(g.b.deg)


def centre(source: str = "omegacat") -> tuple[float, float]:
    """Recover the centre from the catalogue that defines it, in degrees.

    ``source="omegacat"`` fits the local astrometric solution around HST reference pixel
    (15000, 15000); ``source="vasiliev"`` inverts the member table's centred ``x, y``.
    """
    from astropy.table import Table

    from .paths import processed_dir, raw_dir

    if source == "omegacat":
        t = Table.read(raw_dir() / "omegacat_vi_kinematics" / "catalog_and_selections.fits")
        ra, dec = np.asarray(t["RA"], float), np.asarray(t["DEC"], float)
        x, y = np.asarray(t["x"], float), np.asarray(t["y"], float)
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(x) & np.isfinite(y)
        near = ok & (np.abs(x - 15000) < 60) & (np.abs(y - 15000) < 60)
        if near.sum() < 50:
            raise ValueError("too few stars near the reference pixel to fit the centre")
        a = np.column_stack([x[near] - 15000, y[near] - 15000, np.ones(int(near.sum()))])
        return (float(np.linalg.lstsq(a, ra[near], rcond=None)[0][2]),
                float(np.linalg.lstsq(a, dec[near], rcond=None)[0][2]))
    if source == "vasiliev":
        t = Table.read(processed_dir() / "tails" / "vasiliev2021_ocen_members.ecsv")
        ra, dec = np.asarray(t["ra"], float), np.asarray(t["dec"], float)
        x, y = np.asarray(t["x"], float), np.asarray(t["y"], float)
        return (float(np.median(ra - x / np.cos(np.radians(dec)))), float(np.median(dec - y)))
    raise ValueError("source must be 'omegacat' or 'vasiliev'")
