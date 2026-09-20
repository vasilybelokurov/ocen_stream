"""The cluster's centre and systemic motion: one definition, derived from the data.

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

__all__ = ["OCEN_RA", "OCEN_DEC", "OCEN_VSYS_KMS", "MU_SYS", "centre"]

#: adopted centre (deg), = ``centre("omegacat")``; verified by ``tests/test_cluster_centre.py``
OCEN_RA = 201.69683333
OCEN_DEC = -47.47956944
#: systemic line-of-sight velocity, km/s
OCEN_VSYS_KMS = 232.7
#: systemic proper motion (mas/yr), Vasiliev & Baumgardt 2021
MU_SYS = (-3.257, -6.730)


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
