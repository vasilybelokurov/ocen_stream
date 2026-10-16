"""Solar frames and omega Cen's present-day phase-space point in the model (AGAMA) Galactocentric frame.

Model frame convention (bin/nbody/analyse_tails.py): x_model = -x_astropy, v_x,model = -v_x,astropy.
Frames:
  baumgardt  R0 = 8.178 kpc, z_sun = 0, Vsun = (11.1, 252.24, 7.25) km/s; omega Cen d = 5.43 kpc. Reproduces OCEN_TODAY
             (Baumgardt orbits_table.txt) used by all runs before 2026-10-08 evening.
  ibata19    Ibata+2019 (arXiv:1902.09544, Methods): R0 = 8.122 kpc, z_sun = 17 pc, v_c(R0) = 220 km/s plus the Schoenrich+2010
             peculiar motion (11.1, 12.24, 7.25) -> Vsun = (11.1, 232.24, 7.25); omega Cen d = 5.63 kpc.
omega Cen observables (both frames): position, systemic PM and v_los from ocen_dm.cluster (the single reference;
Vasiliev & Baumgardt 2021 PM, Baumgardt & Hilker 2018 v_los). Until 2026-10-16 they were typed in here as the
Baumgardt orbit-table values (201.69683, -47.47958), PM (-3.2499, -6.7461), v_los 232.78: the same within
0.04 arcsec, 0.0001 mas/yr and 0.08 km/s, so runs before that date start from a state differing at that level.
Only the solar parameters, and the ibata19 frame's own omega Cen distance (Ibata+2019's 5.63 kpc), live here.
"""
from __future__ import annotations

import numpy as np

from ..cluster import MU_SYS, OCEN_DEC, OCEN_DISTANCE_KPC, OCEN_RA, OCEN_VSYS_KMS

FRAMES = {
    "baumgardt": dict(r0_kpc=8.178, z_sun_pc=0.0, v_sun=(11.1, 252.24, 7.25), d_ocen_kpc=OCEN_DISTANCE_KPC),
    "ibata19": dict(r0_kpc=8.122, z_sun_pc=17.0, v_sun=(11.1, 232.24, 7.25), d_ocen_kpc=5.63),
}
OCEN_OBS = dict(ra=OCEN_RA, dec=OCEN_DEC, pmra=MU_SYS[0], pmdec=MU_SYS[1], vlos=OCEN_VSYS_KMS)


def galcen(frame="baumgardt"):
    import astropy.coordinates as coord
    import astropy.units as u
    f = FRAMES[frame]
    return coord.Galactocentric(galcen_distance=f["r0_kpc"]*u.kpc, z_sun=f["z_sun_pc"]*u.pc,
                                galcen_v_sun=coord.CartesianDifferential(f["v_sun"]*u.km/u.s))


def to_model(ra, dec, d_kpc, pmra, pmdec, vlos, frame="baumgardt"):
    """ICRS observables -> model-frame (N, 6) [kpc, km/s]."""
    import astropy.coordinates as coord
    import astropy.units as u
    c = coord.SkyCoord(ra=np.atleast_1d(ra)*u.deg, dec=np.atleast_1d(dec)*u.deg, distance=np.atleast_1d(d_kpc)*u.kpc,
                       pm_ra_cosdec=np.atleast_1d(pmra)*u.mas/u.yr, pm_dec=np.atleast_1d(pmdec)*u.mas/u.yr,
                       radial_velocity=np.atleast_1d(vlos)*u.km/u.s).transform_to(galcen(frame))
    return np.column_stack((-c.x.to_value(u.kpc), c.y.to_value(u.kpc), c.z.to_value(u.kpc),
                            -c.v_x.to_value(u.km/u.s), c.v_y.to_value(u.km/u.s), c.v_z.to_value(u.km/u.s)))


def ocen_today(frame="baumgardt", d_kpc=None):
    o = OCEN_OBS
    d = FRAMES[frame]["d_ocen_kpc"] if d_kpc is None else d_kpc
    return to_model(o["ra"], o["dec"], d, o["pmra"], o["pmdec"], o["vlos"], frame)[0]


def observables(xv, frame="baumgardt"):
    """Model-frame (N, 6) -> dict of l, b [deg], pmra*, pmdec [mas/yr], vlos [km/s], dist [kpc] (heliocentric)."""
    import astropy.coordinates as coord
    import astropy.units as u
    gc = coord.SkyCoord(x=-xv[:, 0]*u.kpc, y=xv[:, 1]*u.kpc, z=xv[:, 2]*u.kpc, v_x=-xv[:, 3]*u.km/u.s, v_y=xv[:, 4]*u.km/u.s,
                        v_z=xv[:, 5]*u.km/u.s, frame=galcen(frame))
    ic = gc.transform_to(coord.ICRS()); g = gc.transform_to(coord.Galactic())
    return dict(l=np.asarray(g.l.deg), b=np.asarray(g.b.deg), pmra=np.asarray(ic.pm_ra_cosdec.value),
                pmdec=np.asarray(ic.pm_dec.value), vlos=np.asarray(ic.radial_velocity.value), dist=np.asarray(ic.distance.kpc))
