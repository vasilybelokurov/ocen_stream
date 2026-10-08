"""Internal rotation of omega Cen star tracers by Lynden-Bell flipping (spherical equilibrium preserved).

Geometry follows Ibata+2019 (Nature Astronomy 3, 667; arXiv:1902.09544, Methods): rotation axis on the sky at PA 12 deg
(E of N), the right-handed pole inclined 45 deg towards the observer (Bianchini+2018, Gaia DR2). The sense was checked
against our data (journal 2026-10-08): oMEGACat VI v_los is maximal (receding) at PA 103 deg, so the pole's sky
projection points to PA 192 deg; Gaia DR3 plane-of-sky rotation runs N -> E, so the pole points towards us.

Flip rule: a star whose angular momentum about the spin axis is negative has its velocity component along e_phi
(about the spin axis) reversed with probability q(r_max) = q0 x^2/(1+x^2) / (1 + (r_max/r_q)^2), x = r_max/r_1. E and |L| are unchanged, and for
a spherical DF f(E, L) the density and potential are unchanged, so the cluster stays in equilibrium.

Frames: model Galactocentric vectors use x_model = -x_astropy (bin/nbody/analyse_tails.py convention, R0 = 8.178 kpc,
Vsun = (11.1, 252.24, 7.25) km/s, z_sun = 0). Relative position/velocity vectors are polar, so p_icrs = R^T F p_model
with F = diag(-1, 1, 1) and R the ICRS -> astropy-Galactocentric rotation.
"""
from __future__ import annotations

import numpy as np

R_SUN_KPC, V_SUN = 8.178, (11.1, 12.24+240.0, 7.25)
K_PMV = 4.740470446                    # km/s per (mas/yr kpc)


def _galcen():
    import astropy.coordinates as coord
    import astropy.units as u
    return coord.Galactocentric(galcen_distance=R_SUN_KPC*u.kpc, z_sun=0*u.pc,
                                galcen_v_sun=coord.CartesianDifferential(V_SUN*u.km/u.s))


def icrs_to_model_matrix():
    """3x3 matrix M with p_model = M p_icrs for relative (polar) vectors."""
    import astropy.coordinates as coord
    import astropy.units as u
    gc = _galcen()
    pts = np.vstack((np.zeros(3), np.eye(3)))*1.0
    c = coord.SkyCoord(coord.CartesianRepresentation(pts.T*u.kpc), frame="icrs").transform_to(gc)
    g = np.vstack((c.x.to_value(u.kpc), c.y.to_value(u.kpc), c.z.to_value(u.kpc))).T
    R = (g[1:]-g[0]).T                                   # columns: images of the ICRS unit vectors
    return np.diag([-1., 1., 1.]) @ R


def sky_basis(centre_model):
    """ICRS unit vectors (e_r away from the Sun, e_E, e_N) at the cluster, and its distance [kpc]."""
    import astropy.coordinates as coord
    import astropy.units as u
    x = centre_model
    gc = coord.SkyCoord(x=-x[0]*u.kpc, y=x[1]*u.kpc, z=x[2]*u.kpc, frame=_galcen()).transform_to(coord.ICRS())
    a, d = gc.ra.rad, gc.dec.rad
    e_r = np.array([np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)])
    e_E = np.array([-np.sin(a), np.cos(a), 0.])
    e_N = np.cross(e_r, e_E)
    return e_r, e_E, e_N, float(gc.distance.kpc)


def spin_axis_icrs(centre_model, pa_deg=192., incl_towards_deg=45.):
    """Unit spin vector (ICRS): sky projection at PA pa_deg (E of N), tilted incl_towards_deg towards the Sun."""
    e_r, e_E, e_N, _ = sky_basis(centre_model)
    pa, i = np.radians(pa_deg), np.radians(incl_towards_deg)
    return np.cos(i)*(np.cos(pa)*e_N+np.sin(pa)*e_E) - np.sin(i)*e_r


def flip_probability(rmax_pc, q0, rq_pc, r1_pc=0.):
    """q0 x^2/(1+x^2) / (1+(r_max/r_q)^2) with x = r_max/r1 (no inner suppression for r1 = 0)."""
    r = np.asarray(rmax_pc, float)
    inner = 1. if r1_pc <= 0 else (r/r1_pc)**2/(1.+(r/r1_pc)**2)
    return q0*inner/(1.+(r/rq_pc)**2)


def spin_up(xv_rel_model, rmax_pc, centre_model, q0, rq_pc, pa_deg=192., incl_towards_deg=45., seed=0, r1_pc=0.):
    """Return a copy of the relative phase-space coordinates (model frame, kpc and km/s) with Lynden-Bell flips,
    and the boolean mask of flipped stars."""
    M = icrs_to_model_matrix()
    s = M @ spin_axis_icrs(centre_model, pa_deg, incl_towards_deg)      # spin direction as a model-frame polar vector
    pos_i = xv_rel_model[:, :3] @ M                                       # M^T p  (M orthogonal)
    vel_i = xv_rel_model[:, 3:] @ M
    s_i = M.T @ s
    Ls = np.einsum("ij,j->i", np.cross(pos_i, vel_i), s_i)               # right-handed L in ICRS
    ephi = np.cross(s_i[None, :], pos_i)
    n = np.linalg.norm(ephi, axis=1)
    ephi /= np.where(n > 0, n, 1.)[:, None]
    rng = np.random.default_rng(seed)
    flip = (Ls < 0) & (rng.random(len(Ls)) < flip_probability(rmax_pc, q0, rq_pc, r1_pc))
    vphi = np.einsum("ij,ij->i", vel_i, ephi)
    vel_i = vel_i - 2.*(flip*vphi)[:, None]*ephi
    out = xv_rel_model.copy()
    out[:, 3:] = vel_i @ M.T
    return out, flip


def projected_kinematics(xv_rel_model, centre_model):
    """Sky offsets [arcmin] (x = East, y = North), PMs [mas/yr] relative to the centre and v_los [km/s] relative."""
    M = icrs_to_model_matrix()
    e_r, e_E, e_N, D = sky_basis(centre_model)
    p = xv_rel_model[:, :3] @ M
    v = xv_rel_model[:, 3:] @ M
    x = p @ e_E/D*180/np.pi*60
    y = p @ e_N/D*180/np.pi*60
    return x, y, v @ e_E/(K_PMV*D), v @ e_N/(K_PMV*D), v @ e_r


def rotation_profile(x, y, pmE, pmN, vlos, edges_arcmin):
    """Mean plane-of-sky tangential PM (+ = N -> W, as the Gaia check), and the LOS amplitude/PA of max receding."""
    r = np.hypot(x, y)
    vt = (-pmE*y+pmN*x)/np.where(r > 0, r, 1.)
    out = []
    for lo, hi in zip(edges_arcmin[:-1], edges_arcmin[1:]):
        k = (r > lo) & (r < hi)
        phi = np.arctan2(x[k], y[k])
        A = np.column_stack((np.ones(k.sum()), np.sin(phi), np.cos(phi)))
        c, *_ = np.linalg.lstsq(A, vlos[k], rcond=None)
        out.append(dict(lo=lo, hi=hi, n=int(k.sum()), vtan=float(vt[k].mean()), vtan_err=float(vt[k].std()/np.sqrt(k.sum())),
                        vlos_amp=float(np.hypot(c[1], c[2])), pa_max_receding=float(np.degrees(np.arctan2(c[1], c[2])) % 360)))
    return out
