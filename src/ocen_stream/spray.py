"""Particle-spray streams of omega Cen in a (possibly time-dependent) host (fitting campaign, Stage 1).

Release prescription: Fardal, Huang & Weinberg 2015 (arXiv:1410.1861) with the gala modifications, as implemented in AGAMA's
tutorial_streams.ipynb (create_ic_particle_spray): offsets in the satellite frame
  x = (2.0 + 0.5 N) r_J, z = 0.5 N r_J, v_y = (0.3 + 0.5 N) v_J x/r_J, v_z = 0.5 N v_J
at both Lagrange points. Changes for omega Cen:
  - the host is evaluated at each release time (rotating bar): d2Phi/dr2 from pot.eval(..., der=True, t=t_k);
  - r_J is solved self-consistently with the satellite's enclosed mass M(< r_J) from its mass profile
    (r_J^3 = G M(<r_J) / (Omega^2 - d2Phi/dr2), Omega = |L|/r^2), instead of a single total mass;
  - released particles move in host + the satellite's (fixed) potential carried along the centre orbit (Gibbons+2014: the
    progenitor's gravity is essential for the stream length);
  - each particle carries its arm (+1 = outer Lagrange point = trailing, -1 = leading) and release time.
Units: AGAMA kpc, km/s, Msun; time in AGAMA units (977.79 Myr), t = 0 at the start, t = T today.
"""
from __future__ import annotations

import numpy as np

from .restricted import AGAMA_T_MYR, FrozenCore, agama_kpc, satellite_potential


def rj_vj_R(host, orbit, times, m_of_r, n_iter=30):
    """Jacobi radius r_J [kpc], v_J = Omega r_J and host->satellite rotation matrices along the orbit (N, 6) at times (N,)."""
    agama = agama_kpc()
    x, y, z, vx, vy, vz = orbit.T
    Lx, Ly, Lz = y*vz-z*vy, z*vx-x*vz, x*vy-y*vx
    r = np.sqrt(x*x+y*y+z*z); L = np.sqrt(Lx*Lx+Ly*Ly+Lz*Lz)
    R = np.zeros((len(r), 3, 3))
    R[:, 0] = np.column_stack((x, y, z))/r[:, None]
    R[:, 2] = np.column_stack((Lx, Ly, Lz))/L[:, None]
    R[:, 1] = np.cross(R[:, 2], R[:, 0])
    der = host.eval(orbit[:, :3], der=True, t=times)
    d2 = -(x**2*der[:, 0]+y**2*der[:, 1]+z**2*der[:, 2]+2*x*y*der[:, 3]+2*y*z*der[:, 4]+2*z*x*der[:, 5])/r**2
    Om = L/r**2
    denom = np.maximum(Om**2-d2, 1e-6)
    rj = np.full(len(r), 0.1)
    for _ in range(n_iter):
        rj = (agama.G*m_of_r(rj)/denom)**(1/3)
    return rj, Om*rj, R


def release_ic(orbit, rj, vj, R, rng, spin=None):
    """Fardal+15 (gala-modified) initial conditions, interleaved (trailing, leading) per release point.
    spin (optional, our extension, not part of Fardal+15): dict(r_kpc, vrot_kms, s) = the cluster's mean rotation profile
    v_rot(r) about the unit spin vector s (model frame, fixed in the Galactic frame), measured with e_phi = s x x / |s x x|
    (bin/streams/spin_vrot_profile.py). Each released particle gets + v_rot(r_J) e_phi(x_offset): the mean streaming velocity
    of the cluster at its tidal radius, in the rotation direction at the release point."""
    N = len(rj)
    s = np.tile([1., -1.], N)
    rj2, vj2, R2 = np.repeat(rj, 2)*s, np.repeat(vj, 2)*s, np.repeat(R, 2, axis=0)
    rx = rng.normal(size=2*N)*0.5+2.0
    rz = rng.normal(size=2*N)*0.5*rj2
    rvy = (rng.normal(size=2*N)*0.5+0.3)*vj2*rx
    rvz = rng.normal(size=2*N)*0.5*vj2
    rx = rx*rj2
    ic = np.repeat(orbit, 2, axis=0).copy()
    ic[:, :3] += np.einsum("ni,nij->nj", np.column_stack((rx, 0*rx, rz)), R2)
    ic[:, 3:] += np.einsum("ni,nij->nj", np.column_stack((0*rx, rvy, rvz)), R2)
    if spin is not None:
        off = ic[:, :3]-np.repeat(orbit[:, :3], 2, axis=0)
        ephi = np.cross(np.asarray(spin["s"])[None, :], off); ephi /= np.linalg.norm(ephi, axis=1)[:, None]
        vr = np.interp(np.abs(rj2), spin["r_kpc"], spin["vrot_kms"])
        ic[:, 3:] += vr[:, None]*ephi
    return ic, s.astype(np.int8)


def spray(host, today, T, r_kpc, M_enc, n_release=2000, t_release=None, seed=1, accuracy=1e-8, self_gravity=True):
    """Particle-spray stream at time T (today).
    host: agama.Potential (time-dependent allowed); today: (6,) present-day phase-space point; T: duration [AGAMA units];
    r_kpc, M_enc: satellite enclosed-mass profile; n_release: number of release epochs (2 particles each), uniform in time
    over (0, T) unless t_release (array, AGAMA units) is given.
    Returns dict: xv (2N, 6) today, arm (+1 trailing / -1 leading), t_release [Myr ago], centre track (t, orbit)."""
    agama = agama_kpc()
    rng = np.random.default_rng(seed)
    _, back = agama.orbit(potential=host, ic=today, timestart=T, time=-T, trajsize=2, accuracy=1e-12)
    start = back[-1]
    nt = int(np.ceil(T*AGAMA_T_MYR/0.05))+1
    tc, orb = agama.orbit(potential=host, ic=start, timestart=0., time=T, trajsize=nt, accuracy=1e-12)
    tr = np.sort(rng.uniform(0, T, n_release)) if t_release is None else np.sort(np.asarray(t_release))
    orbit_r = np.array([np.interp(tr, tc, orb[:, i]) for i in range(6)]).T
    m_of_r = lambda rr: np.interp(np.log(rr), np.log(r_kpc), M_enc)
    rj, vj, R = rj_vj_R(host, orbit_r, tr, m_of_r)
    ic, arm = release_ic(orbit_r, rj, vj, R, rng)
    t0 = np.repeat(tr, 2)
    if self_gravity:
        sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=FrozenCore(r_kpc, M_enc))
        total = agama.Potential(host, agama.Potential(potential=sat, center=np.column_stack((tc, orb))))
    else:
        total = host
    res = agama.orbit(potential=total, ic=ic, timestart=t0, time=T-t0, trajsize=1, accuracy=accuracy, verbose=False)
    xv = np.vstack(res[:, 1])
    return dict(xv=xv, arm=arm, t_release_myr_ago=(T-t0)*AGAMA_T_MYR, rj_kpc=np.repeat(rj, 2), centre_today=orb[-1],
                centre_offset_pc=float(np.linalg.norm(orb[-1, :3]-today[:3])*1e3))


def spray_unwrapped(host, today, T, r_kpc, M_enc, t_release, ntraj=240, seed=1, accuracy=1e-8, spin=None):
    """Spray (as spray(), progenitor gravity included) with per-particle trajectories, returning stream-ordering coordinates:
      chi [kpc Myr]: Gibbons+2014 phase, integral over release->today of (|r| - |r_prog|) dt (their Eq. 3 sums over equal
            time steps; this is the time-weighted continuous form; > 0 trailing, < 0 leading);
      psi [deg]: unwrapped angle along the stream in the progenitor's present orbital plane (Chemaly+2026 style, in 3D):
            psi = Psi_prog(t_rel) + [unwrapped particle angle change from t_rel to today] - Psi_prog(today), with Psi the
            unwrapped in-plane azimuth; psi < 0 trailing for prograde-in-plane motion.
    t_release: release epochs [AGAMA units] (two particles each). ntraj: trajectory samples per particle (equal steps over its own
    duration). spin: optional cluster rotation (see release_ic)."""
    agama = agama_kpc()
    rng = np.random.default_rng(seed)
    _, back = agama.orbit(potential=host, ic=today, timestart=T, time=-T, trajsize=2, accuracy=1e-12)
    start = back[-1]
    nt = int(np.ceil(T*AGAMA_T_MYR/0.05))+1
    tc, orb = agama.orbit(potential=host, ic=start, timestart=0., time=T, trajsize=nt, accuracy=1e-12)
    tr = np.sort(np.asarray(t_release))
    orbit_r = np.array([np.interp(tr, tc, orb[:, i]) for i in range(6)]).T
    m_of_r = lambda rr: np.interp(np.log(rr), np.log(r_kpc), M_enc)
    rj, vj, R = rj_vj_R(host, orbit_r, tr, m_of_r)
    ic, arm = release_ic(orbit_r, rj, vj, R, rng, spin=spin)
    t0 = np.repeat(tr, 2)
    sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=FrozenCore(r_kpc, M_enc))
    total = agama.Potential(host, agama.Potential(potential=sat, center=np.column_stack((tc, orb))))
    res = agama.orbit(potential=total, ic=ic, timestart=t0, time=T-t0, trajsize=ntraj, accuracy=accuracy, verbose=False)
    # orbital-plane basis from the progenitor's present angular momentum
    Lp = np.cross(orb[-1, :3], orb[-1, 3:]); ez = Lp/np.linalg.norm(Lp)
    ex = orb[-1, :3]-np.dot(orb[-1, :3], ez)*ez; ex /= np.linalg.norm(ex); ey = np.cross(ez, ex)
    ang = lambda x: np.arctan2(x @ ey, x @ ex)
    Psi_prog = np.unwrap(ang(orb[:, :3]))
    rprog = np.linalg.norm(orb[:, :3], axis=1)
    xv = np.zeros((len(ic), 6)); chi = np.zeros(len(ic)); psi = np.zeros(len(ic))
    for i in range(len(ic)):
        tt, tj = res[i, 0], res[i, 1]
        xv[i] = tj[-1]
        dr = np.linalg.norm(tj[:, :3], axis=1)-np.interp(tt, tc, rprog)
        chi[i] = np.trapezoid(dr, tt)*AGAMA_T_MYR
        a = np.unwrap(ang(tj[:, :3]))
        psi[i] = np.degrees(np.interp(tt[0], tc, Psi_prog)+(a[-1]-a[0])-Psi_prog[-1])
    return dict(xv=xv, arm=arm, chi=chi, psi=psi, t_release_myr_ago=(T-t0)*AGAMA_T_MYR, centre_today=orb[-1])
