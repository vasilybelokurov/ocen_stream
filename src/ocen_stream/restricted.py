"""Restricted N-body for omega Cen's tidal tails (after Vasiliev, Belokurov & Erkal 2021; AGAMA
``example_tidal_stream.py``).

Particles of every species (stars, remnants, DM) are massless tracers moving in the host potential
plus the satellite potential, which moves along the satellite's centre-of-mass orbit. At each update
(interval ``tupd``) the satellite potential is refitted as a spherical multipole from the particles
that are still bound, so the mass loss of every component (including the DM) follows from the orbit
rather than from a prescribed law.

Differences from the AGAMA example: the satellite potential is built from the BOUND particles only
(iterated), not from all particles, so unbound debris does not deepen the potential used for the
bound test.

Units: AGAMA with length kpc, velocity km/s, mass Msun, so one time unit = 977.792 Myr
(``AGAMA_T_MYR``). Public interfaces take and return times in Myr.
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import os

import numpy as np

AGAMA_T_MYR = 977.792221680356        # Myr per AGAMA time unit (kpc / (km/s))
# omega Cen today, Galactocentric Cartesian (Baumgardt orbits_table.txt): X Y Z [kpc], U V W [km/s]
OCEN_TODAY = np.array([4.87, -4.07, 1.40, -95.98, -21.69, -86.82])


def agama_kpc():
    """AGAMA in kpc, km/s, Msun. Must not be mixed with the pc-unit DF code in the same process."""
    import agama
    agama.setUnits(mass=1, length=1, velocity=1)
    return agama


def host_potential(name="McMillan17"):
    agama = agama_kpc()
    ini = glob.glob(os.path.dirname(agama.__file__)+f"/**/{name}.ini", recursive=True)
    if not ini:
        raise FileNotFoundError(f"AGAMA potential file {name}.ini not found")
    return agama.Potential(ini[0])


N_CORE = 300          # particles inside the innermost node; constant density inside it
N_NODES = 50


def spherical_density(rel_pos, mass):
    """Smooth spherical density of a particle set [Msun/kpc^3 vs kpc], from its enclosed-mass profile.

    M(<r) is sampled at log-spaced nodes from the radius enclosing N_CORE particles to the outermost
    particle and interpolated with a monotone (PCHIP) spline in (ln r, ln M), so rho >= 0. Inside the
    first node the density is constant. AGAMA's particle Multipole instead extrapolates a power law
    inside its innermost node, whose slope is set by a few particles; in the restricted runs this
    produced intermittent spurious central cusps that ejected core particles (2026-10-08 debug).
    Returns (callable rho(xyz), r_in, r_out).
    """
    from scipy.interpolate import PchipInterpolator
    r = np.linalg.norm(rel_pos, axis=1)
    order = np.argsort(r)
    r, m = r[order], np.cumsum(mass[order])
    r_in, r_out = r[min(N_CORE, len(r)-1)], r[-1]
    nodes = np.geomspace(r_in, r_out, N_NODES)
    M = np.interp(nodes, r, m)
    M = np.maximum.accumulate(M*(1+1e-12*np.arange(N_NODES)))          # strictly increasing
    spl = PchipInterpolator(np.log(nodes), np.log(M))
    rho_core = 3*M[0]/(4*np.pi*r_in**3)

    def rho(xyz):
        rr = np.linalg.norm(np.atleast_2d(xyz), axis=1)
        out = np.zeros(len(rr))
        inside = rr <= r_in
        out[inside] = rho_core
        mid = (rr > r_in) & (rr < r_out)
        lr = np.log(rr[mid])
        out[mid] = np.exp(spl(lr))*spl(lr, 1)/(4*np.pi*rr[mid]**3)     # dM/dr / (4 pi r^2)
        return out
    return rho, r_in, r_out


def satellite_potential(rel_pos, mass, grid=60):
    """Spherical potential of a particle set given relative to the satellite centre [kpc]."""
    agama = agama_kpc()
    rho, r_in, r_out = spherical_density(np.asarray(rel_pos, float), np.asarray(mass, float))
    return agama.Potential(type="multipole", density=rho, symmetry="s", lmax=0,
                           rmin=r_in/10, rmax=r_out*2, gridSizeR=grid)


def bound_set(xv, centre, mass, start=None, iterations=4):
    """Bound mask and satellite potential from the bound particles (energy in the satellite's own potential).

    ``start`` is the previous bound mask (all particles if None). Iterates until the mask is stable.
    Returns (mask, potential).
    """
    rel = xv - centre
    bound = np.ones(len(xv), bool) if start is None else start.copy()
    pot = None
    for _ in range(iterations):
        if bound.sum() < 10:
            break
        pot = satellite_potential(rel[bound, :3], mass[bound])
        E = pot.potential(rel[:, :3]) + 0.5*np.sum(rel[:, 3:]**2, axis=1)
        new = E < 0
        if np.array_equal(new, bound):
            break
        bound = new
    if pot is None or bound.sum() >= 10:
        pot = satellite_potential(rel[bound, :3], mass[bound]) if bound.sum() >= 10 else pot
    return bound, pot


def centre_orbit(host, ic, t0_myr, dt_myr, n):
    """Point-mass orbit of the satellite centre from t0 over n steps of dt [Myr]; returns (t_agama, xv)."""
    agama = agama_kpc()
    t, traj = agama.orbit(potential=host, ic=ic, timestart=t0_myr/AGAMA_T_MYR, time=n*dt_myr/AGAMA_T_MYR,
                          trajsize=n+1)
    return t, traj


@dataclass
class RestrictedState:
    t_myr: float
    xv: np.ndarray          # (N, 6) Galactocentric kpc, km/s
    centre: np.ndarray      # (6,)
    bound: np.ndarray       # (N,) bool


def advance(state, host, mass, sat_pot, tupd_myr, centre_dt_myr=0.05, accuracy=1e-8, frozen=False):
    """Advance all particles over one update interval in host + moving satellite; refit the satellite.

    The satellite potential ``sat_pot`` (fitted at the start of the interval) is held fixed in shape
    and moved along the centre's point-mass orbit. With ``frozen`` the satellite potential is never
    refitted (the initial one is returned again); the bound mask is still updated, in the frozen
    potential, for diagnostics. Returns (new_state, new_sat_pot, centre_track).
    """
    agama = agama_kpc()
    n = max(2, int(round(tupd_myr/centre_dt_myr)))
    tc, orb = centre_orbit(host, state.centre, state.t_myr, tupd_myr/n, n)
    total = agama.Potential(host, agama.Potential(potential=sat_pot, center=np.column_stack((tc, orb))))
    res = agama.orbit(ic=state.xv, potential=total, timestart=state.t_myr/AGAMA_T_MYR, time=tupd_myr/AGAMA_T_MYR,
                      trajsize=1, accuracy=accuracy, verbose=False)
    xv = np.vstack(res[:, 1]).reshape(len(state.xv), 6)
    centre = orb[-1]
    if frozen:
        rel = xv - centre
        bound = sat_pot.potential(rel[:, :3]) + 0.5*np.sum(rel[:, 3:]**2, axis=1) < 0
        pot = sat_pot
    else:
        bound, pot = bound_set(xv, centre, mass, start=state.bound)
    return RestrictedState(state.t_myr+tupd_myr, xv, centre, bound), pot, (tc*AGAMA_T_MYR, orb)
