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


HUNTER24_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "oCen_bar", "agama_potentials")


def host_potential(name="McMillan17", bar_omega=None, bar_angle_deg=28.0, t_today=None, bar_amp=None, bar_eta=None, bar_size=1.0,
                   bar_model="hunter24", axi_variant=None, mass_scale=1.0):
    """AGAMA host: a bundled potential name (McMillan17), a path to an .ini file (configs/potentials/DB98_Model1.ini),
    'hunter24_axi' (Hunter+2024 axisymmetrised MW, ../oCen_bar/agama_potentials), or, with bar_omega [km/s/kpc], the Hunter+2024
    barred MW rotating at constant pattern speed. Bar convention (verified 2026-10-08; as ../chevron_bar_subhalo): AGAMA
    rotation angle = bar major axis measured anticlockwise from +x in the model frame (Sun at +x, Galactic rotation Lz > 0);
    angle(t) = bar_angle + bar_omega (t - t_today) [AGAMA time units], so the bar is at bar_angle today.
    bar_eta (optional): slowing bar with constant eta = -dOmega/dt / Omega^2 (late branch of Dillamore, Belokurov & Evans 2024,
    arXiv:2402.14907, eq. C3), normalised to bar_omega and bar_angle today: Omega(t) = bar_omega / x, x = 1 + eta bar_omega (t - t_today),
    angle(t) = bar_angle + ln(x)/eta; the bar size scales as S(t) = Omega_today/Omega(t) = x (shorter when faster), with the
    non-axisymmetric potential amplitude kept fixed (AGAMA scale [A S, S], as ../oCen_bar make_potential).
    bar_size (optional, default 1): static size factor S0 of the non-axisymmetric part (scale [A S0, S0]; longer bar for S0 > 1).
    bar_model: 'hunter24' (default) or 'portail17' = Hunter+2024 axisymmetric MW + A x (Portail17.ini - Portail17_axi.ini), the
    Portail+2017 bar as approximated by Sormani+2022 (../oCen_bar/agama_potentials; bar along x in the file, as Hunter's).
    axi_variant (optional): dict(halo_q, disc_scale, keep_vc) -> the axisymmetric part is replaced by mw_variants.axisymmetric_variant
    (flattened halo / rescaled stellar discs at fixed v_c(R0), or a halo normalised to vc_target); the bar's non-axisymmetric part
    is unchanged.
    mass_scale (optional, default 1): multiply the WHOLE host (axisymmetric part and bar term) by this factor at fixed lengths, so
    v_c scales as sqrt(mass_scale) and the bar-to-axisymmetric force ratio is unchanged."""
    agama = agama_kpc()
    if bar_omega is not None:
        if t_today is None:
            raise ValueError("a rotating bar needs t_today (AGAMA time units of the present day)")
        tt = np.linspace(-1., t_today+1., 4001)
        if bar_eta:
            x = 1.+bar_eta*bar_omega*(tt-t_today)
            if np.any(x <= 0):
                raise ValueError("slowing bar: pattern speed diverges inside the time table (eta too large)")
            ang = np.deg2rad(bar_angle_deg) + np.log(x)/bar_eta; S = bar_size*x
        else:
            ang = np.deg2rad(bar_angle_deg) + bar_omega*(tt-t_today); S = np.full_like(tt, bar_size)
        if bar_amp is None and not bar_eta and bar_size == 1.0 and bar_model == "hunter24" and axi_variant is None and mass_scale == 1.0:
            return agama.Potential(potential=agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_full.ini")),
                                   rotation=np.column_stack((tt, ang)))
        A = 1.0 if bar_amp is None else bar_amp
        # bar amplitude A (as ../oCen_bar make_potential): axisymmetric MW + A x (barred baryons - axisymmetrised baryons)
        sc = np.column_stack((tt, mass_scale*A*S, S)); scn = sc.copy(); scn[:, 1] *= -1
        full, axi = dict(hunter24=("MWPotentialHunter24_baryon_full.ini", "MWPotentialHunter24_baryon_axi.ini"),
                         portail17=("Portail17.ini", "Portail17_axi.ini"))[bar_model]
        if axi_variant is None:
            axi_pot = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
        else:
            from .mw_variants import axisymmetric_variant
            axi_pot = axisymmetric_variant(HUNTER24_DIR, **axi_variant)[0]
        if mass_scale != 1.0:
            axi_pot = agama.Potential(potential=axi_pot, scale=[mass_scale, 1.0])
        return agama.Potential(
            axi_pot,
            agama.Potential(file=os.path.join(HUNTER24_DIR, full), scale=sc, rotation=np.column_stack((tt, ang))),
            agama.Potential(file=os.path.join(HUNTER24_DIR, axi), scale=scn))
    if name == "hunter24_axi":
        return agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
    if os.path.isfile(name):
        return agama.Potential(name)
    ini = glob.glob(os.path.dirname(agama.__file__)+f"/**/{name}.ini", recursive=True)
    if not ini:
        raise FileNotFoundError(f"AGAMA potential file {name}.ini not found")
    return agama.Potential(ini[0])


N_CORE = 50           # particles inside the innermost node; constant density inside it
N_NODES = 50
N_MIN_BIN = 400       # minimum particles between consecutive nodes


def profile_nodes(rel_pos, mass):
    """Enclosed-mass nodes of a particle set: (nodes [kpc], M(<nodes) [Msun], r_in, r_out).

    Log-spaced nodes from the radius enclosing N_CORE particles to the outermost particle, merged so that
    every interval holds >= N_MIN_BIN particles (no noisy inner bins); M strictly increasing.
    """
    r = np.linalg.norm(rel_pos, axis=1)
    order = np.argsort(r)
    r, m = r[order], np.cumsum(mass[order])
    r_in, r_out = r[min(N_CORE, len(r)-1)], r[-1]
    cand = np.geomspace(r_in, r_out, N_NODES)
    counts = np.searchsorted(r, cand)
    keep = [0]
    for k in range(1, len(cand)):
        if counts[k] - counts[keep[-1]] >= N_MIN_BIN:
            keep.append(k)
    if keep[-1] != len(cand)-1:
        keep[-1] = len(cand)-1
    nodes = cand[keep]
    if len(nodes) < 4:                                   # tiny particle sets: fall back to plain log nodes
        nodes = np.geomspace(r_in, r_out, 8)
    M = np.interp(nodes, r, m)
    M = np.maximum.accumulate(M*(1+1e-12*np.arange(len(M))))
    return nodes, M, float(r_in), float(r_out)


def density_from_nodes(nodes, M):
    """rho(xyz) [Msun/kpc^3] from enclosed-mass nodes: PCHIP in (ln r, ln M) (rho >= 0), constant inside
    the first node, zero outside the last."""
    from scipy.interpolate import PchipInterpolator
    nodes, M = np.asarray(nodes, float), np.asarray(M, float)
    spl = PchipInterpolator(np.log(nodes), np.log(M))
    r_in, r_out = nodes[0], nodes[-1]
    rho_core = 3*M[0]/(4*np.pi*r_in**3)

    def rho(xyz):
        rr = np.linalg.norm(np.atleast_2d(xyz), axis=1)
        out = np.zeros(len(rr))
        out[rr <= r_in] = rho_core
        mid = (rr > r_in) & (rr < r_out)
        lr = np.log(rr[mid])
        out[mid] = np.exp(spl(lr))*spl(lr, 1)/(4*np.pi*rr[mid]**3)     # dM/dr / (4 pi r^2)
        return out
    return rho


def spherical_density(rel_pos, mass):
    """Smooth spherical density of a particle set [Msun/kpc^3 vs kpc], from its enclosed-mass profile.

    AGAMA's particle Multipole extrapolates a power law inside its innermost node, whose slope is set by a
    few particles; in the restricted runs this produced intermittent spurious central cusps that ejected
    core particles (2026-10-08 debug). Returns (callable rho(xyz), r_in, r_out).
    """
    nodes, M, r_in, r_out = profile_nodes(rel_pos, mass)
    return density_from_nodes(nodes, M), r_in, r_out


class FrozenCore:
    """Fixed spherical mass of the deeply bound particles that are not integrated (they cannot escape).

    Built once at t = 0 from those particles (relative to the satellite centre) and added to every
    satellite-potential fit. Serialised as enclosed-mass nodes (save/load).
    """

    def __init__(self, nodes, M, mass_by_species=None):
        self.nodes, self.M = np.asarray(nodes, float), np.asarray(M, float)
        self.rho = density_from_nodes(self.nodes, self.M)
        self.r_in, self.r_out = float(self.nodes[0]), float(self.nodes[-1])
        self.mass = float(self.M[-1])
        self.mass_by_species = mass_by_species or {}

    @classmethod
    def from_particles(cls, rel_pos, mass, species=None):
        nodes, M, _, _ = profile_nodes(np.asarray(rel_pos, float), np.asarray(mass, float))
        mbs = {}
        if species is not None:
            for s_ in np.unique(species):
                mbs[int(s_)] = float(np.sum(mass[species == s_]))
        return cls(nodes, M, mbs)

    def save(self, path):
        np.savez(path, nodes=self.nodes, M=self.M, species=np.array(list(self.mass_by_species.keys()), int),
                 species_mass=np.array(list(self.mass_by_species.values()), float))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["nodes"], d["M"], {int(k): float(v) for k, v in zip(d["species"], d["species_mass"])})


def satellite_potential(rel_pos, mass, grid=60, core=None):
    """Spherical potential of a particle set given relative to the satellite centre [kpc], plus an optional
    FrozenCore."""
    agama = agama_kpc()
    rel_pos, mass = np.asarray(rel_pos, float), np.asarray(mass, float)
    parts = []
    r_lo, r_hi = [], []
    if len(mass) >= 10:
        rho_a, r_in, r_out = spherical_density(rel_pos, mass)
        parts.append(rho_a); r_lo.append(r_in); r_hi.append(r_out)
    if core is not None:
        parts.append(core.rho); r_lo.append(core.r_in); r_hi.append(core.r_out)
    if not parts:
        raise ValueError("no mass for the satellite potential")

    def rho(xyz):
        return sum(p(xyz) for p in parts)
    return agama.Potential(type="multipole", density=rho, symmetry="s", lmax=0,
                           rmin=min(r_lo)/10, rmax=max(r_hi)*2, gridSizeR=grid)


def bound_set(xv, centre, mass, start=None, iterations=4, core=None):
    """Bound mask and satellite potential from the bound particles (energy in the satellite's own potential,
    including the frozen core if given). ``start`` is the previous bound mask (all particles if None).
    Iterates until the mask is stable. Returns (mask, potential).
    """
    rel = xv - centre
    bound = np.ones(len(xv), bool) if start is None else start.copy()
    pot = None
    for _ in range(iterations):
        if bound.sum() < 10 and core is None:
            break
        pot = satellite_potential(rel[bound, :3], mass[bound], core=core)
        E = pot.potential(rel[:, :3]) + 0.5*np.sum(rel[:, 3:]**2, axis=1)
        new = E < 0
        if np.array_equal(new, bound):
            break
        bound = new
    if bound.sum() >= 10 or core is not None:
        pot = satellite_potential(rel[bound, :3], mass[bound], core=core)
    return bound, pot


def rmax_of_energy(pot, E):
    """Radial-orbit apocentre [same length unit as pot] for energies E in a spherical potential."""
    rg = np.logspace(-5, 1, 800)
    Pg = pot.potential(np.column_stack((rg, 0*rg, 0*rg)))
    return np.interp(E, Pg, rg, left=0., right=np.inf)


def centre_orbit(host, ic, t0_myr, dt_myr, n):
    """Point-mass orbit of the satellite centre from t0 over n steps of dt [Myr]; returns (t_agama, xv)."""
    agama = agama_kpc()
    t, traj = agama.orbit(potential=host, ic=ic, timestart=t0_myr/AGAMA_T_MYR, time=n*dt_myr/AGAMA_T_MYR,
                          trajsize=n+1, accuracy=1e-12)     # default 1e-8 drifts 5-12 pc per 2 Gyr round trip
    return t, traj


@dataclass
class RestrictedState:
    t_myr: float
    xv: np.ndarray          # (N, 6) Galactocentric kpc, km/s
    centre: np.ndarray      # (6,)
    bound: np.ndarray       # (N,) bool


def advance(state, host, mass, sat_pot, tupd_myr, centre_dt_myr=0.05, accuracy=1e-8, frozen=False, core=None):
    """Advance all particles over one update interval in host + moving satellite; refit the satellite.

    The satellite potential ``sat_pot`` (fitted at the start of the interval) is held fixed in shape
    and moved along the centre's point-mass orbit. With ``frozen`` the satellite potential is never
    refitted (the initial one is returned again); the bound mask is still updated, in the frozen
    potential, for diagnostics. ``core`` (FrozenCore) is added to every refit.
    Returns (new_state, new_sat_pot, centre_track).
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
        bound, pot = bound_set(xv, centre, mass, start=state.bound, core=core)
    return RestrictedState(state.t_myr+tupd_myr, xv, centre, bound), pot, (tc*AGAMA_T_MYR, orb)
