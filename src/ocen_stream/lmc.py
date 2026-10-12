"""MW-LMC interaction for the stream models: rigid MW + rigid LMC on their mutual orbit, MW reflex motion included.

Set-up of Vasiliev, Belokurov & Erkal 2021 ("Tango for three", arXiv:2009.10726, Sect. 3.1 and eq. 4; AGAMA
``py/example_lmc_mw_interaction.py``), without the deformation of either halo:
  x_MW'' = -grad Phi_LMC(x_MW - x_LMC),   x_LMC'' = -grad Phi_MW(x_LMC - x_MW) + a_DF,
integrated backward from the present-day LMC phase-space point (MW at rest at the origin today). The particles are then
integrated in the non-inertial frame centred on the MW: host + LMC moving on its relative trajectory + a spatially uniform
field equal to minus the MW-centre acceleration, so the net LMC-induced force at the MW centre is zero at all times and only
the tidal (differential) LMC field acts on omega Cen, its stream and the Sun.
LMC: truncated NFW (Tango scripts/create_lmc.py): r_s = 8.5 (M/1e11)^0.6 kpc, exponential cutoff at 10 r_s.
Dynamical friction on the LMC only: Chandrasekhar with the local MW density, the isotropic Jeans dispersion of the
sphericalised MW and the distance-dependent Coulomb logarithm ln L = sqrt(ln(d / 2 r_s)) of the AGAMA example (calibrated
there against N-body runs).
LMC present day (Tango Sect. 3.1): ICRS (81.0, -69.75) deg, PM (1.8, 0.35) mas/yr, d = 50 kpc, v_los = 260 km/s.
Units: AGAMA kpc, km/s, Msun (restricted.agama_kpc); model time t with the present day at t_today (AGAMA units).
"""
from __future__ import annotations

import functools

import numpy as np

from .restricted import agama_kpc

LMC_OBS = dict(ra=81.0, dec=-69.75, d_kpc=50.0, pmra=1.8, pmdec=0.35, vlos=260.0)   # Vasiliev+2021, Sect. 3.1


def lmc_scale_radius(mass):
    """Tango / AGAMA-example LMC scale radius [kpc] for total mass [Msun]."""
    return 8.5*(mass/1e11)**0.6


def lmc_potential(mass):
    agama = agama_kpc()
    a = lmc_scale_radius(mass)
    return agama.Potential(type="spheroid", mass=mass, scaleradius=a, outercutoffradius=10*a, gamma=1, beta=3, alpha=1)


def _sigma_spline(mw_sph):
    """Isotropic Jeans velocity dispersion sigma(r) [km/s] of a spherical potential's own density."""
    rr = np.logspace(-1, 3.5, 3000); xyz = np.column_stack((rr, 0*rr, 0*rr))
    rho = mw_sph.density(xyz); g = -mw_sph.force(xyz)[:, 0]
    integrand = rho*g
    # P(r) = int_r^inf rho g dr' (trapezoid from the outside in)
    P = np.concatenate((np.cumsum((0.5*(integrand[1:]+integrand[:-1])*np.diff(rr))[::-1])[::-1], [0.]))
    sig = np.sqrt(np.maximum(P, 0)/np.maximum(rho, 1e-300))
    return lambda r: np.interp(r, rr, sig)


@functools.lru_cache(maxsize=16)
def mw_lmc_orbit(mass, t_today, mw_file=None, frame="baumgardt", t_back=3.2, t_fwd=0.1, step=1/512, obs=None):
    """Mutual MW-LMC orbit. mass [Msun]; t_today, t_back, t_fwd, step in AGAMA time units (977.8 Myr); mw_file: AGAMA .ini
    of the (axisymmetric) MW used for the mutual orbit (default Hunter+2024 axisymmetrised MW); obs: tuple of LMC_OBS items
    to override. Returns dict(t, lmc (N, 6) relative to the MW centre, acc (N, 3) = minus the MW-centre acceleration,
    mw (N, 6) MW centre in the inertial frame, today's LMC model-frame vector). Times are model times (t_today = present)."""
    import os
    import scipy.integrate
    import scipy.special
    from .frames import to_model
    from .restricted import HUNTER24_DIR
    agama = agama_kpc()
    o = dict(LMC_OBS, **dict(obs or ()))
    mw = agama.Potential(file=mw_file or os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
    sig = _sigma_spline(agama.Potential(type="Multipole", potential=mw, lmax=0, rmin=0.01, rmax=1000))
    lmc = lmc_potential(mass); bmin = 2*lmc_scale_radius(mass)
    x0 = to_model(o["ra"], o["dec"], o["d_kpc"], o["pmra"], o["pmdec"], o["vlos"], frame)[0]

    def rhs(t, v):
        dx = v[6:9]-v[0:3]; dv = v[9:12]-v[3:6]
        d = np.sqrt(dx@dx); vm = np.sqrt(dv@dv)
        X = vm/(np.sqrt(2)*sig(d)); lnL = np.sqrt(max(0., np.log(d/bmin)))
        a_df = -4*np.pi*mw.density(dx)*dv/vm*(scipy.special.erf(X)-2/np.sqrt(np.pi)*X*np.exp(-X*X))*mass*agama.G**2/vm**2*lnL
        return np.concatenate((v[3:6], lmc.force(-dx), v[9:12], mw.force(dx)+a_df))

    y0 = np.concatenate((np.zeros(6), x0)); out = []
    for t_end in (-t_back, t_fwd):   # backward and forward from today (relative time 0)
        n = int(round(abs(t_end)/step))+1; tg = np.linspace(0., t_end, n)
        s = scipy.integrate.solve_ivp(rhs, (0., t_end), y0, t_eval=tg, rtol=1e-10, atol=1e-10, method="DOP853")
        out.append((tg, s.y.T))
    tg = np.concatenate((out[0][0][::-1], out[1][0][1:])); y = np.vstack((out[0][1][::-1], out[1][1][1:]))
    rel = y[:, 6:12]-y[:, 0:6]
    acc = -lmc.force(-rel[:, :3])    # minus the MW-centre acceleration (rigid LMC pulling the rigid MW)
    return dict(t=tg+t_today, lmc=rel, acc=acc, mw=y[:, 0:6], today=x0, mass=mass)


def with_lmc(host, t_today, mass=1.5e11, **kw):
    """Host potential in the MW-centred non-inertial frame with the LMC: host + moving LMC + uniform reflex field.
    kw: passed to mw_lmc_orbit (mw_file, frame, t_back, t_fwd, step, obs as a tuple of (key, value))."""
    agama = agama_kpc()
    if not mass:
        return host
    orb = mw_lmc_orbit(float(mass), float(t_today), **kw)
    lmc_moving = agama.Potential(potential=lmc_potential(mass), center=np.column_stack((orb["t"], orb["lmc"])))
    reflex = agama.Potential(type="UniformAcceleration", file=np.column_stack((orb["t"], orb["acc"])))
    return agama.Potential(host, lmc_moving, reflex)
