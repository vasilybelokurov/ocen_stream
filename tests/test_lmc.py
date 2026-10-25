"""Tests of the MW-LMC set-up (src/ocen_stream/lmc.py): present-day state, frame consistency (zero net LMC force at the
MW centre), the moving-LMC field, the zero-mass limit, and the LMC pericentre against Vasiliev+2021 (Tango, 47.1-47.4 kpc,
53-61 Myr ago in their N-body runs, Zenodo 4300977 trajlmc.txt)."""
import os
import numpy as np
import pytest

agama = pytest.importorskip("agama")
from ocen_stream.restricted import agama_kpc, HUNTER24_DIR, AGAMA_T_MYR, host_potential
from ocen_stream.frames import to_model
from ocen_stream.lmc import mw_lmc_orbit, with_lmc, lmc_potential, LMC_OBS

T = 1955.58/AGAMA_T_MYR
M = 1.5e11


@pytest.fixture(scope="module")
def axi():
    agama_kpc()
    return agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))


@pytest.fixture(scope="module")
def orb():
    return mw_lmc_orbit(M, T)


def test_today(orb):
    i = np.argmin(abs(orb["t"]-T))
    assert abs(orb["t"][i]-T) < 1e-12
    o = LMC_OBS
    x0 = to_model(o["ra"], o["dec"], o["d_kpc"], o["pmra"], o["pmdec"], o["vlos"])[0]
    np.testing.assert_allclose(orb["lmc"][i], x0, atol=1e-8)
    np.testing.assert_allclose(orb["mw"][i], 0., atol=1e-8)
    assert orb["t"][0] < 0 and orb["t"][-1] > T     # covers the spray (t >= 0) and today


def test_net_force_at_centre_is_zero(axi):
    tot = with_lmc(axi, T, mass=M)
    for t in np.linspace(0.1, T, 7):
        f_lmc = np.linalg.norm(tot.force(np.zeros(3), t=t) - axi.force(np.zeros(3)))
        f_scale = np.linalg.norm(lmc_potential(M).force(np.array([50., 0, 0])))
        assert f_lmc < 1e-3*f_scale, (t, f_lmc, f_scale)


def test_tidal_field(axi, orb):
    """with_lmc - host at x equals F_LMC(x - x_LMC(t)) - F_LMC(-x_LMC(t)) (moving centre interpolated correctly)."""
    tot = with_lmc(axi, T, mass=M); lmc = lmc_potential(M)
    rng = np.random.default_rng(1)
    for t in (0.5, 1.5, T-0.05, T):
        xl = np.array([np.interp(t, orb["t"], orb["lmc"][:, k]) for k in range(3)])
        for x in rng.normal(scale=6., size=(5, 3)):
            want = lmc.force(x-xl) - lmc.force(-xl)
            got = tot.force(x, t=t) - axi.force(x)
            np.testing.assert_allclose(got, want, atol=1e-3*np.linalg.norm(lmc.force(-xl))+1e-6)


def test_zero_mass_is_host(axi):
    assert with_lmc(axi, T, mass=0.) is axi


def test_host_potential_option():
    h0 = host_potential("x", bar_omega=34., bar_angle_deg=16., t_today=T, bar_amp=1.2)
    h1 = host_potential("x", bar_omega=34., bar_angle_deg=16., t_today=T, bar_amp=1.2, lmc=dict(mass=M))
    x = np.array([4.87, -4.07, 1.40])
    np.testing.assert_allclose(h0.force(x, t=T), host_potential("x", bar_omega=34., bar_angle_deg=16., t_today=T, bar_amp=1.2).force(x, t=T))
    d = np.linalg.norm(h1.force(x, t=T)-h0.force(x, t=T))
    assert 1. < d < 100.   # tidal LMC field at omega Cen today: tens of (km/s)^2/kpc


def test_pericentre(orb):
    d = np.linalg.norm(orb["lmc"][:, :3], axis=1); m = (orb["t"] > T-1.) & (orb["t"] <= T)
    i = np.argmin(np.where(m, d, np.inf)); age = (T-orb["t"][i])*AGAMA_T_MYR
    assert 44. < d[i] < 51. and 35. < age < 80., (d[i], age)
