"""JAX spray tables and integrators vs AGAMA at the benchmark point (needs results/gpu/bench_inputs.npz from gpu_bench_export.py)."""
import json
from pathlib import Path
import numpy as np
import pytest

jax = pytest.importorskip("jax"); jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]; INP = ROOT/"results/gpu/bench_inputs.npz"
pytestmark = pytest.mark.skipif(not INP.exists(), reason="run bin/streams/gpu_bench_export.py first")
NP = 300                                                                                   # particles in the integrator tests


@pytest.fixture(scope="module")
def setup():
    from ocen_stream import jaxspray as J
    from ocen_stream.restricted import host_potential, satellite_potential, FrozenCore, agama_kpc
    from ocen_stream.frames import frame_r0
    import grid4_common as G4
    agama = agama_kpc(); z = np.load(INP); P = json.loads(str(z["params"]))
    host = host_potential("x", bar_omega=P["omega"], bar_angle_deg=P["angle"], t_today=float(z["T"]), bar_amp=P["amp"], bar_size=P["size"],
                          axi_variant=dict(halo_q=P["halo_q"], vc_target=P["vc"], r0_kpc=frame_r0("gravity21")))
    sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=FrozenCore(G4.r_kpc, G4.M))
    total = agama.Potential(host, agama.Potential(potential=sat, center=np.column_stack((z["tc"], z["orb"]))))
    sel = np.random.default_rng(1).choice(len(z["ic"]), NP, replace=False)
    return J, J.load(INP), z, host, sat, total, sel


def test_spline_matches_scipy(setup):
    J, tb, z, *_ = setup
    rng = np.random.default_rng(2); x, y = rng.uniform(0, z["host_coef"].shape[1]-1, 500), rng.uniform(0, z["host_coef"].shape[2]-1, 500)
    ref = map_coordinates(z["host_coef"][1], [x, y], order=3, mode="mirror", prefilter=False)
    got = np.asarray(J._spline2(tb["host"][1], jnp.asarray(x), jnp.asarray(y), tb["nu"], tb["nw"]))
    assert np.max(np.abs(got-ref)) < 1e-9*np.max(np.abs(ref))


def test_host_force_vs_agama(setup):
    J, tb, z, host, *_ = setup
    rng = np.random.default_rng(3); i = rng.choice(len(z["ic"]), 5000); x = np.r_[z["xv_agama8"][i, :3], z["ic"][i, :3]]
    t = np.r_[np.full(5000, float(z["T"])), z["t0"][i]]
    F = host.force(x, t=t); Fj = np.asarray(-jax.grad(lambda y: jnp.sum(J.phi_host(y, jnp.asarray(t), tb)))(jnp.asarray(x)))
    rel = np.linalg.norm(Fj-F, axis=1)/np.linalg.norm(F, axis=1)
    assert np.percentile(rel, 99) < 1e-5 and rel.max() < 1e-4, (np.percentile(rel, [50, 99]), rel.max())


def test_progenitor_centre_and_potential(setup):
    J, tb, z, host, sat, *_ = setup
    agama = __import__("agama"); t = np.sort(np.random.default_rng(4).uniform(z["tc"][0], z["tc"][-1], 50))
    o = np.vstack([agama.orbit(potential=host, ic=z["orb"][0], timestart=z["tc"][0], time=tt-z["tc"][0], trajsize=1, accuracy=1e-12)[1]
                   for tt in t])
    assert np.max(np.linalg.norm(np.asarray(J.centre(jnp.asarray(t), tb))-o[:, :3], axis=1)) < 1e-6        # kpc
    r = np.geomspace(1e-4, 40, 300); x = np.column_stack((r, 0*r, 0*r)); c = np.asarray(J.centre(jnp.full(300, z["tc"][0]), tb))
    got = np.asarray(J.phi_sat(jnp.asarray(x+c), jnp.full(300, z["tc"][0]), tb))
    assert np.max(np.abs(got/sat.potential(x)-1)) < 1e-6


def test_total_force_vs_agama(setup):
    J, tb, z, host, sat, total, sel = setup
    x, t = z["ic"][sel, :3]+1e-3, z["t0"][sel]                                                # 1 pc off the release point
    F = total.force(x, t=t); Fj = np.asarray(J.accel(jnp.asarray(x), jnp.asarray(t), tb))
    assert np.percentile(np.linalg.norm(Fj-F, axis=1)/np.linalg.norm(F, axis=1), 99) < 1e-4


@pytest.mark.parametrize("method,kw", [("yoshida4", dict(dt=0.2/977.79)), ("dopri8", dict(rtol=1e-8, atol=1e-8))])
def test_integrator_vs_agama(setup, method, kw):
    """bulk of the particles within 0.01 km/s of AGAMA 1e-10 (~5e-4 mas/yr at 5 kpc); AGAMA 1e-8 itself is at ~1e-3 km/s."""
    J, tb, z, *_, sel = setup
    xv, _ = J.run(method, z["ic"][sel], z["t0"][sel], tb, **kw)
    dv = np.linalg.norm(np.asarray(xv)[:, 3:]-z["xv_agama10"][sel, 3:], axis=1)
    d8 = np.linalg.norm(z["xv_agama8"][sel, 3:]-z["xv_agama10"][sel, 3:], axis=1)
    print(method, "dv pct 50/90/99", np.percentile(dv, [50, 90, 99]), "agama 1e-8", np.percentile(d8, [50, 90, 99]))
    assert np.percentile(dv, 90) < 1e-2
