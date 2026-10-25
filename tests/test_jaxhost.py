"""JAX host at arbitrary parameter points vs AGAMA host_potential (GPU pipeline stage 1; needs results/gpu/host_tables.npz)."""
from pathlib import Path
import numpy as np
import pytest

jax = pytest.importorskip("jax"); jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]; HT = ROOT/"results/gpu/host_tables.npz"
pytestmark = pytest.mark.skipif(not HT.exists(), reason="run bin/streams/gpu_host_export.py first")
T = 2.0


def _points(rng, n=4000):
    """positions in the stream / orbit volume: R 0.5-12 kpc, |z| < 4 kpc, any azimuth; times over the last Gyr."""
    R, ph, z = rng.uniform(0.5, 12, n), rng.uniform(0, 2*np.pi, n), rng.uniform(-4, 4, n)
    return np.column_stack((R*np.cos(ph), R*np.sin(ph), z)), rng.uniform(T-1.1, T, n)


@pytest.mark.parametrize("seed", range(10))
def test_host_force_random_parameters(seed):
    from ocen_stream import jaxspray as J
    from ocen_stream.restricted import host_potential
    rng = np.random.default_rng(seed)
    p = dict(angle=rng.uniform(0, 50), omega=rng.uniform(20, 50), amp=rng.uniform(0.6, 2.0), size=rng.uniform(0.8, 1.5),
             q=rng.uniform(0.65, 1.25), vc=rng.uniform(205, 250))
    if seed == 0:
        p["q"] = 0.725                                                                   # between stack nodes
    host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=p["amp"], bar_size=p["size"],
                          axi_variant=dict(halo_q=p["q"], vc_target=p["vc"], r0_kpc=float(np.load(HT)["r0"])))
    tb = J.host_for(J.load_host(HT), p["angle"], p["omega"], p["amp"], p["size"], p["q"], p["vc"], T)
    x, t = _points(rng)
    F = host.force(x, t=t); Fj = np.asarray(-jax.grad(lambda y: jnp.sum(J.phi_host(y, jnp.asarray(t), tb)))(jnp.asarray(x)))
    rel = np.linalg.norm(Fj-F, axis=1)/np.linalg.norm(F, axis=1)
    assert np.percentile(rel, 99) < 1e-5 and rel.max() < 1e-4, (p, np.percentile(rel, [50, 99]), rel.max())
