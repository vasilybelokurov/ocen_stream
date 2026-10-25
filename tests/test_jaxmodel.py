"""JAX spray model vs the AGAMA spray at final_c0_2 (GPU pipeline stage 2; needs results/gpu/{host_tables,spray_inputs,bench_inputs}.npz)."""
import json, time
from pathlib import Path
import numpy as np
import pytest

jax = pytest.importorskip("jax"); jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]; G = ROOT/"results/gpu"
pytestmark = pytest.mark.skipif(not all((G/f).exists() for f in ("host_tables.npz", "spray_inputs.npz", "bench_inputs.npz")),
                                reason="run bin/streams/gpu_host_export.py, gpu_spray_export.py, gpu_bench_export.py first")


@pytest.fixture(scope="module")
def released():
    from ocen_stream import jaxspray as J, jaxmodel as JM
    run = json.loads((ROOT/"results/streams/fitknot_20261025/runs/final_c0_2.json").read_text()); p = run["params"]
    ht, inp = J.load_host(G/"host_tables.npz"), JM.load_inputs(G/"spray_inputs.npz"); trel, keep = JM.schedule(inp, run["epoch_offset"])
    t1 = time.time(); tb, ic, t0, _ = JM.release(p, ht, inp, trel, keep); ic.block_until_ready(); print(f"release {time.time()-t1:.1f} s")
    return J, JM, p, ht, inp, tb, np.asarray(ic), np.asarray(t0), np.load(G/"bench_inputs.npz")


def test_centre_orbit_vs_agama(released):
    *_, tb, ic, t0, z = released
    """over the release window (last 700 Myr) within 0.1 pc of the AGAMA (back + forward, 1e-12) orbit, whose own closure error
    today is 0.06 pc; over the full 1.96 Gyr the table-force error accumulates to ~0.6 pc (vs AGAMA backward-only: 0.007 pc at
    700 Myr), negligible against r_J ~ 190 pc."""
    c = np.asarray(tb["orb"])[:, :3]                                                  # uniform grid, as AGAMA's
    age = (float(z["T"])-z["tc"])*977.79; d = np.linalg.norm(c-z["orb"][:, :3], axis=1)
    assert d[age < 700].max() < 1e-4 and d.max() < 2e-3, (d[age < 700].max(), d.max())


def test_release_ic_vs_agama(released):
    *_, tb, ic, t0, z = released
    assert ic.shape == z["ic"].shape and np.allclose(t0, z["t0"], rtol=0, atol=1e-12)
    dx = np.linalg.norm(ic[:, :3]-z["ic"][:, :3], axis=1); dv = np.linalg.norm(ic[:, 3:]-z["ic"][:, 3:], axis=1)
    print("ic |dx| max", dx.max(), "|dv| max", dv.max())
    assert dx.max() < 3e-4 and dv.max() < 1e-2                                       # kpc, km/s (orbit closure + host tables)
