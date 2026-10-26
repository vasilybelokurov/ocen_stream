"""PM knot fits over gathered particles (jaxscore.measure(cap=...)) vs over all particles: same summaries; a cap below the largest
knot count is reported (never silently truncated) and measure_capped grows it."""
import json
from pathlib import Path
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS, cspray

I = Path(__file__).resolve().parents[1]/"results"/"gpu"


@pytest.fixture(scope="module")
def model():
    if not (I/"score_inputs.npz").exists():
        pytest.skip("GPU inputs not present")
    ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); trel, keep = JM.schedule(inp, off)
    keep = keep[::4]; p = {k: float(v) for k, v in pts["c0_final"].items()}; tb, ic, t0, rj = JM.release(p, ht, inp, trel, keep)
    y, chi, _ = cspray.integrate(np.asarray(ic), np.asarray(t0), tb, JM.SPRAY_ACC)
    a = (JM.observables(jnp.asarray(y), inp), jnp.asarray(chi*JM.T_MYR), jnp.asarray((float(inp["T"])-np.asarray(t0))*JM.T_MYR))
    return a, si, {k: jnp.asarray(v) for k, v in JS.make_bank(len(keep), 8, 7).items()}


def test_capped_equals_full(model):
    a, si, bank = model
    m0, e0 = jax.jit(lambda o, c, g: JS.measure(o, c, g, si, bank))(*a)
    m1, e1 = JS.measure_capped(*a, si, bank)
    assert np.array_equal(np.isnan(m0), np.isnan(m1))
    assert np.nanmax(np.abs(m1-m0)) < 1e-8 and np.nanmax(np.abs(e1-e0)/e0) < 1e-6
    assert abs(float(JS.chi2(m1, e1, si))-float(JS.chi2(m0, e0, si))) < 1e-6


def test_small_cap_flagged_and_grown(model):
    a, si, bank = model
    _, _, nmax = jax.jit(lambda o, c, g: JS.measure(o, c, g, si, bank, cap=256))(*a)
    assert int(nmax) > 256                                                            # overflow is visible
    m1, e1 = JS.measure_capped(*a, si, bank, cap=256); m2, e2 = JS.measure_capped(*a, si, bank)
    assert np.allclose(m1, m2, rtol=0, atol=1e-8, equal_nan=True)
