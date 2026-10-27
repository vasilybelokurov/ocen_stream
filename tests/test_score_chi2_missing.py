"""jaxscore.chi2: a PM knot with a finite mean but a non-finite bootstrap error (all replicates missing the knot) counts as missing
(1e3 penalty), exactly as a missing mean. Before the fix the NaN went onto the covariance diagonal and the GPU solve returned finite
garbage (chi2 down to -1.6e5 in the 2026-10-27 mock run)."""
from pathlib import Path
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ocen_stream import jaxscore as JS

I = Path(__file__).resolve().parents[1]/"results"/"gpu"


@pytest.mark.parametrize("q", ["pmra", "pmdec"])
def test_nan_error_is_missing(q):
    if not (I/"score_inputs.npz").exists():
        pytest.skip("GPU inputs not present")
    si = JS.load(I/"score_inputs.npz"); D = si["D"]; k = D["pk"].shape[0]; na = D["pmra_k"].shape[0]
    d = np.concatenate([np.asarray(D[x]) for x in ("sky_c", "sky_s", "pmra", "pmdec")])
    m = d+0.01; e = np.full(d.size, 0.01); j = 2*k+(1 if q == "pmra" else na+1)
    mm, em = m.copy(), e.copy(); mm[j] = np.nan                                           # missing mean
    me, ee = m.copy(), e.copy(); ee[j] = np.nan                                           # finite mean, NaN error
    ref = float(JS.chi2(jnp.asarray(mm), jnp.asarray(em), si)); got = float(JS.chi2(jnp.asarray(me), jnp.asarray(ee), si))
    assert np.isfinite(got) and got == ref and got >= 1e3
    for ld in (True,):
        assert float(JS.chi2(jnp.asarray(me), jnp.asarray(ee), si, logdet=ld)) == float(JS.chi2(jnp.asarray(mm), jnp.asarray(em), si, logdet=ld))
