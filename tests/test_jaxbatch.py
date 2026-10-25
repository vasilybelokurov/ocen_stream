"""Batched models (jaxmodel.release_batch / dop853_chi(mi=...)) give the same particles as single-model runs (GPU pipeline).
Not bitwise: different array shapes change XLA's summation order, and adaptive stepping amplifies last-digit differences for
particles passing the cluster core; the bound is the roundoff class, below AGAMA 1e-8 vs 1e-10 (|dv| 99% ~0.3 km/s, 300 particles)."""
import json
from pathlib import Path
import numpy as np
import pytest

jax = pytest.importorskip("jax"); jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

G = Path(__file__).resolve().parents[1]/"results/gpu"
pytestmark = pytest.mark.skipif(not all((G/f).exists() for f in ("host_tables.npz", "spray_inputs.npz", "points.json")),
                                reason="GPU pipeline inputs missing")


def test_batch_equals_single():
    from ocen_stream import jaxspray as J, jaxmodel as JM
    pts = json.loads((G/"points.json").read_text()); ht, inp = J.load_host(G/"host_tables.npz"), JM.load_inputs(G/"spray_inputs.npz")
    trel, keep = JM.schedule(inp, pts["_epoch_offset"]); P = [pts["c0_final"], pts["start_a25"]]
    tb, ic, t0, mi, n = JM.release_batch(P, ht, inp, trel, keep)
    sub = np.random.default_rng(5).choice(n[0], 300, replace=False); rows = np.r_[sub, n[0]+sub]          # same slots in both models
    yb, cb, _ = JM.dop853_chi(ic[rows], t0[rows], tb, 1e-8, 32, mi=mi[rows])
    for k, p in enumerate(P):
        tbs, ics, t0s, _ = JM.release(p, ht, inp, trel, keep)
        assert np.allclose(np.asarray(ics[sub]), np.asarray(ic[k*n[0]+sub]), rtol=0, atol=1e-12)
        ys, cs, _ = JM.dop853_chi(ics[sub], t0s[sub], tbs, 1e-8, 32)
        dv = np.linalg.norm(yb[k*300:(k+1)*300, 3:]-ys[:, 3:], axis=1)
        print(k, "batched vs single |dv| 50/99/max [km/s]", np.percentile(dv, [50, 99, 100]), "chi sign agree", np.mean(np.sign(cb[k*300:(k+1)*300]) == np.sign(cs)))
        assert np.median(dv) < 1e-2 and np.percentile(dv, 99) < 0.3 and np.mean(np.sign(cb[k*300:(k+1)*300]) == np.sign(cs)) >= 0.99
