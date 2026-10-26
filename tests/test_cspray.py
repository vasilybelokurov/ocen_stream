"""C spray library (cspray / csrc/spray.c) vs the JAX implementation (GPU pipeline step 1, 2026-10-26).

Bit-identity with JAX is impossible (different rounding, amplified by passages through the cluster core), so the integration gates are
statistical, set by the measured platform noise (JAX GPU vs JAX CPU on the same particles: chi sign agreement 98.8%, |dchi| median
1.5e-3 kpc Myr). Determinism of the C code itself is exact (repeats, thread chunking)."""
import json
from pathlib import Path
import numpy as np
import pytest

jax = pytest.importorskip("jax"); jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

G = Path(__file__).resolve().parents[1]/"results/gpu"
pytestmark = pytest.mark.skipif(not all((G/f).exists() for f in ("host_tables.npz", "spray_inputs.npz", "points.json")), reason="inputs missing")


@pytest.fixture(scope="module")
def setup():
    from ocen_stream import jaxspray as J, jaxmodel as JM, cspray as CS
    pts = json.loads((G/"points.json").read_text()); ht, inp = J.load_host(G/"host_tables.npz"), JM.load_inputs(G/"spray_inputs.npz")
    trel, keep = JM.schedule(inp, pts["_epoch_offset"]); tb, ic, t0, _ = JM.release(pts["c0_final"], ht, inp, trel, keep)
    return J, JM, CS, tb, np.asarray(ic), np.asarray(t0)


def test_force_vs_autodiff(setup):
    J, JM, CS, tb, ic, t0 = setup
    import ctypes as C
    rng = np.random.default_rng(1); i = rng.choice(len(ic), 3000, replace=False); x = np.ascontiguousarray(ic[i, :3]+rng.normal(0, 0.3, (3000, 3))); t = np.ascontiguousarray(t0[i])
    T_, keep = CS.pack(tb); a, pot, dchi = np.zeros((3000, 3)), np.zeros(3000), np.zeros(3000)
    CS.lib().force_many(C.byref(T_), 3000, CS._p(x), CS._p(t), CS._p(a), CS._p(pot), CS._p(dchi))
    aj = np.asarray(J.accel(jnp.asarray(x), jnp.asarray(t), tb)); rel = np.linalg.norm(a-aj, axis=1)/np.linalg.norm(aj, axis=1)
    assert np.median(rel) < 1e-12 and rel.max() < 1e-9, (np.median(rel), rel.max())


def test_c_deterministic(setup):
    J, JM, CS, tb, ic, t0 = setup
    s = np.arange(0, len(ic), 700)
    a = CS.integrate(ic[s], t0[s], tb, 1e-6, nthreads=1); b = CS.integrate(ic[s], t0[s], tb, 1e-6, nthreads=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_c_vs_jax_and_handoff(setup):
    J, JM, CS, tb, ic, t0 = setup
    fin = np.load(G/"tolscan/c0_final_acc1e-06.npz")["finish"] if (G/"tolscan/c0_final_acc1e-06.npz").exists() else None
    rng = np.random.default_rng(2); s = rng.choice(len(ic), 480, replace=False)
    if fin is not None:
        s = np.unique(np.r_[s, np.argsort(fin)[-32:]])                                     # include the 32 slowest particles
    yj, cj, _ = JM.dop853_chi(ic[s], t0[s], tb, 1e-6)
    yc, cc, nc = CS.integrate(ic[s], t0[s], tb, 1e-6)
    yh, ch, _ = JM.dop853_chi(ic[s], t0[s], tb, 1e-6, handoff=256)                         # JAX bulk, C finishes the rest
    for lab, y, c in (("C", yc, cc*JM.T_MYR), ("hand-off", yh, ch)):
        dx = np.linalg.norm(y[:, :3]-yj[:, :3], axis=1); sg = np.mean(np.sign(c) == np.sign(cj)); dc = np.abs(c-cj)
        print(lab, "vs JAX: |dx| median", np.median(dx), "max", dx.max(), "| chi sign agree", sg, "| |dchi| median", np.median(dc))
        assert np.median(dx) < 1e-3 and sg >= 0.985 and np.median(dc) < 1e-2, lab
    assert (nc > 0).all()


def test_lib_build_thread_safe(tmp_path, monkeypatch):
    """first use from several threads at once (gpupipe.release_many) must build the library once, without a rename race."""
    from concurrent.futures import ThreadPoolExecutor
    from ocen_stream import cspray
    monkeypatch.setenv("OCEN_CACHE", str(tmp_path)); monkeypatch.setattr(cspray, "_LIB", None)
    with ThreadPoolExecutor(8) as ex:
        libs = list(ex.map(lambda _: cspray.lib(), range(8)))
    assert all(L is libs[0] for L in libs)
