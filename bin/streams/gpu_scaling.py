#!/usr/bin/env python3
"""Single-model JAX chain on the current device (2026-10-26): determinism and time vs particle number, tolerance SPRAY_ACC (1e-6).

(1) Determinism: NREP evaluations of the same model at the standard nrel = 256000 (179200 integrated particles) must give bit-identical
    final states, chi and chi2.
(2) Scaling: nrel = 128000 / 256000 / 512000 (about 90k / 179k / 358k integrated particles); release banks for the non-standard sizes
    are numpy default_rng(SEED) draws of the right length (as gpu_spray_export.py). Warm timings (first call reported separately),
    split into release, DOP853 integration and score (bank 7), plus chi2.
Usage: python bin/streams/gpu_scaling.py INPUT_DIR [POINT] [NREP]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS

I = Path(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else "c0_final"; nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ht, inp0, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
pts = json.loads((I/"points.json").read_text()); p = pts[name]; seed = int(np.load(I/"spray_inputs.npz")["seed"])
meas = jax.jit(lambda o, c, g, b: JS.measure(o, c, g, si, b)); dev = jax.devices()[0].device_kind
sync = lambda x: jax.tree_util.tree_map(lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, x)
print(f"{dev}; tolerance {JM.SPRAY_ACC}; point {name}", flush=True)


def chain(inp, trel, keep):
    t = [time.perf_counter()]; tb, ic, t0, _ = sync(JM.release(p, ht, inp, trel, keep)); t.append(time.perf_counter())
    y, chi, natt = JM.dop853_chi(ic, t0, tb); t.append(time.perf_counter())
    bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(chi), 8, 7).items()}
    m, e = sync(meas(JM.observables(jnp.asarray(y), inp), jnp.asarray(chi), (inp["T"]-t0)*JM.T_MYR, bank)); c2 = float(JS.chi2(m, e, si))
    t.append(time.perf_counter()); return y, chi, c2, np.diff(t), len(ic), natt


for nrel in (256000, 128000, 512000):
    inp = dict(inp0)
    if nrel != inp0["nrel"]:
        rng = np.random.default_rng(seed); inp["bank"] = jnp.asarray(np.stack([rng.normal(size=2*nrel) for _ in range(4)])); inp["nrel"] = nrel
    trel, keep = JM.schedule(inp, pts["_epoch_offset"])
    y0, c0, x0, dt0, n, _ = chain(inp, trel, keep)                                        # first call (compiles new shapes)
    reps = [chain(inp, trel, keep) for _ in range(nrep if nrel == 256000 else 1)]
    y, chi, c2, dt, n, natt = reps[-1]
    print(f"nrel {nrel}: {n} particles integrated; release {dt[0]:.2f} s, integration {dt[1]:.2f} s (slowest {natt} attempts), score {dt[2]:.2f} s, "
          f"total {dt.sum():.2f} s (first call {dt0.sum():.0f} s); chi2 {c2!r}", flush=True)
    if nrel == 256000:
        same = all(np.array_equal(r[0], y0) and np.array_equal(r[1], c0) and r[2] == x0 for r in reps)
        print(f"determinism: {nrep + 1} evaluations bit-identical (final states, chi, chi2): {same}; chi2 values {[r[2] for r in reps]}", flush=True)
