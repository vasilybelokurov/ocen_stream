#!/usr/bin/env python3
"""Determinism of the single-model JAX chain (2026-10-25): the same parameters, NREP repeated evaluations (release + DOP853 spray +
score, bank 7) must give bit-identical final states, chi and chi2. Runs on whatever device JAX uses (CPU locally, GPU on Colab).
Usage: python bin/streams/gpu_determinism.py INPUT_DIR [POINT] [NREP]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS

I = Path(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else "c0_final"; nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
pts = json.loads((I/"points.json").read_text()); trel, keep = JM.schedule(inp, pts["_epoch_offset"]); p = pts[name]
meas = jax.jit(lambda o, c, g, b: JS.measure(o, c, g, si, b)); out = []
for r in range(nrep):
    t1 = time.perf_counter(); xv, chi, age, nat = JM.spray(p, ht, inp, trel, keep, method="dop853")
    bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(chi), 8, 7).items()}
    m, e = meas(JM.observables(xv, inp), chi, jnp.asarray(age), bank); c2 = float(JS.chi2(m, e, si))
    out.append((np.asarray(xv), np.asarray(chi), c2)); print(f"{jax.devices()[0].device_kind} {name} repeat {r}: chi2 {c2!r} ({time.perf_counter()-t1:.1f} s)", flush=True)
same = all(np.array_equal(o[0], out[0][0]) and np.array_equal(o[1], out[0][1]) and o[2] == out[0][2] for o in out[1:])
print(f"bit-identical over {nrep} repeats (final states, chi, chi2): {same}", flush=True)
