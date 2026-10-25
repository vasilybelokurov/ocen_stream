#!/usr/bin/env python3
"""Where does the JAX DOP853 spray spend its time? Per compaction call: attempts so far, active particles, array size, seconds
(jaxmodel.dop853_chi(trace=...)); also the release split (centre orbit vs the rest). Prints the share of the integration time spent
while fewer than 1%, 0.1% and 0.01% of the particles are still active (waiting for the slow ones).
Usage: python bin/streams/gpu_dop853_trace.py INPUT_DIR [POINT]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from ocen_stream import jaxspray as J, jaxmodel as JM

I = Path(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else "c0_final"
ht, inp = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz")
pts = json.loads((I/"points.json").read_text()); trel, keep = JM.schedule(inp, pts["_epoch_offset"]); p = pts[name]
for k in range(2):                                                                       # second pass = warm timings
    t1 = time.perf_counter(); tb, ic, t0, _ = JM.release(p, ht, inp, trel, keep); ic.block_until_ready(); t_rel = time.perf_counter()-t1
    tbh = J.host_for(ht, p["angle"], p["omega"], p["amp"], p["size"], p["halo_q"], p["vc"], inp["T"])
    t1 = time.perf_counter(); tc, orb = JM.centre_orbit(JM.today_xv(p["dist"], p["pmra"], p["pmdec"], inp), tbh, inp["T"], 39112); orb.block_until_ready()
    t_orb = time.perf_counter()-t1
    tr = []; t1 = time.perf_counter(); JM.dop853_chi(ic, t0, tb, 1e-8, 32, trace=tr); t_int = time.perf_counter()-t1
tr = np.array(tr); n = len(ic); sec = tr[:, 3]
print(f"{name}: release {t_rel:.2f} s (centre orbit alone {t_orb:.2f} s); DOP853 integration {t_int:.2f} s in {len(tr)} calls; slowest {int(tr[-1, 0])} attempts")
for f in (0.5, 0.1, 0.01, 0.001, 0.0001):
    k = tr[:, 1] < f*n; print(f"  time while < {f*100:g}% of particles active: {sec[k].sum():.2f} s ({100*sec[k].sum()/sec.sum():.0f}%), from attempt {int(tr[k, 0].min()) if k.any() else 0}")
print("  per-call seconds at the smallest array size:", np.median(sec[tr[:, 2] == tr[:, 2].min()]).round(4), "for", int(tr[:, 2].min()), "slots;",
      "first call (all particles):", sec[0].round(3))
