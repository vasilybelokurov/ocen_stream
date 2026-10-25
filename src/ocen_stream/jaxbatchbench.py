"""Timing of batched model evaluations on the GPU (2026-10-25): K models (MCMC walkers) per DOP853 integration.

For K in KS, the K models cycle through the test points (points.json); after a warm-up call, a timed call of the whole chain:
release of all K (orbits on the CPU), one DOP853 integration of all particles (jaxmodel.spray_batch), and the score of every model
(jaxscore, fixed per-slot bank 7). Reported: seconds per model, and each model's chi2 against its value at K = 1 (batching must not
change chi2 beyond roundoff noise). Uses a persistent compilation cache (--cache).
Usage: python -m ocen_stream.jaxbatchbench --inputs DIR --out DIR [--ks 1,2,4,8] [--cache DIR]
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1,2,4,8"); ap.add_argument("--cache", default=None); a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS
    if a.cache:
        J.enable_compilation_cache(a.cache)
    I, O = Path(a.inputs), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); names = list(pts); trel, keep = JM.schedule(inp, off)
    meas = jax.jit(lambda o, c, g, b: JS.measure(o, c, g, si, b)); res = dict(device=jax.devices()[0].device_kind); chi2_1 = {}

    def chain(P):
        out = JM.spray_batch(P, ht, inp, trel, keep); c2 = []
        for xv, chi, age in out:
            bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(chi), 8, 7).items()}
            m, e = meas(JM.observables(xv, inp), chi, jnp.asarray(age), bank); c2.append(float(JS.chi2(m, e, si)))
        return c2
    for K in [int(k) for k in a.ks.split(",")]:
        P = [{k: float(v) for k, v in pts[names[i % len(names)]].items()} for i in range(K)]
        t1 = time.perf_counter(); chain(P); t_first = time.perf_counter()-t1
        t1 = time.perf_counter(); c2 = chain(P); dt = time.perf_counter()-t1
        if K == 1:
            chi2_1[names[0]] = c2[0]
        d = [c-chi2_1[names[i % len(names)]] for i, c in enumerate(c2) if names[i % len(names)] in chi2_1]
        res[f"K{K}"] = dict(seconds=dt, per_model=dt/K, first_call=t_first, chi2=c2, dchi2_vs_single=d)
        print(f"K={K}: {dt:.1f} s for {K} models = {dt/K:.2f} s/model (first call {t_first:.0f} s); chi2 {[round(c, 2) for c in c2]}; "
              f"vs single {[round(x, 2) for x in d]}", flush=True)
        if K == 1:                                                                         # single-model chi2 of every test point
            for nm in names[1:]:
                chi2_1[nm] = chain([{k: float(v) for k, v in pts[nm].items()}])[0]
            print("single-model chi2:", {k: round(v, 2) for k, v in chi2_1.items()}, flush=True)
        (O/"batch_timing.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
