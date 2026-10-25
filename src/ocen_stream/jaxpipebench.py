"""Timing of the full JAX model chain (GPU pipeline, 2026-10-25): release, integration (DOP853 / Yoshida + re-integration), score.

At each test point (points.json) and after a warm-up call (compilation timed separately), synchronised timings of
  release   jaxmodel.release (orbit, r_J, Fardal release)
  spray     jaxmodel.spray with method dop853 (AGAMA's integrator), without and with the CPU hand-off of the last particles at
            1600 / 2000 attempts (max |dv| of the hand-off result vs the GPU-only one)
  score     jaxscore.measure + chi2 with NBOOT = 8 (fixed per-slot banks 7, 8, 9)
Results appended to OUT/timing.json after every point (restartable) and printed as JSON at the end.
Usage: python -m ocen_stream.jaxpipebench --inputs DIR (host_tables, spray_inputs, score_inputs .npz, points.json) --out DIR
"""
import argparse, json, platform, time
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--points", default=None, help="comma-separated subset of points.json keys"); a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS
    I, O = Path(a.inputs), Path(a.out); O.mkdir(parents=True, exist_ok=True); tf = O/"timing.json"
    res = json.loads(tf.read_text()) if tf.exists() else {}
    dev = jax.devices()[0]; info = dict(device=str(dev), kind=dev.device_kind, jax=jax.__version__, host=platform.node()); print(info, flush=True)
    ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); trel, keep = JM.schedule(inp, off)
    meas = jax.jit(lambda o, c, g, b: JS.measure(o, c, g, si, b))
    sync = lambda x: jax.tree_util.tree_map(lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, x)

    def timed(f, *args, **kw):
        t1 = time.perf_counter(); r = sync(f(*args, **kw)); return r, time.perf_counter()-t1
    for name, p in pts.items():
        if name in res or (a.points and name not in a.points.split(",")):
            continue
        r = dict(info); p = {k: float(v) for k, v in p.items()}
        _, r["release_first_s"] = timed(JM.release, p, ht, inp, trel, keep); _, r["release_s"] = timed(JM.release, p, ht, inp, trel, keep)
        ref = None
        for meth, ho in (("dop853", None), ("dop853_ho1600", 1600), ("dop853_ho2000", 2000)):
            out, r[meth+"_first_s"] = timed(JM.spray, p, ht, inp, trel, keep, method="dop853", handoff=ho)
            out, r[meth+"_s"] = timed(JM.spray, p, ht, inp, trel, keep, method="dop853", handoff=ho); xv, chi, age, nx = out
            r[meth+"_attempts"] = int(nx); xv = np.asarray(xv)
            if ref is None:
                ref = xv; obs = JM.observables(jnp.asarray(xv), inp); c2 = []
                for seed in (7, 8, 9):
                    bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(chi), 8, seed).items()}
                    (m, e), dt = timed(meas, obs, chi, age, bank); c2.append(float(JS.chi2(m, e, si))); r.setdefault("score_s", []).append(dt)
                r["chi2"] = c2
            else:
                r[meth+"_max_dv_vs_gpu_only"] = float(np.abs(xv[:, 3:]-ref[:, 3:]).max())
        res[name] = r; tf.write_text(json.dumps(res, indent=1))
        print(f"{name}: release {r['release_s']:.2f} s; spray (incl. release) dop853 {r['dop853_s']:.1f} s, hand-off at 1600 "
              f"{r['dop853_ho1600_s']:.1f} s (max |dv| vs GPU-only {r['dop853_ho1600_max_dv_vs_gpu_only']:.1e} km/s), at 2000 {r['dop853_ho2000_s']:.1f} s; "
              f"score {np.median(r['score_s'][1:]):.2f} s; chi2 {', '.join(f'{c:.2f}' for c in r['chi2'])}", flush=True)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
