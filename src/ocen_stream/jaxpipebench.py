"""Timing of the full JAX model chain (GPU pipeline, 2026-10-25): release, integration (DOP853 / Yoshida + re-integration), score.

At each test point (points.json) and after a warm-up call (compilation timed separately), synchronised timings of
  release   jaxmodel.release (orbit, r_J, Fardal release)
  spray     jaxmodel.spray with method dop853 (AGAMA's integrator, n_att attempts per compaction) and yoshida (+ Dopri8 close passes)
  score     jaxscore.measure + chi2 with NBOOT = 8 and 64 bootstraps (fixed per-slot banks 7, 8, 9: chi2 spread between banks)
Results appended to OUT/timing.json after every point (restartable); the final observables of the c0_final sprays are saved for a
CPU check (gpu_spray_check-style scoring on the laptop).
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
        for meth, kw in (("dop853", dict(method="dop853", n_att=32)), ("dop853_att128", dict(method="dop853", n_att=128)), ("yoshida", dict(method="yoshida"))):
            out, r[meth+"_first_s"] = timed(JM.spray, p, ht, inp, trel, keep, **kw)
            out, r[meth+"_s"] = timed(JM.spray, p, ht, inp, trel, keep, **kw); xv, chi, age, nx = out; r[meth+"_aux"] = int(nx)
            obs = JM.observables(xv, inp)
            if name == "c0_final" and meth in ("dop853", "yoshida"):
                np.savez(O/f"{name}_{meth}.npz", age_cut_myr=700., chi=np.asarray(chi), age=np.asarray(age), d=np.asarray(obs["dist"]),
                         **{k: np.asarray(obs[k]) for k in ("l", "b", "pmra", "pmdec", "vlos")})
            if meth == "dop853":
                for nb in (8, 64):
                    c2 = []
                    for seed in (7, 8, 9):
                        bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(chi), nb, seed).items()}
                        (m, e), dt = timed(meas, obs, chi, age, bank); c2.append(float(JS.chi2(m, e, si)))
                        r.setdefault(f"score_nb{nb}_s", []).append(dt)
                    r[f"chi2_nb{nb}"] = c2
        res[name] = r; tf.write_text(json.dumps(res, indent=1))
        print(f"{name}: release {r['release_s']:.2f} s; spray dop853 {r['dop853_s']:.1f} s (n_att 128: {r['dop853_att128_s']:.1f} s; "
              f"slowest {r['dop853_aux']} attempts), yoshida+reint {r['yoshida_s']:.1f} s ({r['yoshida_aux']} re-integrated); score nb8 "
              f"{np.median(r['score_nb8_s'][1:]):.2f} s, nb64 {np.median(r['score_nb64_s'][1:]):.2f} s; chi2 nb8 {r['chi2_nb8']}, nb64 {r['chi2_nb64']}", flush=True)


if __name__ == "__main__":
    main()
