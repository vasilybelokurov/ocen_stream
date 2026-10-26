"""Timing of the full model chain on a GPU (v3, 2026-10-26: C centre orbit in the release, C hand-off of the last particles).

At each test point (points.json), after a warm-up call (compilation timed separately), synchronised timings of
  release   jaxmodel.release (C centre orbit, r_J, Fardal release)
  integrate jaxmodel.dop853_chi (AGAMA's DOP853 at SPRAY_ACC) on the GPU only, and with the last particles finished in C on the
            CPU cores (cspray.resume) from HANDOFF attempts on; each run twice (determinism: chi2 and positions bit-identical?)
  score     jaxscore.measure + chi2 with NBOOT = 8 (bank 7) for every run
  c_only    (first point only) the whole integration in C on the CPU cores (cspray.integrate), for reference
Results appended to OUT/timing.json after every point (restartable) and printed as JSON at the end.
Usage: python -m ocen_stream.jaxpipebench --inputs DIR (host_tables, spray_inputs, score_inputs .npz, points.json) --out DIR
"""
import argparse, json, os, platform, time
from pathlib import Path
import numpy as np

HANDOFF = (None, 256, 1024, 2048)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--points", default=None, help="comma-separated subset of points.json keys")
    ap.add_argument("--nmax", type=int, default=None, help="first nmax particles only (smoke test)"); a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS, cspray
    I, O = Path(a.inputs), Path(a.out); O.mkdir(parents=True, exist_ok=True); tf = O/"timing.json"
    res = json.loads(tf.read_text()) if tf.exists() else {}
    dev = jax.devices()[0]; info = dict(device=str(dev), kind=dev.device_kind, jax=jax.__version__, host=platform.node(), ncpu=os.cpu_count())
    print(info, flush=True)
    ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); trel, keep = JM.schedule(inp, off)
    meas = jax.jit(lambda o, c, g, b: JS.measure(o, c, g, si, b)); bank = None
    sync = lambda x: jax.tree_util.tree_map(lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, x)

    def timed(f, *args, **kw):
        t1 = time.perf_counter(); r = sync(f(*args, **kw)); return r, time.perf_counter()-t1
    for name, p in pts.items():
        if name in res or (a.points and name not in a.points.split(",")):
            continue
        r = dict(info); p = {k: float(v) for k, v in p.items()}
        _, r["release_first_s"] = timed(JM.release, p, ht, inp, trel, keep)
        (tb, ic, t0, rj), r["release_s"] = timed(JM.release, p, ht, inp, trel, keep); ic, t0 = ic[:a.nmax], t0[:a.nmax]
        age = jnp.asarray((inp["T"]-np.asarray(t0))*JM.T_MYR); ref = None
        if bank is None:
            bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(t0), 8, 7).items()}
        for ho in HANDOFF:
            key = "gpu_only" if ho is None else f"ho{ho}"; runs = []
            for rep in range(3):                                                         # rep 0 = warm-up (compilation)
                fin = np.zeros(len(t0), np.int64); t1 = time.perf_counter()
                y, chi, natt = JM.dop853_chi(ic, t0, tb, handoff=ho, finish=fin); dt = time.perf_counter()-t1
                obs = JM.observables(jnp.asarray(y), inp); (m, e), ds = timed(meas, obs, jnp.asarray(chi), age, bank)
                runs.append(dict(s=dt, score_s=ds, chi2=float(JS.chi2(m, e, si)), attempts=int(natt), y=y,
                                 n_cpu=int(np.sum(fin > ho)) if ho is not None else 0))
            r[key] = dict(first_s=runs[0]["s"], s=[q["s"] for q in runs[1:]], score_s=[q["score_s"] for q in runs[1:]],
                          chi2=[q["chi2"] for q in runs[1:]], attempts=runs[1]["attempts"], n_on_cpu=runs[1]["n_cpu"],
                          repeat_bit_identical=bool(np.array_equal(runs[1]["y"], runs[2]["y"]) and runs[1]["chi2"] == runs[2]["chi2"]))
            if ref is None:
                ref = runs[1]["y"]
            else:
                dv = np.linalg.norm(runs[1]["y"][:, 3:]-ref[:, 3:], axis=1)
                r[key].update(max_dv_vs_gpu_only=float(dv.max()), n_dv_gt_0p01kms=int(np.sum(dv > 0.01)), n_dv_gt_1kms=int(np.sum(dv > 1.)))
            q = r[key]; print(f"{name} {key}: integrate {np.round(q['s'], 2)} s (first {q['first_s']:.1f}), {q['n_on_cpu']} on CPU, "
                              f"score {np.round(q['score_s'], 2)} s, chi2 {q['chi2']}, repeat bit-identical {q['repeat_bit_identical']}", flush=True)
        if not res:
            _, r["c_only_s"] = timed(cspray.integrate, np.asarray(ic), np.asarray(t0), tb, JM.SPRAY_ACC)
            print(f"{name}: whole integration in C on {os.cpu_count()} CPUs {r['c_only_s']:.1f} s", flush=True)
        print(f"{name}: release {r['release_s']:.2f} s (first {r['release_first_s']:.1f})", flush=True)
        res[name] = r; tf.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))

if __name__ == "__main__":
    main()
