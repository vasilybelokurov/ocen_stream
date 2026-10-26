"""Batched model evaluation on a GPU (2026-10-26): B parameter points -> B chi2.

  release   CPU part (host tables, C centre orbit) of all B models in threads; GPU part (r_J, Fardal offsets) model by model with
            jaxmodel's single-model function (bit-identical to jaxmodel.release)
  integrate one CUDA launch for all B x 179,200 particles (cuspray; per-model results independent of the batch), launch order
            sorted by the attempts of the previous call when given
  score     jaxscore.measure_capped + chi2 model by model
A model's chi2 must not depend on its slot or on the other models in the batch: main() checks it (B = 1 vs 10, slot permutation).
Usage (timing + checks on a GPU): python -m ocen_stream.gpupipe --inputs DIR --out DIR
"""
import argparse, json, os, platform, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from . import jaxspray as J, jaxmodel as JM, jaxscore as JS, cspray, cuspray

jax.config.update("jax_enable_x64", True)

def release_many(params_list, ht, inp, trel, keep, nthreads=None):
    """per-model tables (dicts for cuspray), kept initial conditions (list of (N, 6)) and times t0 (N,) (the same for every model).
    The CPU part (host tables, C orbits) runs in threads; the GPU part model by model with jaxmodel's single-model function, so each
    model's initial conditions are bit-identical to jaxmodel.release (a vmapped release differed at 1e-11, enough to move chaotic
    core passages and chi2 by several units)."""
    with ThreadPoolExecutor(nthreads or os.cpu_count()) as ex:                            # ctypes releases the GIL
        prep = list(ex.map(lambda p: JM.release_prepare(p, ht, inp, trel), params_list))
    out = [JM.release_finish(tb, orbit_r, trel, inp, keep) for tb, orbit_r in prep]
    return [o[0] for o in out], [np.asarray(o[1]) for o in out], np.asarray(out[0][2])


def make_score(inp, si, bank):
    """chi2 of B models from final phase space (B, N, 6) and chi [AGAMA time x kpc] (B, N) at ages age (N,) [Myr], model by model
    (a vmapped score was no faster and differed from the single score by 1e-7); PM knot fits over gathered particles
    (jaxscore.measure_capped, ~10x faster than over all particles)."""
    obs = jax.jit(lambda y: JM.observables(y, inp)); c2 = jax.jit(lambda m, e: JS.chi2(m, e, si))
    return lambda y, chi, age: jnp.stack([c2(*JS.measure_capped(obs(y[k]), chi[k]*JM.T_MYR, age, si, bank)) for k in range(y.shape[0])])


def evaluate(params_list, ht, inp, trel, keep, score, order=None, fmad=False, times=None):
    """chi2 (B,) and attempts (B, N) of B models. order (optional): launch order of the B*N particles (e.g. from the attempts of
    the previous call). times (optional dict): filled with the seconds per stage."""
    t1 = time.perf_counter(); tbs, ic, t0 = release_many(params_list, ht, inp, trel, keep)
    t2 = time.perf_counter(); out = cuspray.integrate([(ic[k], t0, tbs[k]) for k in range(len(tbs))], JM.SPRAY_ACC, order=order, fmad=fmad)
    t3 = time.perf_counter(); age = jnp.asarray((float(inp["T"])-t0)*JM.T_MYR)
    c2 = np.asarray(score(jnp.asarray(np.stack([o[0] for o in out])), jnp.asarray(np.stack([o[1] for o in out])), age))
    t4 = time.perf_counter()
    if times is not None:
        times.update(release=t2-t1, integrate=t3-t2, score=t4-t3, total=t4-t1)
    natt = np.stack([o[2] for o in out])
    if np.any(natt < 0):
        raise RuntimeError(f"gpupipe.evaluate: {np.sum(natt < 0)} particles failed")
    return c2, natt


def sort_order(natt):
    """launch order: particles sorted by decreasing attempts (of the previous call), all models together."""
    return np.argsort(-np.asarray(natt).ravel(), kind="stable").astype(np.int32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="1,5,10,20"); a = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    I, O = Path(a.inputs), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    dev = jax.devices()[0]; res = dict(info=dict(kind=dev.device_kind, jax=jax.__version__, host=platform.node(), ncpu=os.cpu_count()))
    print(res["info"], flush=True)
    ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); trel, keep = JM.schedule(inp, off)
    base = [{k: float(v) for k, v in p.items()} for p in pts.values()]
    P = [dict(p, angle=p["angle"]+d) for d in (0., 1., -1., 2.) for p in base]             # 20 distinct models
    bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(keep), 8, 7).items()}
    S = dict(loop=make_score(inp, si, bank))

    def save():
        (O/"gpupipe.json").write_text(json.dumps(res, indent=1))
    _, ic_b, _ = release_many(P[:3], ht, inp, trel, keep)
    res["release_bit_identical_to_single"] = bool(all(np.array_equal(np.asarray(JM.release(P[k], ht, inp, trel, keep)[1]), ic_b[k]) for k in range(3)))
    print("release batch vs single bit-identical:", res["release_bit_identical_to_single"], flush=True)
    ref = []                                                                               # each of the first 10 models alone (B = 1)
    for k in range(10):
        t = {}; c2, natt = evaluate(P[k:k+1], ht, inp, trel, keep, S["loop"], times=t); ref.append((c2[0], natt[0]))
        if k == 0:
            res["B1_first"] = t
    r1 = np.array([r[0] for r in ref]); nat = np.stack([r[1] for r in ref]); perm = np.random.default_rng(3).permutation(10)
    print("B=1 chi2", r1.tolist(), flush=True)
    for mode, sc in S.items():                                                             # B = 10: repeat, slot permutation, vs B = 1
        c_a, _ = evaluate(P[:10], ht, inp, trel, keep, sc, order=sort_order(nat))
        c_b, _ = evaluate(P[:10], ht, inp, trel, keep, sc, order=sort_order(nat))
        c_p, _ = evaluate([P[i] for i in perm], ht, inp, trel, keep, sc, order=sort_order(nat[perm]))
        res[f"B10_checks_{mode}"] = dict(repeat_bit_identical=bool(np.array_equal(c_a, c_b)), permutation_bit_identical=bool(np.array_equal(c_a[perm], c_p)),
                                         same_as_B1_bitwise=bool(np.array_equal(c_a, r1)), max_abs_diff_vs_B1=float(np.max(np.abs(c_a-r1))))
        print(f"B=10 checks ({mode} score)", res[f"B10_checks_{mode}"], flush=True); save()
    for B in map(int, a.sizes.split(",")):                                                 # timing (second call, sorted launch)
        for mode, sc in S.items():
            try:
                ps = P[:B]; _, natt = evaluate(ps, ht, inp, trel, keep, sc); t = {}
                evaluate(ps, ht, inp, trel, keep, sc, order=sort_order(natt), times=t)
                res[f"timing_B{B}_{mode}"] = dict(t, per_model=t["total"]/B)
            except Exception as e:                                                         # e.g. out of memory
                res[f"timing_B{B}_{mode}"] = dict(error=repr(e)[:300])
            print(f"B={B} {mode} score:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res[f"timing_B{B}_{mode}"].items()}, flush=True); save()
    print(json.dumps(res))

if __name__ == "__main__":
    main()
