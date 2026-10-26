"""Validation and timing of the CUDA spray kernel (cuspray) on a GPU (step 2, 2026-10-26).

At each test point (points.json): release (JAX + C orbit), then
  ref        jaxmodel.dop853_chi with the C hand-off at 1024 attempts (the current GPU path; validated vs AGAMA)
  kernel     cuspray.integrate in release order (twice: repeat bit-identical?) and sorted by the attempts of the first run (as a
             chain would sort by the previous model's costs; bit-identical to release order?); fmad off (as the C library) and on
  particles  kernel (fmad off) and ref vs the C library on 2048 random + the 64 slowest particles: |dx|, |dv|, chi-sign agreement
  chi2       jaxscore with NBOOT = 8, bank 7, for ref and the kernel runs
Then all points in one launch (batch of 5, and 10 with each point twice): per-model results bit-identical to the single runs?
Timings are end to end (host -> device copies, launch, device -> host). Results in OUT/cubench.json.
Usage: python -m ocen_stream.cubench --inputs DIR --out DIR
"""
import argparse, json, os, platform, time
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS, cspray, cuspray as CU
    I, O = Path(a.inputs), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    dev = jax.devices()[0]; res = dict(info=dict(kind=dev.device_kind, jax=jax.__version__, host=platform.node(), ncpu=os.cpu_count()))
    print(res["info"], flush=True)
    ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); trel, keep = JM.schedule(inp, off)
    meas = jax.jit(lambda o, c, g, b: JS.measure(o, c, g, si, b)); bank = None; acc = JM.SPRAY_ACC; models, single = [], {}

    def chi2(y, chi, age):
        (m, e) = meas(JM.observables(jnp.asarray(y), inp), jnp.asarray(chi), age, bank); return float(JS.chi2(m, e, si))

    def kern(ms, **kw):
        t1 = time.perf_counter(); r = CU.integrate(ms, acc, **kw); return r, time.perf_counter()-t1
    for name, p in pts.items():
        r = {}; p = {k: float(v) for k, v in p.items()}
        tb, ic, t0, rj = JM.release(p, ht, inp, trel, keep); ic, t0 = np.asarray(ic), np.asarray(t0)
        age = jnp.asarray((inp["T"]-t0)*JM.T_MYR); n = len(t0)
        if bank is None:
            bank = {k: jnp.asarray(v) for k, v in JS.make_bank(n, 8, 7).items()}
        JM.dop853_chi(ic, t0, tb, handoff=1024)                                                   # warm-up
        t1 = time.perf_counter(); yr, cr, _ = JM.dop853_chi(ic, t0, tb, handoff=1024); r["ref_s"] = time.perf_counter()-t1
        r["ref_chi2"] = chi2(yr, cr, age)
        (k0,), r["kernel_first_s"] = kern([(ic, t0, tb)])                                       # includes compilation (first point)
        (k1,), r["kernel_s"] = kern([(ic, t0, tb)])
        r["repeat_bit_identical"] = bool(all(np.array_equal(u, v) for u, v in zip(k0, k1)))
        natt = k1[2]; r["failed"] = int(np.sum(natt < 0)); r["attempts_max"] = int(natt.max()); r["attempts_mean"] = float(natt.mean())
        srt = np.argsort(-natt, kind="stable").astype(np.int32)
        (k2,), r["kernel_sorted_s"] = kern([(ic, t0, tb)], order=srt)
        r["order_bit_identical"] = bool(all(np.array_equal(u, v) for u, v in zip(k1, k2)))
        r["warp_eff_release_order"] = float(CU.warp_efficiency(natt)); r["warp_eff_sorted"] = float(CU.warp_efficiency(natt, srt))
        CU.integrate([(ic, t0, tb)], acc, order=srt, fmad=True)
        (k3,), r["kernel_sorted_fmad_s"] = kern([(ic, t0, tb)], order=srt, fmad=True)
        r["kernel_chi2"] = chi2(k1[0], k1[1]*JM.T_MYR, age); r["kernel_fmad_chi2"] = chi2(k3[0], k3[1]*JM.T_MYR, age)
        rng = np.random.default_rng(1); sub = np.unique(np.r_[rng.choice(n, 2048, replace=False), srt[:64]])
        t1 = time.perf_counter(); yc, cc, nc = cspray.integrate(ic[sub], t0[sub], tb, acc); r["c_subset_s"] = time.perf_counter()-t1
        for lab, (yy, ch) in (("kernel", (k1[0][sub], k1[1][sub])), ("ref", (yr[sub], cr[sub]/JM.T_MYR))):
            dx = np.linalg.norm(yy[:, :3]-yc[:, :3], axis=1); dv = np.linalg.norm(yy[:, 3:]-yc[:, 3:], axis=1)
            r[f"{lab}_vs_c"] = dict(n=len(sub), dx_median_kpc=float(np.median(dx)), dx_p99_kpc=float(np.percentile(dx, 99)),
                                    dv_median_kms=float(np.median(dv)), n_dv_gt_1kms=int(np.sum(dv > 1)), chi_sign_agree=float(np.mean(np.sign(ch) == np.sign(cc))))
        r["kernel_vs_c"]["attempts_equal"] = float(np.mean(k1[2][sub] == nc))
        res[name] = r; single[name] = k1; models.append((name, (ic, t0, tb), srt))
        print(name, json.dumps(r), flush=True); (O/"cubench.json").write_text(json.dumps(res, indent=1))
    for rep in (1, 2):
        ms = [m for _, m, _ in models]*rep; names = [nm for nm, _, _ in models]*rep
        off_ = np.cumsum([0]+[len(m[1]) for m in ms]); natt = np.concatenate([single[nm][2] for nm in names])
        srt = np.argsort(-natt, kind="stable").astype(np.int32)
        CU.integrate(ms, acc, order=srt); out, dt = kern(ms, order=srt)
        same = all(all(np.array_equal(u, v) for u, v in zip(o, single[nm])) for o, nm in zip(out, names))
        res[f"batch{len(ms)}"] = b = dict(s=dt, s_per_model=dt/len(ms), bit_identical_to_single=bool(same),
                                          warp_eff=float(CU.warp_efficiency(natt, srt)), n_particles=int(off_[-1]))
        print(f"batch{len(ms)}", json.dumps(b), flush=True); (O/"cubench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))

if __name__ == "__main__":
    main()
