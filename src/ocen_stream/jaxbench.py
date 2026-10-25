"""Integrator benchmark for the JAX spray (2026-10-25): speed and accuracy of Yoshida-4 and Dopri8 on CPU or GPU.

For each case (method, step / tolerance, particle number, precision): compile once on the case's shapes (time reported
separately), then a timed run (block_until_ready). Accuracy: |dv| of the final states against AGAMA 1e-10 and, for scale, AGAMA 1e-8
against 1e-10 on the same particles; converted to mas/yr at each particle's rough heliocentric distance. Full-N cases also save the
final states (for scoring with stream_model_compare on a CPU machine). Results are appended to OUT/results.json after every case,
and finished cases are skipped on restart.
Usage: python -m ocen_stream.jaxbench --inputs bench_inputs.npz --out DIR [--quick]
"""
import argparse, json, platform, time
from pathlib import Path
import numpy as np

T_MYR = 977.79222168


def cases(n_full, quick=False):
    """(key, method, kw, N, dtype); the step dt in AGAMA time units."""
    if quick:
        return [("yoshida4_dt0.2_N2000_f64", "yoshida4", dict(dt=0.2/T_MYR), 2000, "f64"),
                ("dopri8_tol1e-8_N2000_f64", "dopri8", dict(rtol=1e-8, atol=1e-8), 2000, "f64")]
    out = []
    for n in (10_000, 50_000, n_full):
        for meth, kw, tag in (("yoshida4", dict(dt=0.2/T_MYR), "dt0.2"), ("yoshida4", dict(dt=0.1/T_MYR), "dt0.1"),
                              ("dopri8", dict(rtol=1e-6, atol=1e-6), "tol1e-6"), ("dopri8", dict(rtol=1e-8, atol=1e-8), "tol1e-8")):
            out.append((f"{meth}_{tag}_N{n}_f64", meth, kw, n, "f64"))
    out += [(f"yoshida4_dt0.2_N{n_full}_f32", "yoshida4", dict(dt=0.2/T_MYR), n_full, "f32"),
            (f"dopri8_tol1e-6_N{n_full}_f32", "dopri8", dict(rtol=1e-6, atol=1e-6), n_full, "f32")]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--only", default=None, help="substring filter on case keys")
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ocen_stream import jaxspray as J
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rf = out/"results.json"
    res = json.loads(rf.read_text()) if rf.exists() else {}
    z = np.load(a.inputs); perm = np.random.default_rng(0).permutation(len(z["ic"]))
    dkpc = np.linalg.norm(z["xv_agama10"][:, :3]-np.array([-8.2, 0., 0.02]), axis=1)
    to_mu = lambda dv, i: dv/(4.74047*dkpc[i])
    dev = jax.devices()[0]; info = dict(device=str(dev), kind=dev.device_kind, platform=dev.platform, jax=jax.__version__, host=platform.node())
    print(info, flush=True)
    tabs = {"f64": J.load(a.inputs, jnp.float64), "f32": J.load(a.inputs, jnp.float32)}
    for key, meth, kw, n, prec in cases(len(z["ic"]), a.quick):
        if key in res or (a.only and a.only not in key):
            continue
        n = min(n, len(z["ic"])); i = np.sort(perm[:n]); dt = jnp.float64 if prec == "f64" else jnp.float32
        ic, t0 = jnp.asarray(z["ic"][i], dt), jnp.asarray(z["t0"][i], dt); tb = tabs[prec]
        kwj = {k: dt(v) for k, v in kw.items()}
        t1 = time.perf_counter(); xv, aux = J.run(meth, ic, t0, tb, **kwj); xv.block_until_ready(); t_first = time.perf_counter()-t1
        t1 = time.perf_counter(); xv, aux = J.run(meth, ic, t0, tb, **kwj); xv.block_until_ready(); t_run = time.perf_counter()-t1
        xv = np.asarray(xv, np.float64); aux = np.asarray(aux)
        dv = np.linalg.norm(xv[:, 3:]-z["xv_agama10"][i, 3:], axis=1); d8 = np.linalg.norm(z["xv_agama8"][i, 3:]-z["xv_agama10"][i, 3:], axis=1)
        pct = (50, 90, 99, 99.9)
        nf = int(aux.max()) if meth == "yoshida4" else int(aux.max())*12                     # force evals for the slowest particle
        r = dict(method=meth, kw={k: float(v) for k, v in kw.items()}, n=int(n), precision=prec, seconds=t_run, seconds_first_call=t_first,
                 compile_seconds=max(t_first-t_run, 0.), us_per_particle=1e6*t_run/n, force_evals_max=nf,
                 steps_median=float(np.median(aux)) if meth == "dopri8" else float(aux.max()/3), nonfinite=int((~np.isfinite(xv)).any(1).sum()),
                 dmu_pct={str(p): float(np.percentile(to_mu(dv, i), p)) for p in pct},
                 dmu_agama8_pct={str(p): float(np.percentile(to_mu(d8, i), p)) for p in pct}, **info)
        try:
            r["peak_bytes"] = int(dev.memory_stats().get("peak_bytes_in_use", -1))
        except Exception:                                                                      # noqa: BLE001 -- CPU has no stats
            r["peak_bytes"] = -1
        if n == len(z["ic"]):
            np.save(out/f"{key}_xv.npy", xv)
        res[key] = r; rf.write_text(json.dumps(res, indent=1))
        print(f"{key}: {t_run:.2f} s ({r['us_per_particle']:.2f} us/particle; first call {t_first:.1f} s); |dmu| 90/99/99.9% "
              f"{r['dmu_pct']['90']:.1e} / {r['dmu_pct']['99']:.1e} / {r['dmu_pct']['99.9']:.1e} mas/yr (AGAMA 1e-8: "
              f"{r['dmu_agama8_pct']['90']:.1e} / {r['dmu_agama8_pct']['99']:.1e} / {r['dmu_agama8_pct']['99.9']:.1e}); nonfinite {r['nonfinite']}", flush=True)


if __name__ == "__main__":
    main()
