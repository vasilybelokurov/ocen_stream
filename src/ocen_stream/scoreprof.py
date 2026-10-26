"""Where the score's time goes (2026-10-26): jaxscore pieces timed separately on one model (the best-fit point).

  particles   jaxscore.particles (projection, error assignment, PM masks)
  sky         jaxscore.sky for the 9 weightings (data + 8 bootstraps): histograms + Asimov fits
  pm          jaxscore.pm for the 9 weightings: truncated-Gaussian fits at the 11 knots
  measure     the whole measurement over all particles; measure_capped (PM knot fits over gathered particles); chi2 of both
plus the number of particles that enter each PM knot fit and each sky window (out of N). Warm timings (median of 3).
Usage: python -m ocen_stream.scoreprof --inputs DIR  [spray by the CUDA kernel if CuPy is present, else by the C library]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); a = ap.parse_args()
    from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS, cspray
    I = Path(a.inputs); ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); trel, keep = JM.schedule(inp, off)
    p = {k: float(v) for k, v in pts["c0_final"].items()}; tb, ic, t0, rj = JM.release(p, ht, inp, trel, keep)
    try:
        from ocen_stream import cuspray
        (y, chi, _), = cuspray.integrate([(np.asarray(ic), np.asarray(t0), tb)], JM.SPRAY_ACC); how = "cuda"
    except ImportError:
        y, chi, _ = cspray.integrate(np.asarray(ic), np.asarray(t0), tb, JM.SPRAY_ACC); how = "c"
    obs = JM.observables(jnp.asarray(y), inp); chi = jnp.asarray(chi*JM.T_MYR); age = jnp.asarray((float(inp["T"])-np.asarray(t0))*JM.T_MYR)
    bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(keep), 8, 7).items()}
    sync = lambda x: jax.tree_util.tree_map(lambda v: v.block_until_ready(), x)

    def timed(f, *args):
        sync(f(*args)); ts = []
        for _ in range(3):
            t1 = time.perf_counter(); sync(f(*args)); ts.append(time.perf_counter()-t1)
        return float(np.median(ts))
    f_part = jax.jit(lambda o, c, g: JS.particles(o, c, g, si, bank)); P = f_part(obs, chi, age)
    Wt = jnp.concatenate((P["sel"].astype(float)[None], P["sel"].astype(float)[None]*bank["boot"]), 0)
    f_sky = jax.jit(lambda P, W: jax.vmap(lambda w: JS.sky(P, w, si))(W)); f_pm = jax.jit(lambda P, W: jax.vmap(lambda w: JS.pm(P, w, si))(W))
    f_meas = jax.jit(lambda o, c, g: JS.measure(o, c, g, si, bank))
    r = dict(spray=how, device=str(jax.devices()[0]), n=int(len(keep)), n_sel=int(P["sel"].sum()),
             particles_s=timed(f_part, obs, chi, age), sky9_s=timed(f_sky, P, Wt), pm9_s=timed(f_pm, P, Wt), sky1_s=timed(f_sky, P, Wt[:1]),
             pm1_s=timed(f_pm, P, Wt[:1]), measure_s=timed(f_meas, obs, chi, age),
             measure_capped_s=timed(lambda *x: JS.measure_capped(*x, si, bank), obs, chi, age))
    m, e = f_meas(obs, chi, age); r["chi2"] = float(JS.chi2(m, e, si))
    m, e = JS.measure_capped(obs, chi, age, si, bank); r["chi2_capped"] = float(JS.chi2(m, e, si))
    w = np.asarray(P["sel"]); u = np.asarray(P["u"])
    for q, oth in (("pmra", "pmdec"), ("pmdec", "pmra")):
        vq = np.asarray(P[q])-np.interp(u, np.asarray(si["G"]), np.asarray(si[q+"_c"]))
        base = w & np.asarray(P["inwin"]) & np.asarray(P["m_"+oth]) & (np.abs(vq) < float(si["D"][q+"_W"]))
        r[f"n_in_{q}_knots"] = [int(np.sum(base & (np.abs(u-c0) < 1.5))) for c0 in np.asarray(si["D"][q+"_k"])]
    S = si["sky"]; r["n_in_sky_windows"] = [int(np.sum(w & (u >= float(S["eu0"][k])) & (u < float(S["eu0"][k])+float(S["du"][k])*S["ug"].shape[1])))
                                            for k in range(S["ug"].shape[0])]
    r["sky_grid"] = list(S["ug"].shape)+[S["vg"].shape[1]]
    print(json.dumps(r, indent=1))

if __name__ == "__main__":
    main()
