#!/usr/bin/env python3
"""Stage-3 acceptance: JAX score (ocen_stream.jaxscore) vs the NumPy score (stream_model_compare.measure with the same bank).

For stored sprays spanning chi2 ~165 to ~1000+ (the rescored archive), both measurements use identical fixed per-slot draws, so
they differ only by the fitter (Newton vs Nelder-Mead) and floating-point order. Criteria (plan): |dchi2| < 0.5 and every summary
within 0.05 sigma_data. Also reports the JAX timing per model (first call includes compilation).
Usage: PYTHONPATH=src python bin/streams/gpu_score_check.py [N_MODELS] -> results/plot_data/gpu_score_check.json
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
NM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
sys.argv = [sys.argv[0]]
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import stream_model_compare as MC
from ocen_stream import jaxscore as JS


def pick(n):
    """n stored sprays spread in chi2 (rescore_vec), always including the best."""
    rows = sorted((r for r in csv.DictReader(open(ROOT/"results/plot_data/stream_model_rescore_vec.csv")) if r.get("chi2_total")),
                  key=lambda r: float(r["chi2_total"]))
    q = np.unique(np.round(np.geomspace(1, len(rows), n)).astype(int)-1)
    return [ROOT/rows[i]["file"] for i in q]


def main():
    D = MC.data_vector(); O = MC.Obs(); si = JS.load(ROOT/"results/gpu/score_inputs.npz")
    sd = np.r_[D["sky_ce"], D["sky_se"], np.sqrt(np.diag(D["pmra_C"])), np.sqrt(np.diag(D["pmdec_C"]))]; res = {}
    meas = jax.jit(lambda o, c, a, b: JS.measure(o, c, a, si, b))
    for f in pick(NM):
        z = np.load(f); n = len(z["l"]); bank = JS.make_bank(n, si["nboot"], 7)
        M = MC.measure(O, D, str(f), np.random.default_rng(0), bank=bank); c_np = MC.chi2(D, M)["total"]
        v_np = np.r_[M["sky_c"], M["sky_s"], M["pmra"], M["pmdec"]]; e_np = np.r_[M["sky_ce"], M["sky_se"], M["pmra_e"], M["pmdec_e"]]
        obs = {k: jnp.asarray(z[k]) for k in ("l", "b", "pmra", "pmdec")}; bj = {k: jnp.asarray(v) for k, v in bank.items()}
        t1 = time.time(); m, e = meas(obs, jnp.asarray(z["chi"]), jnp.asarray(z["age"]), bj); m.block_until_ready(); dt = time.time()-t1
        m, e = np.asarray(m), np.asarray(e); c_j = float(JS.chi2(jnp.asarray(m), jnp.asarray(e), si))
        sh = np.abs(m-v_np)/sd; she = np.abs(e-e_np)/sd
        res[f.name] = dict(chi2_np=c_np, chi2_jax=c_j, dchi2=c_j-c_np, max_shift_sigma=float(np.nanmax(sh)), max_err_shift_sigma=float(np.nanmax(she)),
                           nan_match=bool(np.array_equal(np.isnan(m), np.isnan(v_np))), seconds=dt)
        r = res[f.name]; print(f"{f.parent.parent.name}/{f.name}: chi2 numpy {c_np:.3f} jax {c_j:.3f} (d {r['dchi2']:+.3f}); max |dsummary| "
                              f"{r['max_shift_sigma']:.4f} sigma, |d err| {r['max_err_shift_sigma']:.4f} sigma; NaNs match {r['nan_match']}; jax {dt:.1f} s", flush=True)
    (ROOT/"results/plot_data/gpu_score_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
