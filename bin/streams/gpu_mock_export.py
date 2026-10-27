#!/usr/bin/env python3
"""Asimov mock data for a recovery test of the GPU sampler (2026-10-27).

The data vector of score_inputs.npz (sky c, s in the kept windows, pmra and pmdec at the knots) is replaced by the model summaries
at a chosen truth, measured with an independent score bank (seed 999, not the likelihood's bank 7), so the mock is not scored with
its own random draws. Data errors and covariances are unchanged; no noise is added (Asimov). The spray is the C library (same code
as the CUDA kernel). All other inputs are copied unchanged.
Usage: PYTHONPATH=src python bin/streams/gpu_mock_export.py --inputs results/gpu --truth truth.json --out OUTDIR
       (truth.json: the 9 parameters of pilots.NAMES; the box limits in it are only recorded)
"""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ocen_stream import jaxspray as J, jaxmodel as JM, jaxscore as JS, cspray, pilots as PL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--bank", type=int, default=999); a = ap.parse_args()
    I, O = Path(a.inputs), Path(a.out); O.mkdir(parents=True, exist_ok=True); p = {k: float(v) for k, v in json.loads(Path(a.truth).read_text()).items() if k in PL.NAMES}
    ht, inp, si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
    pts = json.loads((I/"points.json").read_text()); off = pts["_epoch_offset"]; trel, keep = JM.schedule(inp, off)
    tb, ic, t0, rj = JM.release(p, ht, inp, trel, keep); y, chi, n = cspray.integrate(np.asarray(ic), np.asarray(t0), tb, JM.SPRAY_ACC)
    assert np.all(n >= 0), "failed particles at the truth"
    bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(keep), 8, a.bank).items()}
    age = jnp.asarray((float(inp["T"])-np.asarray(t0))*JM.T_MYR)
    m, e = JS.measure_capped(JM.observables(jnp.asarray(y), inp), jnp.asarray(chi*JM.T_MYR), age, si, bank); m = np.asarray(m)
    assert np.all(np.isfinite(m)), "missing summaries at the truth"
    z = dict(np.load(I/"score_inputs.npz")); k = z["D_sky_c"].size; na = z["D_pmra"].size
    old = {q: z[q].copy() for q in ("D_sky_c", "D_sky_s", "D_pmra", "D_pmdec")}
    z["D_sky_c"], z["D_sky_s"], z["D_pmra"], z["D_pmdec"] = m[:k], m[k:2*k], m[2*k:2*k+na], m[2*k+na:]
    np.savez(O/"score_inputs.npz", **z)
    for f in ("host_tables.npz", "spray_inputs.npz", "points.json"):
        shutil.copy(I/f, O/f)
    lines = [f"{hashlib.sha256((O/f).read_bytes()).hexdigest()}  {f}" for f in ("host_tables.npz", "spray_inputs.npz", "score_inputs.npz", "points.json")]
    (O/"SHA256SUMS").write_text("\n".join(lines)+"\n")
    shift = {q: np.round((z[q]-old[q]), 3).tolist() for q in old}
    (O/"truth.json").write_text(json.dumps(dict(truth=p, bank=a.bank, data_minus_real_data=shift), indent=1))
    print("mock written to", O, "| truth", p, "| mock - real data:", shift)


if __name__ == "__main__":
    main()
