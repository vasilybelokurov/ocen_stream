#!/usr/bin/env python3
"""Score the GPU benchmark's final states with stream_model_compare, paired against AGAMA (2026-10-25).

Each *_xv.npy (final phase space of the 179200 benchmark particles, order of bench_inputs.npz) is projected to observables with
grid4_common.observables (gravity21 frame, as make_spray) and given the reference chi / age (the AGAMA 1e-8 selection), then scored
with the same rng seed as integrator_spray_test.py, so dchi2 is paired with the AGAMA 1e-8 spray (== stored final_c0_2).
Projection runs in a child process (--project; agama), scoring in the parent (torch): their OpenMP runtimes clash.
Usage: PYTHONPATH=src python bin/streams/gpu_bench_score.py DIR -> results/plot_data/gpu_bench_score.json
"""
import glob, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
INP = ROOT/"results/gpu/bench_inputs.npz"


def project(d):
    from grid4_common import observables, w
    z = np.load(INP); files = sorted(glob.glob(str(Path(d)/"*_xv.npy")))+["agama1e-8"]
    for f in files:
        xv = z["xv_agama8"] if f == "agama1e-8" else np.load(f); o = observables(xv, "gravity21")
        out = Path(d)/(("agama1e-8" if f == "agama1e-8" else Path(f).stem)+"_obs.npz")
        np.savez(out, l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], d=o["dist"], chi=z["chi"], age=z["age"],
                 age_cut_myr=np.float64(700.))


def main():
    d = sys.argv[1]
    if "--project" in sys.argv:
        return project(d)
    subprocess.run([sys.executable, __file__, d, "--project"], check=True)
    sys.argv = [sys.argv[0]]
    import stream_model_compare as MC
    D = MC.data_vector(); O = MC.Obs(); sd = np.r_[D["sky_ce"], D["sky_se"], np.sqrt(np.diag(D["pmra_C"])), np.sqrt(np.diag(D["pmdec_C"]))]
    vec = lambda M: np.r_[M["sky_c"], M["sky_s"], M["pmra"], M["pmdec"]]
    st = np.load(ROOT/"results/streams/fitknot_20261025/runs/final_c0_2.npz"); ref = np.load(Path(d)/"agama1e-8_obs.npz")
    print("AGAMA 1e-8 projection == stored final_c0_2:", all(np.allclose(ref[k], st[k], rtol=0, atol=1e-9) for k in ("l", "b", "pmra", "pmdec")))
    R = MC.measure(O, D, str(Path(d)/"agama1e-8_obs.npz"), np.random.default_rng(12345)); cr = MC.chi2(D, R); res = {"agama1e-8": dict(chi2=cr)}
    for f in sorted(glob.glob(str(Path(d)/"*_xv_obs.npz"))):
        M = MC.measure(O, D, f, np.random.default_rng(12345)); c = MC.chi2(D, M); s = (vec(M)-vec(R))/sd
        name = Path(f).name.replace("_xv_obs.npz", ""); res[name] = dict(chi2=c, dchi2=c["total"]-cr["total"], max_shift_sigma=float(np.nanmax(np.abs(s))))
        print(f"{name}: chi2 {c['total']:.2f} (ref {cr['total']:.2f}); dchi2 {res[name]['dchi2']:+.2f}; max |shift| {res[name]['max_shift_sigma']:.3f} sigma", flush=True)
    (ROOT/"results/plot_data/gpu_bench_score.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
