#!/usr/bin/env python3
"""Pericentric (bursty) particle release for the step-2 best 12.5- and 25-deg bar models (2026-10-21).

Question: the angle preference and the Gaia branch prediction both depend on how much debris was released ~250-500 Myr ago,
which the fits so far fixed by assuming uniform release over the last Gyr. Here the same best-fit parameters
(results/streams/profile_step2_20261020/runs/final_a12.5_3, final_a25_1; gravity21 frame, v_c normalised at its R0) are
sprayed with release='peri' (grid4_common.peri_release_times: equal-weight Gaussians of width sigma around each pericentre
of the centre orbit in the last 1 Gyr), sigma = 10 Myr (bursty) and 30 Myr (moderate), 2 seeds, 256k epochs, age < 700 Myr.
Each spray is scored like the profile finals (data box, jackknife, cutoff 7, chi 0-200) plus the step-2 priors (distance, PMs,
v_c). Parameters are NOT re-optimised.
Output: results/streams/peri_release_20261021/runs/<id>.{npz,json}, results/plot_data/peri_release_test.json
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/peri_release_test.py --workers 8
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/peri_release_20261021"
BEST = {"12.5": "results/streams/profile_step2_20261020/runs/final_a12.5_3.json", "25": "results/streams/profile_step2_20261020/runs/final_a25_1.json"}


def run(args):
    lab, sigma, seed = args
    rid = f"a{lab}_s{sigma:g}_seed{seed}"; f = OUT/"runs"/f"{rid}.npz"
    if (OUT/"runs"/f"{rid}.json").exists():
        return json.loads((OUT/"runs"/f"{rid}.json").read_text())
    from grid4_common import make_spray, win, project_gc, GC
    from ocen_stream.frames import frame_r0
    from ocen_stream.score import score_conditional
    import rescore_batch as RB
    from profile_nm import ln_prior
    p = json.loads((ROOT/BEST[lab]).read_text())["params"]; t0 = time.time()
    m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=256000, seed=seed, pm=(p["pmra"], p["pmdec"]),
                   host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21"))),
                   max_age_myr=700., release="peri", peri_sigma_myr=sigma, frame="gravity21")
    keys = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")
    ms = {k: np.asarray(m[k], np.float64) for k in keys}; ms["age_cut_myr"] = np.float64(700.)
    np.savez_compressed(OUT/"runs"/f"{rid}_tmp.npz", **ms); (OUT/"runs"/f"{rid}_tmp.npz").rename(f)
    c, _, ob = RB.catalogue("real"); BG, VB = RB.bounds("data"); mu, mx = project_gc(ms["l"], ms["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, ms["pmra"], ms["pmdec"])), ms["vlos"],
                          ms["chi"], ms["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    rec = dict(run_id=rid, angle_label=lab, peri_sigma_myr=sigma, seed=seed, params=p, total_jk=float(s["total_jk"]),
               ln_prior=float(ln_prior(p, True)), seconds=time.time()-t0)
    (OUT/"runs"/f"{rid}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    (OUT/"runs").mkdir(parents=True, exist_ok=True)
    jobs = [(lab, sg, seed) for lab in BEST for sg in (10., 30.) for seed in (2200001, 2200002)]
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, jobs))
    uni = {}
    for lab in BEST:
        fs = sorted((ROOT/"results/streams/profile_step2_20261020/runs").glob(f"final_a{lab}_*.json"))
        v = [json.loads(f.read_text()) for f in fs]; uni[lab] = [r["total_jk"] for r in v]
    out = dict(runs=recs, uniform_finals=uni, summary={})
    print("ln L (jk, 256k) + prior; uniform = the 4 step-2 finals at the same parameters")
    for lab in BEST:
        pr = [r["ln_prior"] for r in recs if r["angle_label"] == lab][0]
        out["summary"][f"{lab} uniform"] = (float(np.mean(uni[lab])+pr), float(np.std(uni[lab], ddof=1)/2))
        print(f"  {lab:4s} deg uniform     : {np.mean(uni[lab])+pr:9.0f} +- {np.std(uni[lab], ddof=1)/2:.0f}")
        for sg in (10., 30.):
            t = [r["total_jk"]+r["ln_prior"] for r in recs if r["angle_label"] == lab and r["peri_sigma_myr"] == sg]
            out["summary"][f"{lab} peri {sg:g}"] = (float(np.mean(t)), float(abs(t[0]-t[1])/2))
            print(f"  {lab:4s} deg peri s={sg:2.0f} : {np.mean(t):9.0f} +- {abs(t[0]-t[1])/2:.0f}   (seeds {t[0]:.0f}, {t[1]:.0f})")
    for key in ("uniform", "peri 10", "peri 30"):
        print(f"  12.5 - 25 ({key}): {out['summary'][f'12.5 {key}'][0]-out['summary'][f'25 {key}'][0]:+.0f}")
    (ROOT/"results/plot_data/peri_release_test.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
