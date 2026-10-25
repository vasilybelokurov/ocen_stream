#!/usr/bin/env python3
"""Stage-2 acceptance: whole JAX spray vs the AGAMA spray with the same random draws, paired chi2 (2026-10-25).

Points: final_c0_2 (best free fit) and the fit_knot_nm.py start points (different angle, Omega_b, q, A, S, distance). At each point
an AGAMA spray (grid4_common.make_spray, seed and epoch offset of the bank in results/gpu/spray_inputs.npz, nrel 256000, gravity21)
and a JAX spray (jaxmodel.spray, Yoshida-4 dt 0.2 Myr, tabulated host) are made; both scored with stream_model_compare (same rng
seed, AGAMA chi > 0 selection for both and, separately, each with its own chi). Reported: dchi2, max |summary shift| / sigma_data,
|dPM| percentiles, chi sign agreement. Sprays in a child process (--spray; agama + jax), scoring in the parent (torch).
Usage: PYTHONPATH=src python bin/streams/gpu_spray_check.py -> results/gpu/spray_check/*.npz, results/plot_data/gpu_spray_check.json
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/gpu/spray_check"; G = ROOT/"results/gpu"
POINTS = dict(c0_final="results/streams/fitknot_20261025/runs/final_c0_2.json", start_a12p5="results/streams/profile_real_20261019/runs/final_a12.5_0.json",
              start_a17p5="results/streams/profile_step2_20261020/runs/a17.5_1_e023.json", start_a25="results/streams/profile_real_20261019/runs/a25_0_e001.json",
              start_obsbar="results/streams/obsbar_20261022/runs/peak1_a32_om45_e015.json")
OFFSET = json.loads((ROOT/POINTS["c0_final"]).read_text())["epoch_offset"]                 # the bank's epoch offset (final_c0_2)
KEYS = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")


def sprays():
    import jax; jax.config.update("jax_enable_x64", True)
    from grid4_common import make_spray
    from ocen_stream import jaxspray as J, jaxmodel as JM
    from ocen_stream.frames import frame_r0
    OUT.mkdir(parents=True, exist_ok=True); inp = JM.load_inputs(G/"spray_inputs.npz"); ht = J.load_host(G/"host_tables.npz")
    trel, keep = JM.schedule(inp, OFFSET); seed, nrel = int(np.load(G/"spray_inputs.npz")["seed"]), inp["nrel"]
    for name, f in POINTS.items():
        p = json.loads((ROOT/f).read_text())["params"]
        if not (OUT/f"{name}_agama.npz").exists():
            t1 = time.time()
            m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=nrel, seed=seed, pm=(p["pmra"], p["pmdec"]), max_age_myr=700.,
                           host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21"))),
                           epoch_offset=OFFSET, frame="gravity21")
            np.savez(OUT/f"{name}_agama.npz", age_cut_myr=700., **{k: np.asarray(m[k], float) for k in KEYS}); print(f"{name}: AGAMA {time.time()-t1:.0f} s", flush=True)
        if not (OUT/f"{name}_jax.npz").exists():
            t1 = time.time(); xv, chi, age = JM.spray(p, ht, inp, trel, keep); o = JM.observables(xv, inp)
            np.savez(OUT/f"{name}_jax.npz", age_cut_myr=700., chi=np.asarray(chi), age=np.asarray(age), d=np.asarray(o["dist"]),
                     **{k: np.asarray(o[k]) for k in ("l", "b", "pmra", "pmdec", "vlos")}); print(f"{name}: JAX {time.time()-t1:.0f} s", flush=True)


def main():
    if "--spray" in sys.argv:
        return sprays()
    subprocess.run([sys.executable, __file__, "--spray"], check=True)
    sys.argv = [sys.argv[0]]
    import stream_model_compare as MC
    D = MC.data_vector(); O = MC.Obs(); sd = np.r_[D["sky_ce"], D["sky_se"], np.sqrt(np.diag(D["pmra_C"])), np.sqrt(np.diag(D["pmdec_C"]))]
    vec = lambda M: np.r_[M["sky_c"], M["sky_s"], M["pmra"], M["pmdec"]]; res = {}
    for name in POINTS:
        a, j = dict(np.load(OUT/f"{name}_agama.npz")), dict(np.load(OUT/f"{name}_jax.npz"))
        js = dict(j, chi=a["chi"]); np.savez(OUT/f"{name}_jax_aselect.npz", **js)
        sc = lambda f: (lambda M: (M, MC.chi2(D, M)))(MC.measure(O, D, str(OUT/f), np.random.default_rng(12345)))
        (Ma, ca), (Mj, cj), (Ms, cs) = sc(f"{name}_agama.npz"), sc(f"{name}_jax.npz"), sc(f"{name}_jax_aselect.npz")
        dmu = np.hypot(j["pmra"]-a["pmra"], j["pmdec"]-a["pmdec"])
        res[name] = dict(chi2_agama=ca["total"], dchi2_own_chi=cj["total"]-ca["total"], dchi2_agama_selection=cs["total"]-ca["total"],
                         max_shift_sigma=float(np.nanmax(np.abs((vec(Ms)-vec(Ma))/sd))), chi_sign_agree=float(np.mean(np.sign(j["chi"]) == np.sign(a["chi"]))),
                         dmu_pct={p: float(np.percentile(dmu, p)) for p in (50, 90, 99, 99.9)})
        r = res[name]; print(f"{name}: chi2 AGAMA {r['chi2_agama']:.2f}; JAX dchi2 {r['dchi2_agama_selection']:+.2f} (same selection), "
                             f"{r['dchi2_own_chi']:+.2f} (own chi); max |shift| {r['max_shift_sigma']:.3f} sigma; chi sign agree {r['chi_sign_agree']:.5f}; "
                             f"|dPM| 90/99/99.9% {r['dmu_pct'][90]:.1e} {r['dmu_pct'][99]:.1e} {r['dmu_pct'][99.9]:.1e}", flush=True)
    (ROOT/"results/plot_data/gpu_spray_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
