#!/usr/bin/env python3
"""Full-spray integrator test (2026-10-25, user-approved): fixed-step Yoshida-4 / leapfrog vs AGAMA at the best free-fit point.

Same release times, initial conditions and particles in every variant (grid4_common.make_spray with the final_c0_2 seed and epoch
offset); only the particle integration changes. Fixed-step variants replace agama.orbit for the particle call inside
spray.spray_unwrapped (the centre orbit stays AGAMA 1e-12): per-particle step = age / round(age / dt), AGAMA forces with t,
float64. Sprays run in a child process (--spray; agama only): torch (imported by the score) and agama's OpenMP clash in one process.
The chi > 0 selection is taken from the AGAMA 1e-8 reference for every variant (chi needs trajectories; fixed here so only
the integration differs). Each variant is scored with stream_model_compare (same rng seed -> paired draws).
Usage: OMP_NUM_THREADS=8 PYTHONPATH=src python bin/streams/integrator_spray_test.py
       -> results/streams/integrator_spray_20261025/*.npz, results/plot_data/integrator_spray_test.json, plots/integrator_spray_test.png
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
SPRAY_ONLY = "--spray" in sys.argv; sys.argv = [sys.argv[0]]
import subprocess
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ocen_stream.restricted import AGAMA_T_MYR

RUN = json.loads((ROOT/"results/streams/fitknot_20261025/runs/final_c0_2.json").read_text()); P = RUN["params"]
OUT = ROOT/"results/streams/integrator_spray_20261025"; OUT.mkdir(parents=True, exist_ok=True)
W1 = 1/(2-2**(1/3)); W0 = -2**(1/3)/(2-2**(1/3))
SCHEMES = dict(leapfrog=((0.5, 0.5), (1.,)), yoshida4=((W1/2, (W0+W1)/2, (W0+W1)/2, W1/2), (W1, W0, W1)))
VARIANTS = (("agama_1e-8", None, None), ("agama_1e-6", None, None), ("yoshida4_0.2", "yoshida4", 0.2), ("yoshida4_0.1", "yoshida4", 0.1),
            ("leapfrog_0.05", "leapfrog", 0.05), ("leapfrog_0.02", "leapfrog", 0.02))
KEYS = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")


def fixed_orbit(scheme, dt_myr, real_orbit):
    """agama.orbit stand-in: fixed-step drift-kick composition for the particle call (array timestart); other calls pass through."""
    cd, ck = SCHEMES[scheme]

    def orbit(potential, ic, timestart, time, trajsize, accuracy, **kw):
        if np.ndim(timestart) == 0:
            return real_orbit(potential=potential, ic=ic, timestart=timestart, time=time, trajsize=trajsize, accuracy=accuracy, **kw)
        t0, dur = np.asarray(timestart, float), np.asarray(time, float); n = np.maximum(np.rint(dur/(dt_myr/AGAMA_T_MYR)).astype(int), 1); h0 = dur/n
        x, v = ic[:, :3].copy(), ic[:, 3:].copy()
        for k in range(n.max()):
            h = np.where(k < n, h0, 0.)[:, None]; frac = 0.
            for i in range(len(cd)):
                x += cd[i]*h*v; frac += cd[i]
                if i < len(ck):
                    v += ck[i]*h*potential.force(x, t=t0+(k+frac)*h0)
        res = np.empty((len(x), 2), object)                                          # 2-point "trajectory" (chi is overwritten)
        for j in range(len(x)):
            res[j, 0] = np.array([t0[j], t0[j]+dur[j]]); res[j, 1] = np.vstack((ic[j], np.r_[x[j], v[j]]))
        return res
    return orbit


def spray(name, scheme, dt, acc):
    f = OUT/f"{name}.npz"
    if f.exists():
        return
    from grid4_common import make_spray
    import ocen_stream.spray as SP
    from ocen_stream.frames import frame_r0
    real = SP.agama_kpc().orbit; t1 = time.time()
    if scheme:
        SP.agama_kpc().orbit = fixed_orbit(scheme, dt, real)
    try:
        m = make_spray(P["omega"], P["angle"], P["amp"], P["dist"], nrel=RUN["nrel"], seed=RUN["seed"], pm=(P["pmra"], P["pmdec"]),
                       host_kw=dict(bar_size=P["size"], axi_variant=dict(halo_q=P["halo_q"], vc_target=P["vc"], r0_kpc=frame_r0("gravity21"))),
                       max_age_myr=700., epoch_offset=RUN["epoch_offset"], frame="gravity21", accuracy=acc)
    finally:
        SP.agama_kpc().orbit = real
    m = {k: np.asarray(m[k], np.float64) for k in KEYS}; m["age_cut_myr"] = np.float64(700.); m["seconds"] = np.float64(time.time()-t1)
    np.savez(f, **m); print(f"{name}: spray {m['seconds']:.0f} s", flush=True)


def main():
    if SPRAY_ONLY:
        for name, scheme, dt in VARIANTS:
            spray(name, scheme, dt, 1e-6 if name == "agama_1e-6" else 1e-8)
        return
    subprocess.run([sys.executable, __file__, "--spray"], check=True)
    import stream_model_compare as MC
    D = MC.data_vector(); O = MC.Obs(); res = {}; Ms = {}
    for name, scheme, dt in VARIANTS:
        m = dict(np.load(OUT/f"{name}.npz")); sec = float(m.pop("seconds"))
        if name == "agama_1e-8":
            ref = m; st = np.load(ROOT/"results/streams/fitknot_20261025/runs/final_c0_2.npz")
            same = all(np.array_equal(ref[k], st[k]) for k in ("l", "b", "pmra", "pmdec", "chi", "age"))
            print(f"reference reproduces final_c0_2.npz exactly: {same}", flush=True)
        else:
            m["chi"] = ref["chi"]; m.pop("seconds", None); np.savez(OUT/f"{name}_sel.npz", **m)                 # reference selection
        fs = OUT/("agama_1e-8.npz" if name == "agama_1e-8" else f"{name}_sel.npz")
        M = MC.measure(O, D, str(fs), np.random.default_rng(12345)); c = MC.chi2(D, M); Ms[name] = M
        dmu = np.hypot(m["pmra"]-ref["pmra"], m["pmdec"]-ref["pmdec"])
        res[name] = dict(seconds=sec, chi2=c, dmu_pct={p: float(np.percentile(dmu, p)) for p in (50, 90, 99, 99.9)})
        print(f"{name}: spray {sec:.0f} s; chi2 {c['total']:.2f} (sky_c {c['sky_c']:.2f}, sky_s {c['sky_s']:.2f}, pmra {c['pmra']:.2f}, "
              f"pmdec {c['pmdec']:.2f}); |dPM| vs ref pct 50/90/99/99.9: " + " ".join(f"{v:.1e}" for v in res[name]["dmu_pct"].values()), flush=True)
    # summary shifts in units of sigma_data
    sd = np.r_[D["sky_ce"], D["sky_se"], np.sqrt(np.diag(D["pmra_C"])), np.sqrt(np.diag(D["pmdec_C"]))]
    vec = lambda M: np.r_[M["sky_c"], M["sky_s"], M["pmra"], M["pmdec"]]; R = Ms["agama_1e-8"]
    for name in Ms:
        s = (vec(Ms[name])-vec(R))/sd; res[name]["max_shift_sigma"] = float(np.nanmax(np.abs(s))); res[name]["shift_sigma"] = s.tolist()
        res[name]["dchi2"] = res[name]["chi2"]["total"]-res["agama_1e-8"]["chi2"]["total"]
        print(f"{name}: dchi2 {res[name]['dchi2']:+.2f}; max |summary shift| {res[name]['max_shift_sigma']:.3f} sigma_data", flush=True)
    (ROOT/"results/plot_data/integrator_spray_test.json").write_text(json.dumps(res, indent=1))
    # figure: summary shifts + particle-level PM differences
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True); lab = [f"c{x:.0f}" for x in D["pk"]]+[f"s{x:.0f}" for x in D["pk"]]+\
        [f"ra{x:.0f}" for x in D["pmra_k"]]+[f"de{x:.0f}" for x in D["pmdec_k"]]
    for i, name in enumerate(n for n in Ms if n != "agama_1e-8"):
        ax[0].plot(np.arange(len(sd))+0.1*i, res[name]["shift_sigma"], "o", label=f"{name}: dchi2 {res[name]['dchi2']:+.2f}")
    ax[0].axhspan(-0.1, 0.1, color="0.85"); ax[0].set_xticks(range(len(sd))); ax[0].set_xticklabels(lab, rotation=90, fontsize=7)
    ax[0].set(ylabel="(variant - AGAMA 1e-8) / sigma_data", title="shift of each of the 25 summaries (grey: +-0.1 sigma)"); ax[0].legend(fontsize=8)
    for name, *_ in VARIANTS[1:]:
        m = dict(np.load(OUT/f"{name}.npz")); dmu = np.hypot(m["pmra"]-ref["pmra"], m["pmdec"]-ref["pmdec"]); s = np.sort(dmu)
        ax[1].plot(s, 1-np.arange(s.size)/s.size, label=name)
        k = (ref["chi"] > 0) & (ref["age"] < 700.); ax[2].scatter(MC.project_gc(ref["l"][k], ref["b"][k], O.GC)[0][::5], dmu[k][::5], s=1, label=name)
    ax[1].set(xscale="log", yscale="log", xlabel="|dPM| vs AGAMA 1e-8 [mas/yr] (observed PMs)", ylabel="fraction with larger error",
              title=f"particle-level differences ({ref['l'].size} particles)", ylim=(1/ref["l"].size, 1.05)); ax[1].legend(fontsize=8)
    ax[2].set(yscale="log", xlabel=r"$\phi_1$ [deg]", ylabel="|dPM| [mas/yr]", title="where along the stream (every 5th particle)", ylim=(1e-8, 30), xlim=(-5, 35))
    ax[2].legend(fontsize=8, markerscale=6)
    fig.suptitle("Full-spray integrator test at the best free-fit point (final_c0_2 seed; same particles in every variant)")
    fig.savefig(ROOT/"plots/integrator_spray_test.png", dpi=85); print("plots/integrator_spray_test.png")


if __name__ == "__main__":
    main()
