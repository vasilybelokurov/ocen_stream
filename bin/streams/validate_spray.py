#!/usr/bin/env python3
"""Stage 1: particle spray (src/ocen_dm/streams/spray.py) vs the prescribed-potential tracer runs and vs the data, through
the same track estimator (src/ocen_dm/streams/track.py). Model A mass profile; hosts as in the runs (Hunter+24 axi, bars).
Spray: trailing-arm particles only; tracer runs: unbound tracers. Northern-arm selection as for the data (b > 15, -75<l<-20).
Usage: python bin/streams/validate_spray.py --nrel 4000 hunter_axi hunter_bar33 hunter_bar37.5 hunter_bar41
       -> plots/streams_spray_validation.png, results/plot_data/streams_spray_validation.json
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ocen_dm.streams.analysis import load, run_core, run_particles
from ocen_dm.streams.restricted import AGAMA_T_MYR, bound_set, host_potential
from ocen_dm.streams.frames import observables
from ocen_dm.streams.spray import spray
from ocen_dm.streams.track import measure

w = lambda x: np.where(x > 180, x-360, x)


def track_of(xv, frame="baumgardt"):
    o = observables(xv, frame); l = w(o["l"]); k = (o["b"] > 15) & (l < -20) & (l > -75)
    return measure(l[k], o["b"][k], o["pmra"][k], o["pmdec"][k], nboot=40), int(k.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--nrel", type=int, default=4000)
    a = ap.parse_args()
    data = json.loads((ROOT/"results/plot_data/stream54_track.json").read_text())["all"]
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text())
    r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    out = {}
    fig, ax = plt.subplots(len(a.runs), 3, figsize=(17, 4.2*len(a.runs)), squeeze=False)
    for i, run in enumerate(a.runs):
        R = ROOT/"results/streams"/run/"A_nodm"; rj = json.loads((R/"run.json").read_text())
        T = rj["tback_myr"]/AGAMA_T_MYR; today = np.array(rj["ocen_today"])
        host = host_potential(rj["mw"], bar_omega=rj.get("bar_omega"), bar_angle_deg=rj.get("bar_angle_deg") or 28., t_today=T)
        m, s = run_particles(R); _, xv = load(R, name="snap_today.npz")
        bnd, _ = bound_set(xv, today, m, start=np.linalg.norm(xv[:, :3]-today[:3], axis=1) < 0.5, core=run_core(R))
        tt, nt = track_of(xv[~bnd])
        t0 = time.time(); sp = spray(host, today, T, r_kpc, M, n_release=a.nrel); dt = time.time()-t0
        ts, ns = track_of(sp["xv"][sp["arm"] == 1])
        print(f"{run}: spray {2*a.nrel} particles in {dt:.1f} s (centre off {sp['centre_offset_pc']:.2f} pc); northern-arm N: tracer {nt}, spray {ns}")
        print("   b     dpmdec(spray-tracer)  dpmra   dl     | n_tracer n_spray | spray-data pmdec  pmra   l")
        for j, bc in enumerate(ts["b"]):
            f = lambda d, q: d[q]["mu"][j]
            print(f"   {bc:4.0f}  {f(ts,'pmdec')-f(tt,'pmdec'):+6.2f}  {f(ts,'pmra')-f(tt,'pmra'):+6.2f}  {f(ts,'l')-f(tt,'l'):+6.2f}"
                  f"   | {tt['n'][j]:5d} {ts['n'][j]:5d} | {f(ts,'pmdec')-data['pmdec']['mu'][j]:+6.2f} {f(ts,'pmra')-data['pmra']['mu'][j]:+6.2f} {f(ts,'l')-data['l']['mu'][j]:+6.2f}")
        out[run] = dict(spray_s=dt, n_tracer=nt, n_spray=ns, tracer={q: {k: np.asarray(v).tolist() for k, v in tt[q].items()} for q in ("l", "pmra", "pmdec")},
                        spray={q: {k: np.asarray(v).tolist() for k, v in ts[q].items()} for q in ("l", "pmra", "pmdec")}, b=ts["b"].tolist())
        for jj, q in enumerate(("l", "pmra", "pmdec")):
            x = ax[i, jj]
            x.errorbar(data["b"], data[q]["mu"], yerr=data[q]["sig_mu"], fmt="o", color="k", ms=4, label="data (stream 54)")
            x.errorbar(np.array(tt["b"])+0.25, tt[q]["mu"], yerr=tt[q]["sig_mu"], fmt="s", color="C0", ms=4, label="tracer run")
            x.errorbar(np.array(ts["b"])+0.5, ts[q]["mu"], yerr=ts[q]["sig_mu"], fmt="^", color="C3", ms=4, label="spray")
            x.set_title(f"{run}: {q}", fontsize=9); x.set_xlabel("b [deg]"); x.grid(alpha=0.3)
        ax[i, 0].invert_yaxis(); ax[i, 1].set_ylim(-16, -2); ax[i, 2].set_ylim(-12, -5); ax[i, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT/"plots/streams_spray_validation.png", dpi=75)
    (ROOT/"results/plot_data/streams_spray_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
