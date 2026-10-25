#!/usr/bin/env python3
"""Spray of model A in the Hunter+2024 bar (Omega_b, angle 28) for several disruption times; trailing-arm particles through the
Stage-0 track estimator vs the data. Particle number scales with T (n_release = rate x T).
Usage: python bin/streams/spray_length_test.py --omega 33 --tgyr 1.96 3 5 --rate 10000
       -> plots/streams_spray_length_bar<omega>.png, results/plot_data/streams_spray_length_bar<omega>.json
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ocen_stream.restricted import AGAMA_T_MYR, OCEN_TODAY, host_potential
from ocen_stream.frames import observables
from ocen_stream.spray import spray
from ocen_stream.track import measure

w = lambda x: np.where(x > 180, x-360, x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--omega", type=float, default=33.); ap.add_argument("--angle", type=float, default=28.)
    ap.add_argument("--tgyr", type=float, nargs="+", default=[1.96, 3., 5.]); ap.add_argument("--rate", type=float, default=10000, help="release epochs per Gyr")
    a = ap.parse_args()
    data = json.loads((ROOT/"results/plot_data/stream54_track.json").read_text())["all"]
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text())
    r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    fig, ax = plt.subplots(2, 3, figsize=(18, 9)); out = {}
    for c, tg in zip(("C0", "C2", "C3", "C1"), a.tgyr):
        T = tg*1e3/AGAMA_T_MYR
        host = host_potential("hunter24_bar", bar_omega=a.omega, bar_angle_deg=a.angle, t_today=T)
        t0 = time.time(); sp = spray(host, OCEN_TODAY.copy(), T, r_kpc, M, n_release=int(a.rate*tg)); dt = time.time()-t0
        xv = sp["xv"][sp["arm"] == 1]; age = sp["t_release_myr_ago"][sp["arm"] == 1]
        o = observables(xv); l = w(o["l"]); k = (o["b"] > 15) & (l < -20) & (l > -75)
        tr = measure(l[k], o["b"][k], o["pmra"][k], o["pmdec"][k], nboot=40)
        print(f"T = {tg} Gyr: {len(sp['xv'])} particles in {dt:.0f} s; northern arm N {k.sum()}; n per bin {tr['n'].tolist()}")
        print("   b   l(model-data)  pmra  pmdec   median release age [Myr]")
        for j, bc in enumerate(tr["b"]):
            kk = k & (o["b"] >= bc-1) & (o["b"] < bc+1)
            print(f"  {bc:4.0f}  {tr['l']['mu'][j]-data['l']['mu'][j]:+6.2f}  {tr['pmra']['mu'][j]-data['pmra']['mu'][j]:+6.2f}  "
                  f"{tr['pmdec']['mu'][j]-data['pmdec']['mu'][j]:+6.2f}   {np.median(age[kk]) if kk.any() else np.nan:6.0f}")
        out[str(tg)] = dict(seconds=dt, n=tr["n"].tolist(), **{q: {kk: np.asarray(v).tolist() for kk, v in tr[q].items()} for q in ("l", "pmra", "pmdec")})
        ax[0, 0].plot(l[k], o["b"][k], ".", ms=1, color=c, alpha=0.4, label=f"{tg} Gyr")
        for jj, q in enumerate(("l", "pmra", "pmdec")):
            ax[1, jj].errorbar(np.array(tr["b"])+0.15*a.tgyr.index(tg), tr[q]["mu"], yerr=tr[q]["sig_mu"], fmt="^", color=c, ms=4, label=f"spray {tg} Gyr")
    import os, astropy.coordinates as coord
    from astropy.table import Table
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
    g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
    ax[0, 0].plot(w(g.l.deg), g.b.deg, ".", ms=1.5, color="k", alpha=0.3, label="stream 54")
    ax[0, 0].set_xlim(-20, -80); ax[0, 0].set_ylim(10, 50); ax[0, 0].set_xlabel("l"); ax[0, 0].set_ylabel("b"); ax[0, 0].legend(fontsize=8, markerscale=6)
    for jj, q in enumerate(("l", "pmra", "pmdec")):
        x = ax[1, jj]; x.errorbar(data["b"], data[q]["mu"], yerr=data[q]["sig_mu"], fmt="o", color="k", ms=5, label="data")
        x.set_xlabel("b [deg]"); x.set_ylabel(q); x.grid(alpha=0.3); x.legend(fontsize=8)
    ax[1, 0].invert_yaxis(); ax[1, 1].set_ylim(-16, -2); ax[1, 2].set_ylim(-12, -5)
    ax[0, 1].axis("off"); ax[0, 2].axis("off")
    fig.suptitle(f"Spray, model A, Hunter+24 bar Omega_b = {a.omega:g}, angle {a.angle:g}: disruption time", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/f"plots/streams_spray_length_bar{a.omega:g}.png", dpi=75)
    (ROOT/f"results/plot_data/streams_spray_length_bar{a.omega:g}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
