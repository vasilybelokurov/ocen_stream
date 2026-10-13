#!/usr/bin/env python3
"""Spray model(s) vs Ibata+2024 stream 54 members: l-b, pmra-b, pmdec-b, v_los-b, distance-b (model heliocentric distance vs the
CMD-shift relative distances scaled to 5.43 kpc at b = 15-20). Trailing arm coloured by release age; leading arm grey.
Usage: python bin/streams/plot_spray_vs_data.py --pts 34.5,32,1.2 33,28,1.2 --nrel 20000 -> plots/spray_vs_data_best.png
"""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from ocen_dm.streams.restricted import AGAMA_T_MYR, OCEN_TODAY, host_potential
from ocen_dm.streams.frames import observables
from ocen_dm.streams.spray import spray

w = lambda x: np.where(x > 180, x-360, x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pts", nargs="+", default=["34.5,32,1.2"]); ap.add_argument("--nrel", type=int, default=20000)
    ap.add_argument("--tgyr", type=float, default=1.95558); a = ap.parse_args()
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
    g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic; gl, gb = w(g.l.deg), g.b.deg
    v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float); hv = np.isfinite(v) & (ev < 300)
    cmd = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())["bins"]
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    T = a.tgyr*1e3/AGAMA_T_MYR
    fig, ax = plt.subplots(len(a.pts), 5, figsize=(26, 5.2*len(a.pts)), squeeze=False)
    for i, p in enumerate(a.pts):
        om, an, am = map(float, p.split(","))
        host = host_potential("hunter24_bar", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am)
        sp = spray(host, OCEN_TODAY.copy(), T, r_kpc, M, n_release=a.nrel, seed=2)
        o = observables(sp["xv"]); l = w(o["l"]); tr = sp["arm"] == 1; age = sp["t_release_myr_ago"]
        sel = (o["b"] > 5) & (l < -15) & (l > -80)
        kt, kl = sel & tr, sel & ~tr
        kw = dict(c=age[kt], cmap="viridis_r", vmin=0, vmax=1000, s=2, zorder=3)
        a0 = ax[i, 0]; a0.plot(gl, gb, ".", color="0.6", ms=2, zorder=1); a0.plot(l[kl], o["b"][kl], ".", color="0.85", ms=1, zorder=0)
        sc = a0.scatter(l[kt], o["b"][kt], **kw); a0.set_xlim(-15, -80); a0.set_ylim(5, 50); a0.set_xlabel("l [deg]"); a0.set_ylabel("b [deg]")
        a0.set_title(f"Omega_b {om:g}, angle {an:g}, amp {am:g} (model A, 1.96 Gyr, {len(sp['xv'])} particles)", fontsize=9)
        for jj, (q, dq, yl) in enumerate((("pmra", np.asarray(t["pmRA"], float), (-22, 0)), ("pmdec", np.asarray(t["pmDE"], float), (-15, -2)))):
            x = ax[i, jj+1]; x.plot(gb, dq, ".", color="0.6", ms=2, zorder=1); x.scatter(o["b"][kt], o[q][kt], **kw)
            x.set_xlim(10, 45); x.set_ylim(*yl); x.set_xlabel("b [deg]"); x.set_ylabel(q + " [mas/yr]")
        x = ax[i, 3]; x.scatter(o["b"][kt], o["vlos"][kt], **kw); x.plot(gb[hv], v[hv], "o", color="k", mfc="w", ms=6, zorder=5)
        x.set_xlim(10, 45); x.set_ylim(120, 300); x.set_xlabel("b [deg]"); x.set_ylabel("v_los [km/s]")
        x = ax[i, 4]; x.scatter(o["b"][kt], o["dist"][kt], **kw)
        x.errorbar([0.5*(c["b"][0]+c["b"][1]) for c in cmd]+[17.5], [5.43*c["d_ratio"] for c in cmd]+[5.43],
                   yerr=[5.43*c["d_ratio"]*np.log(10)/5*c["dm_err"] for c in cmd]+[0], fmt="s", color="k", mfc="w", ms=7, zorder=5)
        x.set_xlim(10, 45); x.set_ylim(3, 7); x.set_xlabel("b [deg]"); x.set_ylabel("distance [kpc] (data: CMD shift, ref 5.43)")
        for x in ax[i]:
            x.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax[i, 4], label="release age [Myr] (trailing arm)")
    fig.suptitle("Best spray-grid models vs Ibata+2024 stream 54 (grey; v_los and CMD distances: open symbols)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/"plots/spray_vs_data_best.png", dpi=72); print("plots/spray_vs_data_best.png")


if __name__ == "__main__":
    main()
