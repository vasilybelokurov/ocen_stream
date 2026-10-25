#!/usr/bin/env python3
"""Investigate stream unwrapping for the spray models: Gibbons+2014 phase chi, orbital-plane angle psi (Chemaly+2026 style)
and release age as ordering coordinates. For each model: (1) monotonicity of l, b vs each coordinate along the trailing arm,
(2) sky map coloured by each coordinate, (3) observables vs psi / chi with the data mapped where possible.
Usage: python bin/streams/test_unwrapping.py --pts 33,28,1.0 34.5,32,1.2 --nrel 6000 --window 800
       -> plots/unwrapping_<pt>.png, results/plot_data/unwrapping_<pt>.npz
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import spearmanr
from ocen_stream.restricted import AGAMA_T_MYR, OCEN_TODAY, host_potential
from ocen_stream.frames import observables
from ocen_stream.spray import spray_unwrapped

w = lambda x: np.where(x > 180, x-360, x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pts", nargs="+", default=["33,28,1.0"]); ap.add_argument("--nrel", type=int, default=6000)
    ap.add_argument("--window", type=float, default=800., help="release over the last <window> Myr"); a = ap.parse_args()
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
    g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic; gl, gb = w(g.l.deg), g.b.deg
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    T = 1955.58/AGAMA_T_MYR
    for p in a.pts:
        om, an, am = map(float, p.split(","))
        host = host_potential("x", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am)
        t0 = time.time()
        sp = spray_unwrapped(host, OCEN_TODAY.copy(), T, r_kpc, M, np.linspace(T-a.window/AGAMA_T_MYR, T, a.nrel), seed=3)
        dt = time.time()-t0
        o = observables(sp["xv"]); l = w(o["l"]); tr = sp["arm"] == 1
        tag = p.replace(",", "_")
        np.savez_compressed(ROOT/f"results/plot_data/unwrapping_{tag}.npz", xv=sp["xv"], arm=sp["arm"], chi=sp["chi"], psi=sp["psi"],
                            age=sp["t_release_myr_ago"])
        print(f"\n{p}: {len(sp['xv'])} particles in {dt:.0f} s")
        print(f"  sign check: chi>0 for trailing (arm=+1): {np.mean(sp['chi'][tr] > 0):.3f}; chi<0 for leading: {np.mean(sp['chi'][~tr] < 0):.3f}")
        print(f"  psi sign: trailing psi<0: {np.mean(sp['psi'][tr] < 0):.3f}; leading psi>0: {np.mean(sp['psi'][~tr] > 0):.3f}")
        print(f"  Spearman rank correlation along the trailing arm: age-chi {spearmanr(sp['t_release_myr_ago'][tr], sp['chi'][tr])[0]:.3f}, "
              f"age-|psi| {spearmanr(sp['t_release_myr_ago'][tr], np.abs(sp['psi'][tr]))[0]:.3f}, chi-|psi| {spearmanr(sp['chi'][tr], np.abs(sp['psi'][tr]))[0]:.3f}")
        # medians along |psi| for the trailing arm
        ps = -sp["psi"][tr]; order = np.argsort(ps)
        print("  trailing arm binned in -psi [deg]: N, median l, b, pmra, pmdec, vlos, d, age, chi; frac in observed window (b 15-41, l -62..-42)")
        for lo in range(0, 400, 20):
            s = (ps >= lo) & (ps < lo+20)
            if s.sum() < 10:
                continue
            L, B = l[tr][s], o["b"][tr][s]
            print(f"   {lo:3d}-{lo+20:3d} {s.sum():5d}  {np.median(L):6.1f} {np.median(B):6.1f} {np.median(o['pmra'][tr][s]):6.1f} {np.median(o['pmdec'][tr][s]):6.1f} "
                  f"{np.median(o['vlos'][tr][s]):5.0f} {np.median(o['dist'][tr][s]):5.2f} {np.median(sp['t_release_myr_ago'][tr][s]):5.0f} {np.median(sp['chi'][tr][s]):8.0f}"
                  f"   {np.mean((B > 15) & (B < 41) & (L > -62) & (L < -42)):.2f}")
        fig, ax = plt.subplots(2, 4, figsize=(24, 10))
        for j, (key, lab, vmin, vmax) in enumerate((("age", "release age [Myr]", 0, a.window), ("chi", "Gibbons chi [kpc Myr]", None, None),
                                                    ("psi", "psi [deg] (orbital-plane angle)", None, None))):
            val = {"age": sp["t_release_myr_ago"], "chi": sp["chi"], "psi": sp["psi"]}[key][tr]
            if vmin is None:
                vmin, vmax = np.percentile(val, [2, 98])
            x = ax[0, j]; x.plot(gl, gb, ".", color="0.6", ms=1.5); sc = x.scatter(l[tr], o["b"][tr], c=val, s=2, cmap="viridis", vmin=vmin, vmax=vmax)
            x.set_xlim(-10, -80); x.set_ylim(0, 50); x.set_xlabel("l"); x.set_ylabel("b"); fig.colorbar(sc, ax=x, label=lab)
        for j, (q, yl) in enumerate((("l", (-75, -20)), ("b", (0, 50)), ("pmra", (-22, 0)), ("pmdec", (-15, -3)))):
            x = ax[1, j]; val = l[tr] if q == "l" else o[q][tr]
            x.scatter(-sp["psi"][tr], val, c=sp["t_release_myr_ago"][tr], s=2, cmap="viridis", vmin=0, vmax=a.window)
            x.set_xlabel("-psi [deg] (along the trailing arm)"); x.set_ylabel(q); x.set_ylim(*yl); x.set_xlim(-5, 300); x.grid(alpha=0.3)
        ax[0, 3].scatter(sp["t_release_myr_ago"][tr], sp["chi"][tr], s=2, c=-sp["psi"][tr], cmap="plasma")
        ax[0, 3].set_xlabel("release age [Myr]"); ax[0, 3].set_ylabel("chi [kpc Myr]"); ax[0, 3].grid(alpha=0.3)
        fig.suptitle(f"Unwrapping test: spray model A, bar Omega {om:g}, angle {an:g}, amp {am:g}; trailing arm (grey: stream 54)", fontsize=11)
        fig.tight_layout(); fig.savefig(ROOT/f"plots/unwrapping_{tag}.png", dpi=70)


if __name__ == "__main__":
    main()
