#!/usr/bin/env python3
"""Present-day debris within 20 deg of omega Cen for the three DM models vs the Ibata+2024 STREAMFINDER members.

Columns: A (no DM), B (moderate DM), C (5x DM inside 200 pc) from bin/streams/run_prescribed.py (unbound star tracers
today), and Ibata+2024 (ApJ 967, 89; 2024ApJ...967...89I) streams 54 (Fimbulthul) and 55 (Fimbulthul-S),
$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits. Rows: l vs b, l vs pmra*, l vs pmdec, l vs v_los (heliocentric).
Debris = tracers unbound in the satellite potential about the known centre (the point-mass orbit ends 0.5 pc
from omega Cen's catalogue position, OCEN_TODAY). Selection: angular distance < 20 deg from omega Cen (l, b) = (309.10, +14.97). Model phase-space -> observables
with the solar parameters of bin/nbody/analyse_tails.py (R0 = 8.178 kpc, Vsun = (11.1, 252.24, 7.25) km/s,
z_sun = 0), which reproduce Baumgardt's heliocentric values for omega Cen. Observed v_los only where measured.
Options: --radius (deg, default 20); --xlower b|l: abscissa of the three lower rows (default l); the top row is
always l vs b. Output names carry the radius and abscissa unless they are the defaults.
Usage: python bin/streams/plot_debris_vs_ibata.py [--radius 40 --xlower b]
Output: plots/streams_debris_vs_ibata2024[_r40_xb].png, results/plot_data/streams_debris_vs_ibata2024[...].json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ocen-df-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table

from ocen_dm.streams.analysis import load, run_core, run_particles
from ocen_dm.streams.restricted import OCEN_TODAY, bound_set

R_SUN_KPC, V_SUN = 8.178, (11.1, 12.24+240.0, 7.25)
OCEN_LB = (309.10, 14.97)
RAD = 20.
MODELS = [("A_nodm", "A: no DM", "#2a78d6"), ("B_dm_phot", "B: moderate DM", "#eb6834"),
          ("C_dm5x_shell", "C: 5x DM inside 200 pc", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def observables(xv):
    gc = coord.Galactocentric(x=-xv[:, 0]*u.kpc, y=xv[:, 1]*u.kpc, z=xv[:, 2]*u.kpc,
                              v_x=-xv[:, 3]*u.km/u.s, v_y=xv[:, 4]*u.km/u.s, v_z=xv[:, 5]*u.km/u.s,
                              galcen_distance=R_SUN_KPC*u.kpc, z_sun=0*u.pc,
                              galcen_v_sun=coord.CartesianDifferential(V_SUN*u.km/u.s))
    ic = gc.transform_to(coord.ICRS()); g = gc.transform_to(coord.Galactic())
    return dict(l=np.asarray(g.l.deg), b=np.asarray(g.b.deg), pmra=np.asarray(ic.pm_ra_cosdec.value),
                pmdec=np.asarray(ic.pm_dec.value), vlos=np.asarray(ic.radial_velocity.value), dist=np.asarray(ic.distance.kpc))


def sep_deg(l, b):
    c0 = coord.SkyCoord(*OCEN_LB, unit="deg", frame="galactic")
    return coord.SkyCoord(l, b, unit="deg", frame="galactic").separation(c0).deg


def main():
    global RAD
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--radius", type=float, default=20.)
    ap.add_argument("--xlower", choices=("l", "b"), default="l")
    ap.add_argument("--runs", default="prescribed", help="results/streams/<runs>/<model> (e.g. prescribed_rot)")
    args = ap.parse_args()
    RAD = args.radius
    tag = ("" if (RAD == 20. and args.xlower == "l") else f"_r{RAD:g}_x{args.xlower}") + ("" if args.runs == "prescribed" else f"_{args.runs}")
    cols = []
    out = {}
    for key, lab, col in MODELS:
        R = ROOT/"results/streams"/args.runs/key
        m, s = run_particles(R)
        _, xv = load(R, name="snap_today.npz")
        c = OCEN_TODAY.copy()                     # the prescribed centre ends 0.5 pc from today's position (run.json)
        near = np.linalg.norm(xv[:, :3]-c[:3], axis=1) < 0.5
        b, _ = bound_set(xv, c, m, start=near, core=run_core(R))
        o = observables(xv[~b]); oc = observables(c[None, :])
        k = sep_deg(o["l"], o["b"]) < RAD
        cols.append((lab, col, {q: v[k] for q, v in o.items()}, oc))
        out[key] = dict(n_within=int(k.sum()), cluster={q: float(v[0]) for q, v in oc.items()})
    t = Table.read(os.path.expanduser("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits"))
    st = np.asarray(t["Stream"])
    ra, de = np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float)
    g = coord.SkyCoord(ra, de, unit="deg").galactic
    l, b = g.l.deg, g.b.deg
    v = np.asarray(t["VHel"].filled(np.nan) if hasattr(t["VHel"], "filled") else t["VHel"], float)
    ev = np.asarray(t["e_VHel"], float); v[(ev >= 300) | ~np.isfinite(ev)] = np.nan
    obs = {}
    for sid, name, col in ((54, "Fimbulthul (54)", "#4a4945"), (55, "Fimbulthul-S (55)", "#b0864f")):
        k = (st == sid) & (sep_deg(l, b) < RAD)
        obs[name] = (col, dict(l=l[k], b=b[k], pmra=np.asarray(t["pmRA"], float)[k], pmdec=np.asarray(t["pmDE"], float)[k], vlos=v[k]))
        out[f"ibata2024_{sid}"] = dict(n_within=int(k.sum()), n_vlos=int(np.isfinite(v[k]).sum()))
    wide = RAD > 40                      # wide fields include Fimbulthul-S (stream 55, ~3 kpc, large PMs)
    rows = [("b", "b [deg]", (max(OCEN_LB[1]-RAD, -90), min(OCEN_LB[1]+RAD, 90))),
            ("pmra", r"$\mu_{\alpha*}$ [mas/yr]", (-25, 8) if wide else (-15, 8)),
            ("pmdec", r"$\mu_\delta$ [mas/yr]", (-35, 8) if wide else (-15, 8)),
            ("vlos", r"$v_{\rm los}$ [km/s]", (-100, 450) if wide else (-50, 450))]
    fig, ax = plt.subplots(4, 4, figsize=(19, 17))
    wrap = lambda x: np.where(x > 180, x-360, x)
    xl = (wrap(np.array([OCEN_LB[0]+RAD]))[0], wrap(np.array([OCEN_LB[0]-RAD]))[0])
    xb = (max(OCEN_LB[1]-RAD, -90), min(OCEN_LB[1]+RAD, 90))

    def xof(o, i):
        return o["b"] if (i > 0 and args.xlower == "b") else wrap(o["l"])
    for j, (lab, col, o, oc) in enumerate(cols):
        for i, (q, ylab, yl) in enumerate(rows):
            a = ax[i, j]
            a.plot(xof(o, i), o[q], ".", color=col, ms=2.5 if RAD <= 20 else 1.5, alpha=0.5, rasterized=True)
            a.plot(xof(oc, i), oc[q], "*", color=INK, ms=13, mec="white", mew=0.6)
            if i == 0:
                a.set_title(f"{lab}\nunbound tracers within {RAD:g} deg: N = {len(o['l']):,}", fontsize=10, color=INK, loc="left")
    for i, (q, ylab, yl) in enumerate(rows):
        a = ax[i, 3]
        for name, (col, o) in obs.items():
            a.plot(xof(o, i), o[q], "o", color=col, ms=3 if q != "vlos" else 6, alpha=0.8, mec="white", mew=0.3,
                   label=f"{name}: N = {np.isfinite(o[q]).sum()}")
        a.plot(xof(cols[0][3], i), [cols[0][3][q][0]], "*", color=INK, ms=13, mec="white", mew=0.6, label="omega Cen")
        a.legend(fontsize=8, frameon=False, loc="best")
        if i == 0:
            a.set_title(f"Ibata+2024 STREAMFINDER (Gaia DR3)\nwithin {RAD:g} deg", fontsize=10, color=INK, loc="left")
    for i, (q, ylab, yl) in enumerate(rows):
        for j in range(4):
            a = ax[i, j]
            a.set_xlim(*(xb if (i > 0 and args.xlower == "b") else xl)); a.set_ylim(*yl); a.grid(True, color=GRID, lw=0.5)
            a.tick_params(colors=MUTED, labelsize=9)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if j == 0:
                a.set_ylabel(ylab, color=INK)
            if i == 0 or i == 3 or args.xlower == "b":
                a.set_xlabel("b [deg]" if (i > 0 and args.xlower == "b") else "l [deg]", color=INK)
    fig.suptitle(f"Debris within {RAD:g} deg of omega Cen today: three DM models (prescribed potential, McMillan17, 1.96 Gyr) "
                 f"vs Ibata+2024 members [{args.runs}]", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT/f"plots/streams_debris_vs_ibata2024{tag}.png", dpi=110)
    (ROOT/f"results/plot_data/streams_debris_vs_ibata2024{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
