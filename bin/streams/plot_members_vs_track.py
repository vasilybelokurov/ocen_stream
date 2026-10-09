#!/usr/bin/env python3
"""Measured observables of Ibata+2024 stream 54 (members grey; Stage-0 track black with errors; 29 v_los open circles; CMD-shift
distances open squares, scaled to 5.43 kpc at b = 15-20) vs the spray model track (red): medians of each observable in 5 kpc Myr
bins of the Gibbons phase chi for chi <= 105 (where the model stream is single-valued in chi), trailing arm, age < 700 Myr.
Panels: l-b, and l, pmra, pmdec, v_los, distance vs b.
Usage: python bin/streams/plot_members_vs_track.py om34.5_an24_am1.2 [--kde] -> plots/members_vs_track_<tag>[_kde].png
--kde: model track = mode of the per-chi-bin 4D KDE (score.chi_kde_track) instead of separate medians.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data
sys.path.insert(0, str(ROOT/'src'))
from ocen_dm.streams.score import chi_kde_track

d = load_data()
tr = json.loads((ROOT/"results/plot_data/stream54_track.json").read_text())["all"]
cmd = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())["bins"]
KDE = "--kde" in sys.argv
for tag in [a for a in sys.argv[1:] if not a.startswith("--")]:
    m = np.load(ROOT/f"results/streams/spray_grid2/{tag}.npz")
    k = (m["chi"] > 0) & (m["age"] < 700)
    edges = np.arange(0, 105.1, 5.); T = {q: [] for q in ("chi", "l", "b", "pmra", "pmdec", "vlos", "d")}
    for e0, e1 in zip(edges[:-1], edges[1:]):
        s = k & (m["chi"] >= e0) & (m["chi"] < e1)
        if s.sum() >= 15:
            T["chi"].append(0.5*(e0+e1))
            for q in ("l", "b", "pmra", "pmdec", "vlos", "d"):
                T[q].append(np.median(m[q][s]))
    T = {q: np.array(v) for q, v in T.items()}
    if KDE:
        T = chi_kde_track(dict(m))
    fig, ax = plt.subplots(2, 3, figsize=(19, 10)); ax = ax.ravel()
    a = ax[0]; a.plot(d["l"], d["b"], ".", color="0.65", ms=1.5); a.plot(T["l"], T["b"], "r.-", lw=2, ms=6)
    a.set_xlim(-30, -70); a.set_ylim(10, 45); a.set_xlabel("l [deg]"); a.set_ylabel("b [deg]")
    for a, (q, lab, yl, dq) in zip(ax[1:], (("l", "l [deg]", (-65, -35), d["l"]), ("pmra", "pmra [mas/yr]", (-20, -1), d["pmra"]),
                                             ("pmdec", "pmdec [mas/yr]", (-13, -4), d["pmdec"]), ("vlos", "v_los [km/s]", (150, 280), None),
                                             ("d", "distance [kpc]", (3, 6.5), None))):
        if dq is not None:
            a.plot(d["b"], dq, ".", color="0.65", ms=1.5)
            a.errorbar(tr["b"], tr[q]["mu"], yerr=tr[q]["sig_mu"], fmt="o", color="k", ms=5, zorder=4, label="measured track")
        if q == "vlos":
            hv = np.isfinite(d["v"]); a.errorbar(d["b"][hv], d["v"][hv], yerr=d["e_v"][hv], fmt="o", color="k", mfc="w", ms=6, label="members with v_los")
        if q == "d":
            a.errorbar([17.5]+[0.5*(c["b"][0]+c["b"][1]) for c in cmd], [5.43]+[5.43*c["d_ratio"] for c in cmd],
                       yerr=[0]+[5.43*c["d_ratio"]*np.log(10)/5*c["dm_err"] for c in cmd], fmt="s", color="k", mfc="w", ms=7,
                       label="CMD-shift distance (ref 5.43 at b 15-20)")
        a.plot(T["b"], T[q], "r.-", lw=2, ms=6, zorder=5, label="model track (chi <= 105)" + (": KDE mode per chi bin" if KDE else ": medians per chi bin"))
        a.set_xlim(10, 45); a.set_ylim(*yl); a.set_xlabel("b [deg]"); a.set_ylabel(lab); a.legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Measured stream-54 observables vs model track ({'KDE mode' if KDE else 'median'} per chi bin; spray {tag}, model A, 1.96 Gyr)", fontsize=11)
    out = ROOT/f"plots/members_vs_track_{tag}{'_kde' if KDE else ''}.png"; fig.tight_layout(); fig.savefig(out, dpi=75); print(out)
