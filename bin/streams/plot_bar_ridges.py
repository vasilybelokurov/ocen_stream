#!/usr/bin/env python3
"""Ridge lines of the northern arm vs b (2-deg bins) for several runs over Ibata+2024 stream 54: l(b), pmra(b), pmdec(b).
Ridge = mode of the quantity in the bin (histogram, bin width 1 deg in l / 0.5 mas/yr in PM), refined as the median within
+-3 deg / +-1.5 mas/yr of the mode, so the dense arm (not the diffuse spray) is traced; bins with < 8 stars skipped.
Arm selection: b > 12, -75 < l < -20, unbound tracers (model) / all members (data).
Usage: python bin/streams/plot_bar_ridges.py prescribed_rot hunter_axi hunter_bar33 hunter_bar37.5 hunter_bar41
       -> plots/streams_bar_ridges.png, results/plot_data/streams_bar_ridges.json
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from ocen_dm.streams.analysis import load, run_core, run_particles
from ocen_dm.streams.restricted import bound_set
from ocen_dm.streams.frames import observables

w = lambda x: np.where(x > 180, x-360, x)
EDGES = np.arange(12, 46.1, 2.)


def ridge(b, q, step, half):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = (b >= lo) & (b < hi)
        if k.sum() < 8:
            out.append(np.nan); continue
        x = q[k]; h, e = np.histogram(x, bins=np.arange(x.min(), x.max()+step, step))
        mode = e[h.argmax()]+step/2; out.append(float(np.median(x[np.abs(x-mode) < half])))
    return np.array(out)


def profiles(l, b, pmra, pmdec):
    return dict(l=ridge(b, l, 1.0, 3.0), pmra=ridge(b, pmra, 0.5, 1.5), pmdec=ridge(b, pmdec, 0.5, 1.5), n=np.histogram(b, EDGES)[0])


def main():
    runs = sys.argv[1:]
    t = Table.read(os.path.expanduser("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
    g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
    gl, gb = w(g.l.deg), g.b.deg; k = (gb > 12) & (gl < -20) & (gl > -75)
    res = {"Ibata+2024 54": profiles(gl[k], gb[k], np.asarray(t["pmRA"], float)[k], np.asarray(t["pmDE"], float)[k])}
    for run in runs:
        R = ROOT/"results/streams"/run/"A_nodm"; rj = json.loads((R/"run.json").read_text())
        m, s = run_particles(R); _, xv = load(R, name="snap_today.npz"); c = np.array(rj["ocen_today"])
        bnd, _ = bound_set(xv, c, m, start=np.linalg.norm(xv[:, :3]-c[:3], axis=1) < 0.5, core=run_core(R))
        o = observables(xv[~bnd], rj.get("frame", "baumgardt"))
        k = (o["b"] > 12) & (w(o["l"]) < -20) & (w(o["l"]) > -75)
        if rj.get("bar_omega") is None:
            name = {"McMillan17": "McMillan17 axi", "hunter24_axi": "Hunter+24 axi"}.get(rj["mw"], rj["mw"])
        else:
            name = f"Hunter+24 bar {rj['bar_omega']:g}"
        res[name] = profiles(w(o["l"][k]), o["b"][k], o["pmra"][k], o["pmdec"][k])
    bc = 0.5*(EDGES[1:]+EDGES[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    styles = {"Ibata+2024 54": dict(color="k", lw=3, marker="o")}
    cols = iter(["0.55", "C0", "C2", "C1", "C3", "C4"])
    for name, p in res.items():
        st = styles.get(name, dict(color=next(cols), lw=1.8, marker="."))
        for a, q in zip(ax, ("l", "pmra", "pmdec")):
            a.plot(bc, p[q], label=name, **st)
    for a, q, yl in zip(ax, ("l [deg] (ridge)", "pmra* [mas/yr] (ridge)", "pmdec [mas/yr] (ridge)"), ((-30, -65), (-16, -2), (-12, -5))):
        a.set_xlabel("b [deg]"); a.set_ylabel(q); a.set_ylim(*yl); a.grid(alpha=0.3); a.set_xlim(12, 46)
    ax[0].legend(fontsize=8)
    fig.suptitle("Northern-arm ridges vs b: model A (fitted rotation, 1.96 Gyr) in axisymmetric and barred hosts vs Ibata+2024 stream 54", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/"plots/streams_bar_ridges.png", dpi=80)
    (ROOT/"results/plot_data/streams_bar_ridges.json").write_text(json.dumps(dict(b_centres=bc.tolist(),
        ridges={k: {q: np.asarray(v[q]).tolist() for q in ("l", "pmra", "pmdec", "n")} for k, v in res.items()}), indent=1))
    print("b:      " + " ".join(f"{x:6.0f}" for x in bc))
    for name, p in res.items():
        for q in ("l", "pmra", "pmdec"):
            print(f"{name[:18]:18s} {q:5s} " + " ".join(f"{x:6.1f}" for x in p[q]))


if __name__ == "__main__":
    main()
