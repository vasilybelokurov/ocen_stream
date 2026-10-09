#!/usr/bin/env python3
"""Re-score the saved sprays (results/streams/spray_grid2/*.npz: grid 3 and the fit best points) with score_conditional in three
variants: A = chord great-circle frame (u = phi1, x = dphi2; results/plot_data/stream54_gc_frame_chord.json); B = A without the
broad near-cluster region (members and particles with phi1 < 4 deg; predeclared); C = spline path (u = s, x;
results/plot_data/stream54_path.json). Fixed bandwidths (0.5 deg, 0.5 deg, 0.2, 0.2 mas/yr, 5 km/s); eps 0.05; u window = members' range.
Compares with the robust chi-KDE ranking (results/plot_data/spray_bar_grid3.json).
Usage: python bin/streams/rescore_conditional.py -> results/plot_data/rescore_conditional.json
"""
import json, sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from scipy.stats import spearmanr
from spray_bar_grid2 import load_data
from ocen_dm.streams.path import project, project_gc
from ocen_dm.streams.score import score_conditional

d = load_data(); n = len(d["l"])
cov = np.zeros((n, 2, 2)); cov[:, 0, 0] = d["e_pmra"]**2; cov[:, 1, 1] = d["e_pmdec"]**2; cov[:, 0, 1] = cov[:, 1, 0] = d["rho"]*d["e_pmra"]*d["e_pmdec"]
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text()); SP = json.loads((ROOT/"results/plot_data/stream54_path.json").read_text())
du_gc, dx_gc = project_gc(d["l"], d["b"], GC); du_sp, dx_sp = project(d["l"], d["b"], SP)
g3 = {f"om{x['omega']:g}_an{x['angle']:g}_am{x['amp']:g}": x for x in json.loads((ROOT/"results/plot_data/spray_bar_grid3.json").read_text())["grid"]}
tags = sorted(set(g3) | {"fit28_best", "fitfree_best"})
out = {}
for t in tags:
    f = ROOT/f"results/streams/spray_grid2/{t}.npz"
    if not f.exists():
        continue
    m = np.load(f); r = {}
    mu_gc, mx_gc = project_gc(m["l"], m["b"], GC); mu_sp, mx_sp = project(m["l"], m["b"], SP)
    for name, (du, dx, mu, mx, keep_d, keep_m) in dict(
            A=(du_gc, dx_gc, mu_gc, mx_gc, np.ones(n, bool), np.ones(len(mu_gc), bool)),
            B=(du_gc, dx_gc, mu_gc, mx_gc, du_gc >= 4, mu_gc >= 4),
            C=(du_sp, dx_sp, mu_sp, mx_sp, np.ones(n, bool), np.ones(len(mu_sp), bool))).items():
        kd, km = keep_d, keep_m
        res = score_conditional(du[kd], np.column_stack((dx[kd], d["pmra"][kd], d["pmdec"][kd])), d["v"][kd], d["e_v"][kd], cov[kd],
                                mu[km], np.column_stack((mx[km], m["pmra"][km], m["pmdec"][km])), m["vlos"][km], m["chi"][km], m["age"][km],
                                u_window=(du[kd].min(), du[kd].max()))
        r[name] = res["total"]; r[name+"_bg"] = res["frac_bg"]
    r["old_chikde"] = g3[t]["chikde"] if t in g3 else None
    out[t] = r
    print(f"{t:22s} A {r['A']:10.1f}  B {r['B']:10.1f}  C {r['C']:10.1f}  (bg share A {r['A_bg']:.2f})" + (f"  old {r['old_chikde']:10.1f}" if r['old_chikde'] else ""), flush=True)
(ROOT/"results/plot_data/rescore_conditional.json").write_text(json.dumps(out, indent=1))
g = [t for t in out if out[t]["old_chikde"] is not None]
for name in ("A", "B", "C"):
    print(f"Spearman {name} vs old chi-KDE (grid 3): {spearmanr([out[t][name] for t in g], [out[t]['old_chikde'] for t in g])[0]:.3f}")
print(f"Spearman A vs C: {spearmanr([out[t]['A'] for t in g], [out[t]['C'] for t in g])[0]:.3f};  A vs B: {spearmanr([out[t]['A'] for t in g], [out[t]['B'] for t in g])[0]:.3f}")
for name in ("A", "B", "C", "old_chikde"):
    top = sorted(g+["fit28_best", "fitfree_best"] if name != "old_chikde" else g, key=lambda t: -out[t][name])[:6]; b = out[top[0]][name]
    print(f"top 6 by {name}: " + ", ".join(f"{t} ({out[t][name]-b:.0f})" for t in top))
