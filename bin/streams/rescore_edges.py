#!/usr/bin/env python3
"""Lower envelope of the knot chi2 vs each parameter over the rescored stored models + the fit_knot_nm.py evaluations (2026-10-25).

Question: is chi2 still falling at the fit box edges (distance 5.68, v_c 220, q 0.7 / 0.8 lower edge of stored sets)? Points: every
model (grey), lower envelope in parameter bins (black), fit_knot_nm.py chain evaluations (colour) and box limits (dashed).
Usage: python bin/streams/rescore_edges.py [CSV] -> plots/rescore_edges.png
"""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
CSV = sys.argv[1] if len(sys.argv) > 1 else ROOT/"results/plot_data/stream_model_rescore.csv"
PAR = ("angle", "omega", "amp", "size", "halo_q", "vc", "dist", "pmra", "pmdec")
LIM = dict(angle=(0, 50), omega=(20, 50), amp=(0.6, 2.0), size=(0.8, 1.5), halo_q=(0.7, 1.2), vc=(220, 250), dist=(5.28, 5.68),
           pmra=(-3.338, -3.162), pmdec=(-6.834, -6.658))

rows = [r for r in csv.DictReader(open(CSV)) if r.get("chi2_total")]
X = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in PAR}; C = np.array([float(r["chi2_total"]) for r in rows])
fk = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT/"results/streams/fitknot_20261025/runs/c*_e*.json")))]
fk = [r for r in fk if "chi2" in r]
fig, ax = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
for a, k in zip(ax.ravel(), PAR):
    ok = np.isfinite(X[k]) & (C < 1500); a.scatter(X[k][ok], C[ok], s=4, c="0.7", label="stored models (rescored)")
    e = np.linspace(np.nanmin(X[k][ok]), np.nanmax(X[k][ok]), 16); j = np.digitize(X[k][ok], e)
    env = [(0.5*(e[i-1]+e[i]), C[ok][j == i].min()) for i in range(1, len(e)) if (j == i).any()]
    a.plot(*np.array(env).T, "k-o", ms=3, label="lower envelope")
    for cid, col in zip(("c0", "c1", "c2", "c3"), ("C3", "C0", "C2", "C1")):
        s = [(r["params"][k], r["chi2"]["total"]) for r in fk if r["run_id"].startswith(cid)]
        a.scatter(*np.array(s).T, s=12, color=col, label=f"fit_knot_nm {cid}")
    for v in LIM[k]:
        a.axvline(v, color="k", ls="--", lw=0.8)
    a.set(xlabel=k, ylabel="knot chi2 (25 points)", ylim=(140, 600))
ax[0, 0].legend(fontsize=7)
fig.suptitle("chi2 vs parameter: all rescored stored models (grey; chi2 < 1500), lower envelope (black), free-fit evaluations (colour); dashed: fit box")
out = ROOT/"plots/rescore_edges.png"; fig.savefig(out, dpi=80); print(out.relative_to(ROOT))
