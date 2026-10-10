#!/usr/bin/env python3
"""Two checks with existing sprays (4x particles, seed 1; best grid4 models at 16, 20, 24 and 28 deg, as D1):
(5) Spread: in the knee (chord-frame phi1 15-27, 2-deg bins, footprint |dphi2| < 6 deg) the pmra and pmdec distributions of the
    16-deg and 28-deg model debris (age < 700 Myr, chi > 0) vs the members: medians, 16-84% ranges, histograms; and the
    PM-only per-star Delta lnL(16 - 28) (D1 single-observable score) summed in bins of the member's pmdec offset from the 28-deg
    model median, to see which members drive the 16-deg gain. Figure plots/knee_spread.png.
(6) Frame sensitivity: score_conditional in the independent spline-path frame (s, x) of results/plot_data/stream54_path.json instead
    of the chord great-circle frame; Delta lnL vs 28 deg per segment, with segments defined by the same members as in D1
    (chord phi1 < 4, 4-17, > 17). Output results/plot_data/frame_check.json.
Usage: python bin/streams/knee_spread_and_frame.py
"""
import json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, project_gc, GC, du, dx, d, cov, score_conditional
from ocen_dm.streams.path import project, load
from d1_segments import MODELS, single_scores

M = {a: dict(np.load(ROOT/"results/streams"/p)) for a, (p, _) in MODELS.items()}
for m in M.values():
    m["u"], m["x"] = project_gc(m["l"], m["b"], GC)
# ---- (5) spread in the knee
edges = np.arange(15., 27.01, 2.); fig, ax = plt.subplots(2, len(edges)-1, figsize=(4*(len(edges)-1), 7.5)); out5 = {}
dsel = np.abs(dx) < 6
for j, (a0, a1) in enumerate(zip(edges[:-1], edges[1:])):
    mk = dsel & (du >= a0) & (du < a1); row = {}
    for r, q in enumerate(("pmra", "pmdec")):
        rng_ = (np.percentile(d[q][mk], 1)-1.5, np.percentile(d[q][mk], 99)+1.5)
        ax[r, j].hist(d[q][mk], bins=30, range=rng_, density=True, histtype="stepfilled", color="0.8", label=f"members ({mk.sum()})")
        row[f"members_{q}"] = np.percentile(d[q][mk], [16, 50, 84]).round(2).tolist()
        for a, c in ((16, "C3"), (28, "C0")):
            m = M[a]; k = (np.abs(m["x"]) < 6) & (m["age"] < 700) & (m["chi"] > 0) & (m["u"] >= a0) & (m["u"] < a1)
            ax[r, j].hist(m[q][k], bins=30, range=rng_, density=True, histtype="step", lw=1.6, color=c, label=f"{a} deg model ({k.sum()})")
            row[f"{a}_{q}"] = np.percentile(m[q][k], [16, 50, 84]).round(2).tolist()
        ax[r, j].set_xlabel(q); ax[r, j].set_title(f"phi1 {a0:g}-{a1:g}", fontsize=9)
    out5[f"{a0:g}-{a1:g}"] = row
    print(f"phi1 {a0:g}-{a1:g} pmdec 16/50/84: members {row['members_pmdec']}  16deg {row['16_pmdec']}  28deg {row['28_pmdec']} | pmra: "
          f"members {row['members_pmra']} 16deg {row['16_pmra']} 28deg {row['28_pmra']}")
ax[0, 0].legend(fontsize=7); fig.suptitle("Knee: PM distributions of the 16-deg (red) and 28-deg (blue) model debris vs members (grey), footprint |dphi2| < 6")
fig.tight_layout(); fig.savefig(ROOT/"plots/knee_spread.png", dpi=70)
s16, s28 = single_scores(M[16]), single_scores(M[28]); dpm = s16["pm"]-s28["pm"]
k28 = (np.abs(M[28]["x"]) < 6) & (M[28]["age"] < 700) & (M[28]["chi"] > 0)
med28 = np.array([np.median(M[28]["pmdec"][k28 & (np.abs(M[28]["u"]-u0) < 1)]) if (k28 & (np.abs(M[28]["u"]-u0) < 1)).sum() > 10 else np.nan for u0 in du])
off = d["pmdec"]-med28; kn = (du > 17) & np.isfinite(off)
print("knee members (phi1 > 17): PM-only Delta lnL(16 - 28) summed by the member's pmdec offset from the 28-deg model median:")
for lo, hi in ((-9, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 9)):
    s = kn & (off >= lo) & (off < hi); print(f"  offset {lo:+5.1f}..{hi:+5.1f} mas/yr: {s.sum():4d} members, sum {dpm[s].sum():+7.0f}, per member {dpm[s].mean() if s.sum() else 0:+.2f}")
# ---- (6) frame sensitivity: spline path frame
P = load(ROOT/"results/plot_data/stream54_path.json"); ds, dxs = project(d["l"], d["b"], P); win = (ds.min(), ds.max())
dW = np.column_stack((dxs, d["pmra"], d["pmdec"])); per = {}
for a, m in M.items():
    ms, mxs = project(m["l"], m["b"], P)
    per[a] = score_conditional(ds, dW, d["v"], d["e_v"], cov, ms, np.column_stack((mxs, m["pmra"], m["pmdec"])), m["vlos"], m["chi"], m["age"], u_window=win)["per_star"]
SEG = [(-99, 4, "phi1<4"), (4, 17, "phi1 4-17"), (17, 99, "phi1>17")]; out6 = {}
print("Spline-path frame: Delta lnL vs 28 deg (segments = same members as chord-frame phi1 segments):")
for a in (16, 20, 24):
    dl = per[a]-per[28]; row = {nm: float(dl[(du >= lo) & (du < hi)].sum()) for lo, hi, nm in SEG}; row["total"] = float(dl.sum()); out6[str(a)] = row
    print(f"  {a} deg: " + "  ".join(f"{k} {v:+7.0f}" for k, v in row.items()))
(ROOT/"results/plot_data/frame_check.json").write_text(json.dumps(dict(knee_spread=out5, spline_frame=out6), indent=1)); print("plots/knee_spread.png")
