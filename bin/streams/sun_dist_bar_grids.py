#!/usr/bin/env python3
"""Codex follow-ups (docs/codex_distances_and_misfit_2026-10-10.md; plan steps 4-5). 32000 epochs (4x), seed 1, catalogue PMs
unless noted.
  T1 solar frame x omega Cen distance: frames baumgardt / ibata19 (frames.FRAMES) x d 5.43, 5.6, 5.8 kpc, for
     W   = working literature-angle model (Omega_b 36, 28 deg, amp 1.2, bar size 1.15, catalogue PMs) and
     R16 = 16-deg reference (34.5, 16 deg, 1.4, size 1, grid PMs -3.2223, -6.7517)            -- 12 sprays;
  T2 coupled bar angle x bar size: angle 16, 20, 24, 28 x size 1.0, 1.15, 1.3 x Omega_b 34-37 x amp 1.2, 1.4
     (baumgardt frame, d 5.6, catalogue PMs)                                                    -- 96 sprays.
Stores lnL, segment sums (chord phi1 < 4, 4-17, > 17) and footprint pmra/pmdec medians per 2-deg phi1 bin.
Output results/plot_data/sun_dist_bar_grids.json (resumable). Usage: OMP_NUM_THREADS=8 python bin/streams/sun_dist_bar_grids.py [T1|T2]
"""
import itertools, json, sys, time
import numpy as np
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, OCEN_OBS, PMRA, PMDEC

CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"])
CFG = []
for fr, dd in itertools.product(("baumgardt", "ibata19"), (5.43, 5.6, 5.8)):
    CFG.append(dict(test="T1", model="W", om=36., an=28., am=1.2, size=1.15, frame=fr, dist=dd, pm=CAT))
    CFG.append(dict(test="T1", model="R16", om=34.5, an=16., am=1.4, size=1.0, frame=fr, dist=dd, pm=(PMRA, PMDEC)))
for an, sz, om, am in itertools.product((16., 20., 24., 28.), (1.0, 1.15, 1.3), (34., 35., 36., 37.), (1.2, 1.4)):
    CFG.append(dict(test="T2", model="grid", om=om, an=an, am=am, size=sz, frame="baumgardt", dist=5.6, pm=CAT))
only = sys.argv[1] if len(sys.argv) > 1 else None
SEG = [(-99, 4), (4, 17), (17, 99)]; edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
outd = ROOT/"results/streams/sun_dist_bar"; outd.mkdir(parents=True, exist_ok=True); outp = ROOT/"results/plot_data/sun_dist_bar_grids.json"
res = json.loads(outp.read_text()) if outp.exists() else []; done = {r["tag"] for r in res}
for c in CFG:
    if only and c["test"] != only:
        continue
    tag = f"{c['test']}_{c['model']}_om{c['om']:g}_an{c['an']:g}_am{c['am']:g}_sz{c['size']:g}_{c['frame']}_d{c['dist']:g}"
    if tag in done:
        continue
    t0 = time.time(); f = outd/f"{tag}.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        m = make_spray(c["om"], c["an"], c["am"], c["dist"], nrel=32000, seed=1, pm=c["pm"], host_kw=dict(bar_size=c["size"]), frame=c["frame"])
        np.savez_compressed(f, **m)
    s = score(m); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0); ib = np.digitize(mu[k], edges)-1
    med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else None for j in range(len(cen))] for q in ("pmra", "pmdec")}
    r = dict(c, tag=tag, pm=list(c["pm"]), lnL=s["total"], seg=[float(s["per_star"][(du >= a) & (du < b)].sum()) for a, b in SEG], med=med, phi1=cen.tolist())
    res.append(r); outp.write_text(json.dumps(res))
    print(f"{tag:55s} lnL {s['total']:9.1f}  seg " + " ".join(f"{x:9.1f}" for x in r["seg"]) + f"  [{time.time()-t0:.0f} s]", flush=True)
