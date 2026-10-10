#!/usr/bin/env python3
"""Does omega Cen's mass change the knee? W model (28 deg, bar size 1.15, Omega_b 36, amp 1.2, catalogue PMs, d 5.6, baumgardt),
4x particles, seeds 1-3; omega Cen's M(<r) (model A, 3.26e6 Msun) scaled by 0.7, 1.0, 1.5 at fixed radii. Records lnL, segment
sums, per-observable segment sums (obs_split) and footprint PM medians. Output results/plot_data/cluster_mass_test.json.
Usage: OMP_NUM_THREADS=8 python bin/streams/cluster_mass_test.py"""
import itertools, json, os, time
import numpy as np
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, OCEN_OBS
from obs_split import single_scores

CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"]); SEG = [(-99, 4), (4, 17), (17, 99)]
edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
outd = ROOT/"results/streams/cluster_mass"; outd.mkdir(parents=True, exist_ok=True); outp = ROOT/"results/plot_data/cluster_mass_test.json"
res = json.loads(outp.read_text()) if outp.exists() else []; done = {r["tag"] for r in res}
for mf, s in itertools.product((1.0, 0.7, 1.5), (1, 2, 3)):
    tag = f"mf{mf:g}_s{s}"
    if tag in done:
        continue
    t0 = time.time(); f = outd/f"{tag}.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        m = make_spray(36., 28., 1.2, 5.6, nrel=32000, seed=s, pm=CAT, host_kw=dict(bar_size=1.15), mass_factor=mf); np.savez_compressed(f, **m)
    sc = score(m); ss = single_scores(m); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0)
    ib = np.digitize(mu[k], edges)-1
    med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else None for j in range(len(cen))] for q in ("pmra", "pmdec")}
    sg = lambda v: [float(np.nansum(v[(du >= a) & (du < b)])) for a, b in SEG]
    r = dict(tag=tag, mass_factor=mf, seed=s, lnL=sc["total"], seg=sg(sc["per_star"]), seg_x=sg(ss["x"]), seg_pm=sg(ss["pm"]), med=med, phi1=cen.tolist())
    res.append(r); tmp = outp.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); os.replace(tmp, outp)
    print(f"{tag:10s} lnL {sc['total']:9.1f} seg " + " ".join(f"{x:8.0f}" for x in r["seg"]) + f"  [{time.time()-t0:.0f} s]", flush=True)
