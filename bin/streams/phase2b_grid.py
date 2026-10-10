#!/usr/bin/env python3
"""Phase 2b (docs/STREAM_DIAGNOSTIC_PLAN.md): follow-ups at the literature bar angle 28 deg, d 5.6 kpc, 32000 epochs, seed 1.
  longer bar: bar_size 1.15, 1.3 x amp 1.0, 1.2 x Omega_b 33, 34, 35, 36 x omega Cen PMs (catalogue, -1 sigma, -2 sigma in both;
              sigma 0.027 mas/yr) -- 48 sprays (size <= 1.3 keeps the bar inside corotation ~6.6 kpc at Omega_b 35);
  slowing bar: eta 0.002, 0.004 x Omega_b(today) 31, 32, 33, 34 (amp 1.2, size 1, catalogue PMs) -- 8 sprays.
Stores lnL, per-segment sums (chord phi1 < 4, 4-17, > 17) and footprint medians of pmra/pmdec per 2-deg phi1 bin.
Output results/plot_data/phase2b_grid.json (resumable; sprays results/streams/phase2b/). Usage: OMP_NUM_THREADS=8 python bin/streams/phase2b_grid.py
"""
import itertools, json, time
import numpy as np
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, OCEN_OBS

SIG = 0.027; CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"])
CFG = []
for sz, am, om, k in itertools.product((1.15, 1.3), (1.0, 1.2), (33., 34., 35., 36.), (0, 1, 2)):
    CFG.append(dict(kind="longer bar", om=om, am=am, size=sz, eta=None, pm_sig=k, pm=(CAT[0]-k*SIG, CAT[1]-k*SIG)))
for eta, om in itertools.product((0.002, 0.004), (31., 32., 33., 34.)):
    CFG.append(dict(kind="slowing bar", om=om, am=1.2, size=1.0, eta=eta, pm_sig=0, pm=CAT))
SEG = [(-99, 4), (4, 17), (17, 99)]; edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
outd = ROOT/"results/streams/phase2b"; outd.mkdir(parents=True, exist_ok=True); outp = ROOT/"results/plot_data/phase2b_grid.json"
res = json.loads(outp.read_text()) if outp.exists() else []; done = {r["tag"] for r in res}
for c in CFG:
    tag = f"{c['kind'][:4]}_om{c['om']:g}_am{c['am']:g}_sz{c['size']:g}_eta{c['eta'] or 0:g}_pm{c['pm_sig']}"
    if tag in done:
        continue
    t0 = time.time(); f = outd/f"{tag}.npz"
    hk = dict(bar_size=c["size"]) if c["eta"] is None else dict(bar_eta=c["eta"])
    if f.exists():
        m = dict(np.load(f))
    else:
        m = make_spray(c["om"], 28., c["am"], 5.6, nrel=32000, seed=1, pm=c["pm"], host_kw=hk); np.savez_compressed(f, **m)
    s = score(m); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0); ib = np.digitize(mu[k], edges)-1
    med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else None for j in range(len(cen))] for q in ("pmra", "pmdec")}
    r = dict(c, tag=tag, pm=list(c["pm"]), lnL=s["total"], seg=[float(s["per_star"][(du >= a) & (du < b)].sum()) for a, b in SEG], med=med, phi1=cen.tolist())
    res.append(r); outp.write_text(json.dumps(res))
    print(f"{tag:40s} lnL {s['total']:9.1f}  seg " + " ".join(f"{x:9.1f}" for x in r["seg"]) + f"  [{time.time()-t0:.0f} s]", flush=True)
