#!/usr/bin/env python3
"""Codex follow-up (docs/codex_T1T2_next_steps_2026-10-10.md). 4x particles, seed 1, d 5.6.
  F  IC-frame x projection-frame cross terms at W (28 deg, size 1.15, Omega_b 36, amp 1.2, catalogue PMs); the diagonal cells
     are the T1 runs (sun_dist_bar_grids.py)                                                   -- 2 sprays;
  C  one-at-a-time frame controls at W (same frame for IC and projection): baumgardt with V_sun,y 232.24 / R0 8.122 / z_sun 17 pc
                                                                                                -- 3 sprays;
  E  T2 grid edges (baumgardt): 28 deg size 1.3 amp 1.6 x Omega_b 35-37; 16 deg size 1.0 x Omega_b 32, 33 x amp 1.0, 1.2
                                                                                                -- 7 sprays.
Output results/plot_data/frame_decomp_edges.json (same fields as sun_dist_bar_grids.py).
Usage: OMP_NUM_THREADS=8 python bin/streams/frame_decomp_edges.py
"""
import itertools, json, os, time
import numpy as np
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, OCEN_OBS
from ocen_dm.streams.frames import FRAMES

B = FRAMES["baumgardt"]
FRAMES["b_vy232"] = dict(B, v_sun=(11.1, 232.24, 7.25)); FRAMES["b_r08122"] = dict(B, r0_kpc=8.122); FRAMES["b_zsun17"] = dict(B, z_sun_pc=17.)
CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"]); W = dict(om=36., an=28., am=1.2, size=1.15)
CFG = [dict(test="F", **W, ic="baumgardt", proj="ibata19"), dict(test="F", **W, ic="ibata19", proj="baumgardt")]
CFG += [dict(test="C", **W, ic=f, proj=f) for f in ("b_vy232", "b_r08122", "b_zsun17")]
CFG += [dict(test="E", om=om, an=28., am=1.6, size=1.3, ic="baumgardt", proj="baumgardt") for om in (35., 36., 37.)]
CFG += [dict(test="E", om=om, an=16., am=am, size=1.0, ic="baumgardt", proj="baumgardt") for om, am in itertools.product((32., 33.), (1.0, 1.2))]
SEG = [(-99, 4), (4, 17), (17, 99)]; edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
outd = ROOT/"results/streams/frame_decomp_edges"; outd.mkdir(parents=True, exist_ok=True); outp = ROOT/"results/plot_data/frame_decomp_edges.json"
res = json.loads(outp.read_text()) if outp.exists() else []; done = {r["tag"] for r in res}
for c in CFG:
    tag = f"{c['test']}_om{c['om']:g}_an{c['an']:g}_am{c['am']:g}_sz{c['size']:g}_ic-{c['ic']}_proj-{c['proj']}"
    if tag in done:
        continue
    t0 = time.time(); f = outd/f"{tag}.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        m = make_spray(c["om"], c["an"], c["am"], 5.6, nrel=32000, seed=1, pm=CAT, host_kw=dict(bar_size=c["size"]), frame=c["ic"], proj_frame=c["proj"])
        np.savez_compressed(f, **m)
    s = score(m); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0); ib = np.digitize(mu[k], edges)-1
    med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else None for j in range(len(cen))] for q in ("pmra", "pmdec")}
    r = dict(c, tag=tag, lnL=s["total"], seg=[float(s["per_star"][(du >= a) & (du < b)].sum()) for a, b in SEG], med=med, phi1=cen.tolist())
    res.append(r); tmp = outp.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); os.replace(tmp, outp)
    print(f"{tag:70s} lnL {s['total']:9.1f}  seg " + " ".join(f"{x:9.1f}" for x in r["seg"]) + f"  [{time.time()-t0:.0f} s]", flush=True)
