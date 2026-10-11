#!/usr/bin/env python3
"""Likelihood sweep over the 3D bar space: angle x pattern speed x bar size (test run: 1x particles, seed 1).
Grid: angle 16-36 deg (step 2), Omega_b 30-40 km/s/kpc (step 1), size 0.9-1.5 (step 0.1); points with the bar end beyond
corotation (size x 5.0 kpc > R_CR(Omega_b) of the axisymmetric Hunter+2024 host; R_bar0 = 5.0 kpc UNVERIFIED, as in
profile_stage1.py; --rbar0 4.25 = the model's A2 half-peak length) are skipped. Fixed: amp 1.2, d 5.6 kpc, catalogue PMs, baumgardt frame, 8000 epochs (1x), seed 1, debris
< 700 Myr (make_spray max_age_myr=700; identical lnL to the full spray). No sprays saved; per point: lnL, phi1-segment lnL,
footprint PM medians per 2-deg phi1 bin.
Parallel: independent worker processes, each takes every n-th point and appends JSON lines to its own file
(results/plot_data/bar3d_sweep/w{i}.jsonl; resumable).
Usage: bin/streams/run_bar3d_sweep.sh   (7 workers x OMP_NUM_THREADS=2)
       python bin/streams/bar3d_sweep.py --worker i --nworkers n [--nrel 8000 --seed 1 --amp 1.2]
"""
import argparse, itertools, json, os, time
import numpy as np
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, OCEN_OBS
from ocen_dm.streams.restricted import agama_kpc, HUNTER24_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--rbar0", type=float, default=5.0); ap.add_argument("--nrel", type=int, default=8000); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--amp", type=float, default=1.2)
a = ap.parse_args()

RBAR0 = a.rbar0   # kpc, bar half-length at size 1 (5.0 star counts, Wegg+2015; 4.25 = A2 half-peak of the model)
agama = agama_kpc(); _axi = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
_R = np.linspace(2., 12., 2001); _OM = np.sqrt(-_R*_axi.force(np.column_stack((_R, 0*_R, 0*_R)))[:, 0])/_R
r_cr = lambda om: float(np.interp(-om, -_OM, _R))
ANG = np.arange(16., 36.1, 2.); OM = np.arange(30., 40.1, 1.); SIZE = np.round(np.arange(0.9, 1.51, 0.1), 2)
PTS = [(an, om, sz) for an, om, sz in itertools.product(ANG, OM, SIZE) if sz*RBAR0 <= r_cr(om)]
CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"]); SEG = [(-99, 4), (4, 17), (17, 99)]
edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
outd = ROOT/"results/plot_data/bar3d_sweep"; outd.mkdir(parents=True, exist_ok=True)
outp = outd/f"w{a.worker}.jsonl"
done = {json.loads(l)["tag"] for f in outd.glob("w*.jsonl") for l in f.read_text().splitlines()}   # all workers (point split may change)
if a.worker == 0:
    print(f"{len(PTS)} points (of {len(ANG)*len(OM)*len(SIZE)}) after the corotation cut", flush=True)
for an, om, sz in PTS[a.worker::a.nworkers]:
    tag = f"an{an:g}_om{om:g}_sz{sz:g}_am{a.amp:g}_n{a.nrel}_s{a.seed}"
    if tag in done:
        continue
    t0 = time.time()
    m = make_spray(om, an, a.amp, 5.6, nrel=a.nrel, seed=a.seed, pm=CAT, host_kw=dict(bar_size=sz), max_age_myr=700.)
    s = score(m); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0)
    ib = np.digitize(mu[k], edges)-1
    med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else None for j in range(len(cen))] for q in ("pmra", "pmdec")}
    r = dict(tag=tag, an=float(an), om=float(om), size=float(sz), am=a.amp, nrel=a.nrel, seed=a.seed, rcr_over_rbar=r_cr(om)/(sz*RBAR0),
             lnL=float(s["total"]), seg=[float(s["per_star"][(du >= x0) & (du < x1)].sum()) for x0, x1 in SEG], med=med, phi1=cen.tolist())
    with open(outp, "a") as fh:
        fh.write(json.dumps(r)+"\n")
    print(f"w{a.worker} {tag:40s} lnL {r['lnL']:9.1f}  [{time.time()-t0:.1f} s]", flush=True)
