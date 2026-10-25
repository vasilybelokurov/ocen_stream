#!/usr/bin/env python3
"""Profile-likelihood stage 1 (docs/reviews/codex_profile_likelihood_plan_2026-10-10.md): broad space-filling search at fixed bar angle.
For each angle (28, 16 deg): 40 scrambled-Sobol points in
  Omega_b 30-40 km/s/kpc, amp 0.8-1.6, bar size 0.9-1.4, omega Cen d 5.3-5.9 kpc, pmra/pmdec catalogue +- 3 sigma (0.027 mas/yr),
subject to the bar ending inside corotation: size x 5.0 kpc <= R_CR(Omega_b) of the axisymmetric host (R_bar0 = 5.0 kpc is the
working assumption used since Phase 2b, UNVERIFIED against Hunter+2024); rejected draws are replaced by further Sobol points.
The SAME 40 nuisance points are used at both angles (paired design). Each point: 1x particles (8000 epochs), seeds 1-3, baumgardt
frame; lnL = mean over seeds. Sprays saved (results/streams/profile_s1/) for re-scoring against mocks (stage 4).
Output results/plot_data/profile_stage1.json (resumable). Usage: OMP_NUM_THREADS=8 python bin/streams/profile_stage1.py
"""
import json, os, time
import numpy as np
from scipy.stats import qmc
from grid4_common import ROOT, make_spray, score, du, OCEN_OBS
from ocen_stream.restricted import agama_kpc, HUNTER24_DIR

SIG = 0.027; RBAR0 = 5.0; ANGLES = (28., 16.); SEEDS = (1, 2, 3); NPT = 40
LO = np.array([30., 0.8, 0.9, 5.3, OCEN_OBS["pmra"]-3*SIG, OCEN_OBS["pmdec"]-3*SIG])
HI = np.array([40., 1.6, 1.4, 5.9, OCEN_OBS["pmra"]+3*SIG, OCEN_OBS["pmdec"]+3*SIG])
NAMES = ("om", "am", "size", "dist", "pmra", "pmdec")
agama = agama_kpc(); _axi = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
_R = np.linspace(2., 12., 2001); _OM = np.sqrt(-_R*_axi.force(np.column_stack((_R, 0*_R, 0*_R)))[:, 0])/_R   # decreasing in R
r_cr = lambda om: float(np.interp(-om, -_OM, _R))

pts, sob = [], qmc.Sobol(6, scramble=True, seed=20261010)
while len(pts) < NPT:
    x = LO+sob.random(1)[0]*(HI-LO)
    if x[2]*RBAR0 <= r_cr(x[0]):
        pts.append(x)
SEG = [(-99, 4), (4, 17), (17, 99)]
outd = ROOT/"results/streams/profile_s1"; outd.mkdir(parents=True, exist_ok=True); outp = ROOT/"results/plot_data/profile_stage1.json"
res = json.loads(outp.read_text()) if outp.exists() else []; done = {r["tag"] for r in res}
for i, x in enumerate(pts):
    for an in ANGLES:
        tag = f"an{an:g}_p{i:02d}"
        if tag in done:
            continue
        t0 = time.time(); lnl, seg = [], []
        for s in SEEDS:
            f = outd/f"{tag}_s{s}.npz"
            if f.exists():
                m = dict(np.load(f))
            else:
                m = make_spray(x[0], an, x[1], x[3], nrel=8000, seed=s, pm=(x[4], x[5]), host_kw=dict(bar_size=x[2])); np.savez_compressed(f, **m)
            sc = score(m); lnl.append(sc["total"]); seg.append([float(sc["per_star"][(du >= a) & (du < b)].sum()) for a, b in SEG])
        r = dict(tag=tag, an=an, point=i, **{k: float(v) for k, v in zip(NAMES, x)}, rcr_over_rbar=r_cr(x[0])/(x[2]*RBAR0),
                 lnL_seeds=lnl, lnL=float(np.mean(lnl)), seg=np.mean(seg, 0).tolist())
        res.append(r); tmp = outp.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); os.replace(tmp, outp)
        print(f"{tag} " + " ".join(f"{k} {v:.4g}" for k, v in zip(NAMES, x)) + f"  lnL {r['lnL']:9.1f} (sd {np.std(lnl, ddof=1):5.0f})  [{time.time()-t0:.0f} s]", flush=True)
