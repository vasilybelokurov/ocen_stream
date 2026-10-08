#!/usr/bin/env python3
"""Release time of model debris in the knee box (l -45..-28, b 31..41) vs the arm box (l -62..-50, b 20..30): for each
tracer, the last snapshot at which it was within 0.2 kpc of the cluster centre (centre track re-integrated from run.json
start with the run's host). Also the pericentre times of the centre orbit.
Usage: python bin/streams/knee_release_time.py db98_rot A_nodm
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from plot_debris_vs_ibata import observables
from ocen_dm.streams.analysis import load, run_core, run_particles, snapshot_times
from ocen_dm.streams.restricted import AGAMA_T_MYR, OCEN_TODAY, agama_kpc, bound_set, host_potential
from knee_region_check import KNEE, ARM, box

run, key = sys.argv[1], sys.argv[2]
R = ROOT/"results/streams"/run/key
rj = json.loads((R/"run.json").read_text())
ag = agama_kpc(); host = host_potential(rj["mw"])
times = np.asarray(rj["snap_times_myr"])
tc, orb = ag.orbit(potential=host, ic=np.array(rj["start"]), timestart=0., time=times[-1]/AGAMA_T_MYR, trajsize=int(times[-1]/0.5)+1)
tc = tc*AGAMA_T_MYR; r = np.linalg.norm(orb[:, :3], axis=1)
peri = tc[1:-1][(r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])]
cen = np.array([np.interp(times, tc, orb[:, i]) for i in range(3)]).T
m, s = run_particles(R); _, xv = load(R, name="snap_today.npz")
bnd, _ = bound_set(xv, OCEN_TODAY.copy(), m, start=np.linalg.norm(xv[:, :3]-OCEN_TODAY[:3], axis=1) < 0.5, core=run_core(R))
o = observables(xv)
sel = {n: np.where(box(o["l"], o["b"], B) & ~bnd)[0] for n, B in (("knee", KNEE), ("arm", ARM))}
last_in = {n: np.zeros(len(i)) for n, i in sel.items()}
for k, tk in enumerate(times):
    _, x = load(R, tk) if k < len(times)-1 else load(R, name="snap_today.npz")
    for n, i in sel.items():
        d = np.linalg.norm(x[i, :3]-cen[k], axis=1)
        last_in[n] = np.where(d < 0.2, tk, last_in[n])
T = times[-1]
print(f"{run}/{key}: pericentres (Myr ago): {np.round(T-peri[::-1][:12]).astype(int).tolist()}")
for n in sel:
    ago = T-last_in[n]
    print(f"  {n}: N {len(ago)}  release (Myr ago) median {np.median(ago):.0f}, 16-84 {np.percentile(ago,16):.0f}-{np.percentile(ago,84):.0f}, "
          f"fraction released > 1000 Myr ago {np.mean(ago > 1000):.2f}")
