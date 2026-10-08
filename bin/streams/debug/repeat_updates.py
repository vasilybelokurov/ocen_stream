"""Debug: repeat restricted updates from one snapshot, with and without the time-dependent centre wrapper,
and count energy-violating particles per update (|dE/E| > 1% in the potential held fixed in that update)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from ocen_dm.streams.restricted import agama_kpc, bound_set, AGAMA_T_MYR
from ocen_dm.streams.analysis import load, run_particles

run, t0, n, mode = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
agama = agama_kpc()
mass, sp = run_particles(run)
t, xv = load(run, t0)
_, pot = bound_set(xv, np.zeros(6), mass)
dt = 2.0/AGAMA_T_MYR
for k in range(n):
    ts = t/AGAMA_T_MYR + k*dt
    if mode == "centre":
        tc = np.linspace(ts, ts+dt, 41)
        P = agama.Potential(potential=pot, center=np.column_stack((tc, np.zeros((41, 6)))))
    else:
        P = pot
    res = agama.orbit(ic=xv, potential=P, timestart=ts, time=dt, trajsize=1, accuracy=1e-8, verbose=False)
    x1 = np.vstack(res[:, 1])
    E0 = pot.potential(xv[:, :3]) + 0.5*np.sum(xv[:, 3:]**2, 1)
    E1 = pot.potential(x1[:, :3]) + 0.5*np.sum(x1[:, 3:]**2, 1)
    bad = np.abs(E1-E0) > 0.01*np.abs(E0)
    print(mode, k, "bad", int(bad.sum()), np.bincount(sp[bad], minlength=2).tolist(), "max |dE|", float(np.max(np.abs(E1-E0))), flush=True)
    xv = x1
