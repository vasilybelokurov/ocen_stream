"""Debug: replay one restricted-N-body update from a snapshot and inspect particles that become unbound."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from ocen_dm.streams.restricted import agama_kpc, bound_set, AGAMA_T_MYR
from ocen_dm.streams.analysis import load, run_particles

run, t0, dt = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
agama = agama_kpc()
mass, sp = run_particles(run)
t, xv = load(run, t0)
b0, pot = bound_set(xv, np.zeros(6), mass)
print("t", t, "bound", b0.sum(), "Phi(0)", pot.potential([0, 0, 0]))
res = agama.orbit(ic=xv, potential=pot, time=dt/AGAMA_T_MYR, trajsize=1, accuracy=float(sys.argv[4]) if len(sys.argv) > 4 else 1e-8, verbose=False)
xv1 = np.vstack(res[:, 1])
E0 = pot.potential(xv[:, :3]) + 0.5*np.sum(xv[:, 3:]**2, 1)
E1 = pot.potential(xv1[:, :3]) + 0.5*np.sum(xv1[:, 3:]**2, 1)
dE = E1 - E0
r0 = np.linalg.norm(xv[:, :3], axis=1)*1e3; r1 = np.linalg.norm(xv1[:, :3], axis=1)*1e3
bad = np.abs(dE) > 0.01*np.abs(E0)
print("particles with |dE/E| > 1% in a STATIC potential:", bad.sum(), "by species", np.bincount(sp[bad], minlength=2))
k = np.argsort(-np.abs(dE))[:10]
for i in k:
    print(f"sp {sp[i]} r0 {r0[i]:.4f} pc r1 {r1[i]:.2f} pc  E0 {E0[i]:.1f} E1 {E1[i]:.1f}  v0 {np.linalg.norm(xv[i,3:]):.1f} v1 {np.linalg.norm(xv1[i,3:]):.1f}")
print("r0 of bad: percentiles", np.percentile(r0[bad], [0, 50, 100]) if bad.any() else None)
