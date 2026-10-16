"""LMC perturbation at omega Cen from the Vasiliev+2021 (Tango) evolving MW+LMC potentials, in Tango units/frame
(kpc, km/s, mass 232500 Msun, G=1; Sun at x=-8, right-handed; today t=0). Perturbation = full evolving potential minus the
initial MW (mw00 halo + bulge + disc): moving LMC + reflex UniformAcceleration + halo deformation (incl. l=1).
Data: Zenodo 10.5281/zenodo.4300977 (Sgr_stream_simulations.zip, sha256 6adcd571...1a37) unpacked in
$OCEN_CATALOGUES/vasiliev21_tango_zenodo4300977. omega Cen orbit integrated in the full evolving potential.
Usage: python bin/streams/lmc_tango_check.py potentials_axisym|potentials_triax   (1 core, ~1 min)"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
import ocen_dm  # noqa: E402,F401  (loads the local .env)
import numpy as np, agama  # noqa: E402
agama.setUnits(mass=232500, length=1, velocity=1)
D = os.path.expandvars("$OCEN_CATALOGUES/vasiliev21_tango_zenodo4300977/") + sys.argv[1]
os.chdir(D)
full = agama.Potential("potential_evolving.ini")
# components (AGAMA reorders them): Evolving halo, DiskAnsatz, UniformAcceleration, Multipole (bulge+disc residual), Shifted Evolving LMC
names = [str(full[i]).split(" (")[0] for i in range(len(full))]; print(names)
halo, acc, lmc = full[names.index("Evolving")], full[names.index("UniformAcceleration")], full[names.index("Shifted Evolving")]
bary = agama.Potential(full[names.index("DiskAnsatz")], full[names.index("Multipole")])
halo0 = agama.Potential(file="mw00.pot")
mw0 = agama.Potential(halo0, bary)
for r in (8, 50, 100):
    print(f"{sys.argv[1]}: initial MW r {r} kpc vc {np.sqrt(-r*mw0.force([r,0,0])[0]):.1f}; M(<r) {-r*r*mw0.force([r,0,0])[0]*232500:.3e} Msun")
oc = np.array([-4.87, -4.07, 1.40, 95.98, -21.69, -86.82])   # project OCEN_TODAY with x -> -x (Sun at -x)
T = 1.0/0.9778
tt, orb = agama.orbit(potential=full, ic=oc, timestart=0, time=-T, trajsize=2001, accuracy=1e-10)
f_full = np.array([full.force(orb[i, :3], t=tt[i]) for i in range(len(tt))])
f_mw0 = mw0.force(orb[:, :3])
f_lmcacc = np.array([lmc.force(orb[i, :3], t=tt[i]) + acc.force(orb[i, :3], t=tt[i]) for i in range(len(tt))])
f_halo = np.array([halo.force(orb[i, :3], t=tt[i]) for i in range(len(tt))]) - halo0.force(orb[:, :3])
for name, a in (("total perturbation", f_full - f_mw0), ("LMC + reflex", f_lmcacc), ("halo deformation", f_halo)):
    for lab, m in (("250 Myr", tt > -0.25/0.9778), ("1 Gyr", tt > -9)):
        print(f"  {name:20s} {lab:8s}: |int a dt| {np.linalg.norm(np.trapezoid(a[m], tt[m], axis=0)):.2f} km/s; "
              f"median |a|/|a_MW| {np.median(np.linalg.norm(a[m],axis=1)/np.linalg.norm(f_mw0[m],axis=1)):.1e}; today {np.linalg.norm(a[0])/np.linalg.norm(f_mw0[0]):.1e}")
traj = np.loadtxt("trajlmc.txt"); d = np.linalg.norm(traj[:, 1:4], axis=1); i = np.argmin(np.where(traj[:, 0] > -1, d, 1e9))
print(f"  LMC today at {d[np.argmin(abs(traj[:,0]))]:.1f} kpc; pericentre {d[i]:.1f} kpc at t = {traj[i,0]*977.8:.0f} Myr")
# decomposition: perturbation at the origin (uniform part), tidal part, and the halo change's monopole (spherical average)
pert = lambda x, t: full.force(x, t=t) - mw0.force(x)
a0 = np.array([pert(np.zeros(3), t) for t in tt]); ax = f_full - f_mw0
rng = np.random.default_rng(1); u = rng.normal(size=(300, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
def mono(i):
    r = np.linalg.norm(orb[i, :3]); pts = r*u
    dfr = np.sum((halo.force(pts, t=tt[i]) - halo0.force(pts))*u, axis=1).mean()
    return dfr*orb[i, :3]/r
sel = np.arange(0, len(tt), 10); am = np.array([mono(i) for i in sel])
for lab, m in (("250 Myr", tt > -0.25/0.9778), ("1 Gyr", tt > -9)):
    ms = m[sel]
    print(f"  {lab}: |int a_pert(0) dt| {np.linalg.norm(np.trapezoid(a0[m], tt[m], axis=0)):.2f}; tidal |int (a(x)-a(0)) dt| "
          f"{np.linalg.norm(np.trapezoid(ax[m]-a0[m], tt[m], axis=0)):.2f}; halo monopole |int dt| {np.linalg.norm(np.trapezoid(am[ms], tt[sel][ms], axis=0)):.2f}; "
          f"tidal minus monopole {np.linalg.norm(np.trapezoid((ax-a0)[sel][ms]-am[ms], tt[sel][ms], axis=0)):.2f} km/s")
print(f"  |a_pert(0)| today {np.linalg.norm(a0[0]):.1f}, median {np.median(np.linalg.norm(a0,axis=1)):.1f}; |halo monopole| median {np.median(np.linalg.norm(am,axis=1)):.1f}; "
      f"|tidal - mono| median {np.median(np.linalg.norm((ax-a0)[sel]-am,axis=1)):.1f} (km/s)^2/kpc; |a_MW| median {np.median(np.linalg.norm(f_mw0,axis=1)):.0f}")
aL = np.array([lmc.force(orb[i, :3], t=tt[i]) - lmc.force(np.zeros(3), t=tt[i]) for i in range(len(tt))])   # LMC tidal only
for lab, m in (("250 Myr", tt > -0.25/0.9778), ("1 Gyr", tt > -9)):
    print(f"  LMC-only tidal {lab}: |int dt| {np.linalg.norm(np.trapezoid(aL[m], tt[m], axis=0)):.2f} km/s; max |a| {np.linalg.norm(aL[m],axis=1).max():.1f}")
