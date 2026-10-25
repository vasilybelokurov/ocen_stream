"""Read-only diagnostic: omega Cen's orbit and the bar's resonances in the step-2 best 25- and 12.5-deg hosts."""
import os, sys, numpy as np
ROOT = os.getcwd()
sys.path[:0] = [ROOT+"/bin/streams", ROOT+"/src"]
from ocen_stream.restricted import host_potential, agama_kpc, AGAMA_T_MYR, HUNTER24_DIR
from ocen_stream.mw_variants import axisymmetric_variant
from ocen_stream.frames import to_model, OCEN_OBS, frame_r0
agama = agama_kpc()
T = 1955.58/AGAMA_T_MYR
R0 = frame_r0("gravity21")
MODELS = {
 "25":   dict(angle=25.0, omega=35.411964, amp=1.3842206, size=1.2356600, dist=5.523893, pmra=-3.2616917, pmdec=-6.7332134, vc=223.00973, halo_q=0.9718893),
 "12.5": dict(angle=12.5, omega=33.140546, amp=1.3652125, size=1.3284842, dist=5.526422, pmra=-3.2612456, pmdec=-6.7595030, vc=225.03203, halo_q=0.8960698),
}

# --- AGAMA scale semantics check -------------------------------------------------------
pf = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_baryon_full.ini"))
ps = agama.Potential(potential=pf, scale=[2.0, 1.5])
x = np.array([[3., 1., 0.5], [1., 0.3, 0.2]])
print("scale check: scaled Phi(x) =", ps.potential(x), " 2/1.5*Phi(x/1.5) =", 2/1.5*pf.potential(x/1.5), " 2*Phi(x/1.5) =", 2*pf.potential(x/1.5))

def axi_of(p):
    return axisymmetric_variant(HUNTER24_DIR, halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=R0)[0]

def freqs_axi(pot, R):
    """Omega, kappa, nu [km/s/kpc] for circular orbits at R in the plane."""
    xyz = np.column_stack((R, 0*R, 0*R))
    f, d = pot.forceDeriv(xyz)
    Om = np.sqrt(-f[:, 0]/R)
    kap = np.sqrt(-d[:, 0] + 3*Om**2)       # d2Phi/dx2 = -dFx/dx ; d[:,0] = dFx/dx
    nu = np.sqrt(-d[:, 2])
    return Om, kap, nu

def res_radii(pot, omb):
    R = np.linspace(0.3, 15, 3000); Om, kap, nu = freqs_axi(pot, R)
    out = {}
    for name, fn in [("CR", Om), ("OLR", Om+kap/2), ("ILR", Om-kap/2), ("UHR(4:1)", Om-kap/4), ("vert 2:1", Om-nu/2)]:
        s = np.sign(fn-omb); i = np.where(np.diff(s) != 0)[0]
        out[name] = [round(float(R[j]), 2) for j in i]
    return out

for key, p in MODELS.items():
    print("\n==================== model", key, "deg ====================")
    host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=p["amp"],
                          bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=R0))
    axi = axi_of(p)
    print("resonance radii (axisymmetric part, in-plane):", res_radii(axi, p["omega"]))
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"], "gravity21")[0]
    print("omega Cen today (model frame) x,y,z =", np.round(today[:3], 3), " v =", np.round(today[3:], 1),
          " R =", round(float(np.hypot(*today[:2])), 3), " Lz =", round(float(today[0]*today[4]-today[1]*today[3]), 1))
    # frequencies in the axisymmetric host
    af = agama.ActionFinder(axi)
    J, ang, fr = af(today, angles=True)
    Jr, Jz, Jphi = J; Or, Oz, Op = fr
    print("actions Jr,Jz,Jphi =", np.round(J, 1), " freqs Omega_r,Omega_z,Omega_phi [km/s/kpc] =", np.round(fr, 2))
    print("  T_r = %.1f Myr, T_phi = %.1f Myr" % (2*np.pi/Or*AGAMA_T_MYR, 2*np.pi/abs(Op)*AGAMA_T_MYR))
    print("  (Omega_b - Omega_phi)/Omega_r = %.3f   (Omega_b - Omega_phi)/Omega_z = %.3f   2(Omega_b-Omega_phi)/Omega_r = %.3f"
          % ((p["omega"]-Op)/Or, (p["omega"]-Op)/Oz, 2*(p["omega"]-Op)/Or))
    # orbit backwards 1 Gyr in the barred host
    tb = 1000./AGAMA_T_MYR
    tt, orb = agama.orbit(potential=host, ic=today, timestart=T, time=-tb, trajsize=4001, accuracy=1e-10)
    tt = tt[::-1]; orb = orb[::-1]
    age = (T-tt)*AGAMA_T_MYR
    r = np.linalg.norm(orb[:, :3], axis=1); R = np.hypot(orb[:, 0], orb[:, 1])
    ip = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0]+1
    ia = np.where((r[1:-1] > r[:-2]) & (r[1:-1] > r[2:]))[0]+1
    print("  peri r (kpc):", np.round(r[ip], 2), " apo r:", np.round(r[ia], 2), " |z|max %.2f" % np.abs(orb[:, 2]).max())
    print("  peri ages (Myr):", np.round(age[ip], 0))
    # bar phase relative to omega Cen's azimuth at pericentres (mod 180)
    bar = np.rad2deg(np.deg2rad(p["angle"]) + p["omega"]*(tt-T))
    phi = np.rad2deg(np.arctan2(orb[:, 1], orb[:, 0]))
    rel = ((phi-bar) + 90) % 180 - 90
    print("  bar-relative azimuth of omega Cen at pericentres (deg, mod 180, -90..90):", np.round(rel[ip], 1))
    print("  ... at apocentres:", np.round(rel[ia], 1), " R at apo:", np.round(R[ia], 2), " z at apo:", np.round(orb[ia, 2], 2))
    # Jacobi integral along the orbit (rigid bar -> conserved) and its value vs the L1/L2 effective potential
    E = 0.5*np.sum(orb[:, 3:]**2, axis=1) + host.potential(orb[:, :3], t=tt)
    Lz = orb[:, 0]*orb[:, 4]-orb[:, 1]*orb[:, 3]
    EJ = E - p["omega"]*Lz
    print("  E_J range over 1 Gyr: %.1f .. %.1f (km/s)^2  (sd %.2f); Lz range %.1f .. %.1f" % (EJ.min(), EJ.max(), EJ.std(), Lz.min(), Lz.max()))
    # effective potential saddle (L1/L2) along the bar major axis today, and L4/L5 along the minor axis
    s = np.linspace(3., 10., 1401); a = np.deg2rad(p["angle"])
    for nm, ang in [("L1/L2 (major)", a), ("L4/L5 (minor)", a+np.pi/2)]:
        pts = np.column_stack((s*np.cos(ang), s*np.sin(ang), 0*s))
        Peff = host.potential(pts, t=T) - 0.5*p["omega"]**2*s**2
        j = np.argmax(Peff)
        print("  Phi_eff %s: max %.1f at R=%.2f kpc;  E_J(omega Cen) - Phi_eff_max = %.1f" % (nm, Peff[j], s[j], EJ.mean()-Peff[j]))
    # m=2 (non-axisymmetric) force relative to the axisymmetric force, today's bar, in-plane and at z = 1.4
    nonaxi = agama.Potential(host[1], host[2])
    print("  |F_nonaxi|max/|F_axi| at R = 1.5 2 3 4 5 6 7 kpc:")
    for z in (0.0, 1.4):
        row = []
        for Rr in (1.5, 2, 3, 4, 5, 6, 7):
            ph = np.linspace(0, 2*np.pi, 73)
            pts = np.column_stack((Rr*np.cos(ph), Rr*np.sin(ph), np.full_like(ph, z)))
            fn = np.linalg.norm(nonaxi.force(pts, t=T), axis=1).max(); fa = np.linalg.norm(axi.force(pts), axis=1).mean()
            row.append(round(float(fn/fa), 3))
        print("    z = %.1f:" % z, row)

# --- Same force ratio for the published bars (A = 1, S = 1): Hunter24 and Portail17, in the Hunter axisymmetric host
print("\n==================== published bars, A=1, S=1 ====================")
axi0 = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
for bm in ("hunter24", "portail17"):
    h = host_potential("x", bar_omega=37.5, bar_angle_deg=25., t_today=T, bar_amp=1.0, bar_model=bm)
    nonaxi = agama.Potential(h[1], h[2])
    for z in (0.0, 1.4):
        row = []
        for Rr in (1.5, 2, 3, 4, 5, 6, 7):
            ph = np.linspace(0, 2*np.pi, 73)
            pts = np.column_stack((Rr*np.cos(ph), Rr*np.sin(ph), np.full_like(ph, z)))
            fn = np.linalg.norm(nonaxi.force(pts, t=T), axis=1).max(); fa = np.linalg.norm(axi0.force(pts), axis=1).mean()
            row.append(round(float(fn/fa), 3))
        print("  %s z = %.1f:" % (bm, z), row)
print("resonance radii Hunter axi, Omega_b 37.5:", res_radii(axi0, 37.5), " 35.4:", res_radii(axi0, 35.4), " 33.1:", res_radii(axi0, 33.1))
