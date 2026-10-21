import sys, os, numpy as np
ROOT = os.environ["OCEN_ROOT"]; sys.path[:0] = [ROOT+"/src", ROOT+"/bin/streams"]
from ocen_dm.streams.restricted import host_potential, agama_kpc, AGAMA_T_MYR, HUNTER24_DIR
from ocen_dm.streams.frames import to_model, OCEN_OBS, frame_r0
from ocen_dm.streams.mw_variants import axisymmetric_variant
agama = agama_kpc(); T = 1955.58/AGAMA_T_MYR; r0 = frame_r0("gravity21")
p = dict(angle=12.5, omega=33.1405, amp=1.3652, size=1.3285, dist=5.5264, pmra=-3.26125, pmdec=-6.7595, vc=225.03, q=0.8961)
for lab, A, S in (("fitted", 1.3652, 1.3285), ("published", 1.0, 1.0)):
    host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=A, bar_size=S,
                          axi_variant=dict(halo_q=p["q"], vc_target=p["vc"], r0_kpc=r0))
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"], "gravity21")[0]
    t, orb = agama.orbit(potential=host, ic=today, timestart=T, time=-2000./AGAMA_T_MYR, trajsize=4001, accuracy=1e-10)
    age = (T-t)*AGAMA_T_MYR; Lz = orb[:, 0]*orb[:, 4]-orb[:, 1]*orb[:, 3]; R = np.hypot(orb[:, 0], orb[:, 1])
    # smooth Lz over one radial period (~92 Myr) to show the slow modulation
    k = int(92/0.5); sm = np.convolve(Lz, np.ones(k)/k, mode="same")
    print(f"{lab}: Lz (smoothed over 92 Myr) at ages 50..1950 step 100 Myr:")
    print("  " + " ".join(f"{sm[np.argmin(abs(age-a))]:.0f}" for a in range(50, 2000, 100)))
    print(f"  apocentres (R>6.5 maxima) ages: " + " ".join(f"{age[i]:.0f}" for i in np.where((R[1:-1] > R[:-2]) & (R[1:-1] > R[2:]))[0][:12]+1 if R[np.where((R[1:-1] > R[:-2]) & (R[1:-1] > R[2:]))[0][0]+1] > 0))
    print(f"  R_apo values: " + " ".join(f"{R[i]:.2f}" for i in (np.where((R[1:-1] > R[:-2]) & (R[1:-1] > R[2:]))[0]+1)[:12]))
# Fourier content of the non-axi force on rings (z=0, z=1.5) for fitted vs published, and radial m=2 amplitude at R0
axi = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
full = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_baryon_full.ini"))
bax = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_baryon_axi.ini"))
ring = np.linspace(0, 2*np.pi, 144, endpoint=False)
for lab, A, S in (("fitted 12.5", 1.3652, 1.3285), ("published", 1.0, 1.0)):
    f1 = agama.Potential(potential=full, scale=[A*S, S]); f2 = agama.Potential(potential=bax, scale=[A*S, S])
    for z in (0., 1.5):
        out = []
        for Rr in (4., 5., 6., 7., 8.275):
            P = np.column_stack((Rr*np.cos(ring), Rr*np.sin(ring), np.full_like(ring, z)))
            F = f1.force(P)-f2.force(P); c, s = np.cos(ring), np.sin(ring); FR = F[:, 0]*c+F[:, 1]*s
            nax = np.mean(np.linalg.norm(axi.force(P), axis=1))
            cf = np.fft.rfft(FR-FR.mean())/len(ring)*2
            out.append(f"R{Rr:.1f}: m2 {abs(cf[2])/nax:.3f} m4 {abs(cf[4])/nax:.3f} m6 {abs(cf[6])/nax:.3f}")
        print(f"{lab} z={z}: radial-force Fourier amplitudes / |F_axi|: " + " | ".join(out))
