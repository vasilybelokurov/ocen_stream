"""Non-axisymmetric force along omega Cen's orbit: fitted vs published bar (read-only check, 1 core)."""
import sys, os, numpy as np
ROOT = os.environ["OCEN_ROOT"]; sys.path[:0] = [ROOT+"/src", ROOT+"/bin/streams"]
from ocen_stream.restricted import host_potential, agama_kpc, AGAMA_T_MYR, HUNTER24_DIR
from ocen_stream.frames import to_model, OCEN_OBS, frame_r0
from ocen_stream.mw_variants import axisymmetric_variant
agama = agama_kpc()
T = 1955.58/AGAMA_T_MYR
r0 = frame_r0("gravity21")
cases = {
 "12.5 fitted": dict(angle=12.5, omega=33.1405, amp=1.3652, size=1.3285, dist=5.5264, pmra=-3.26125, pmdec=-6.7595, vc=225.03, q=0.8961),
 "12.5 A=S=1":  dict(angle=12.5, omega=33.1405, amp=1.0, size=1.0, dist=5.5264, pmra=-3.26125, pmdec=-6.7595, vc=225.03, q=0.8961),
 "25 fitted":   dict(angle=25.0, omega=35.412, amp=1.3842, size=1.2357, dist=5.5239, pmra=-3.26169, pmdec=-6.7332, vc=223.01, q=0.9719),
 "25 A=S=1":    dict(angle=25.0, omega=35.412, amp=1.0, size=1.0, dist=5.5239, pmra=-3.26169, pmdec=-6.7332, vc=223.01, q=0.9719),
}
for name, p in cases.items():
    axi = axisymmetric_variant(HUNTER24_DIR, halo_q=p["q"], vc_target=p["vc"], r0_kpc=r0)[0]
    host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=p["amp"],
                          bar_size=p["size"], axi_variant=dict(halo_q=p["q"], vc_target=p["vc"], r0_kpc=r0))
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"], "gravity21")[0]
    n = 1401
    t, orb = agama.orbit(potential=host, ic=today, timestart=T, time=-700./AGAMA_T_MYR, trajsize=n, accuracy=1e-10)
    x = orb[:, :3]; R = np.hypot(x[:, 0], x[:, 1]); phi = np.arctan2(x[:, 1], x[:, 0])
    Fna = np.array([host.force(x[i], t=t[i]) for i in range(n)]) - axi.force(x)
    Fax = axi.force(x); nax = np.linalg.norm(Fax, axis=1)
    c, s = np.cos(phi), np.sin(phi)
    FR = (Fna[:, 0]*c+Fna[:, 1]*s)/nax; Fp = (-Fna[:, 0]*s+Fna[:, 1]*c)/nax; Fz = Fna[:, 2]/nax
    # radial component of F_axi is inward (negative): FR > 0 = outward = weaker binding
    age = (T-t)*AGAMA_T_MYR
    # bar angle at each time; azimuth of cluster relative to bar major axis
    barang = np.deg2rad(p["angle"]) + p["omega"]*(t-T)
    rel = np.rad2deg((phi-barang+np.pi/2) % np.pi - np.pi/2)
    print(f"\n{name}: today R={R[0]:.2f} z={x[0,2]:.2f} rel. azimuth to bar {rel[0]:.0f} deg; "
          f"peri/apo {R.min():.2f}/{R.max():.2f}; today F_na/|F_axi| R,phi,z = {FR[0]:+.3f} {Fp[0]:+.3f} {Fz[0]:+.3f} (|.|={np.hypot(np.hypot(FR[0],Fp[0]),Fz[0]):.3f})")
    for amax in (100, 300, 700):
        m = age <= amax
        print(f"  mean over last {amax:3d} Myr: F_R {FR[m].mean():+.4f}  F_phi {Fp[m].mean():+.4f}  F_z {Fz[m].mean():+.4f} | rms {np.sqrt((FR[m]**2+Fp[m]**2+Fz[m]**2).mean()):.3f}"
              f" | R>4: F_R {FR[m&(R>4)].mean():+.4f} F_phi {Fp[m&(R>4)].mean():+.4f} F_z {Fz[m&(R>4)].mean():+.4f}")
    # effective torque: mean F_phi*R (km/s)^2 ... in units of  |F_axi| R
    Lz = x[:, 0]*orb[:, 4]-x[:, 1]*orb[:, 3]
    print(f"  Lz today {Lz[0]:+.0f}, 300 Myr ago {Lz[np.argmin(abs(age-300))]:+.0f}, 700 Myr ago {Lz[-1]:+.0f} kpc km/s; E today {host.potential(x[0], t=t[0])+0.5*(orb[0,3:]**2).sum():.0f}")
