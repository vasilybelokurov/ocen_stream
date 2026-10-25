#!/usr/bin/env python3
"""Where along omega Cen's orbit the bar torques it: published vs strong bar (2026-10-23; numbers in the note's Sec. 4.5).

Orbit: the best step-2 strong-bar fit (12.5 deg; results/streams/profile_step2_20261020/runs/final_a12.5_3.json), last 700 Myr.
Reported: time fractions at R > 3, 4, 5 kpc; torque tau_z = x F_y - y F_x of the bar term (published A = S = 1, fitted A, S) in
units of R |F_axi| (rms, mean) and the share of its mean square from R > 4 kpc; the same for the extra torque (fitted minus
published); L_z range over 700 Myr integrated in each host from the same initial conditions.
Usage: PYTHONPATH=src python bin/streams/diagnostics/orbit_torque.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]


def main():
    from ocen_stream.restricted import host_potential, agama_kpc, AGAMA_T_MYR
    from ocen_stream.frames import to_model, OCEN_OBS, frame_r0
    agama = agama_kpc(); T = 1955.58/AGAMA_T_MYR; r0 = frame_r0("gravity21")
    p = json.loads((ROOT/"results/streams/profile_step2_20261020/runs/final_a12.5_3.json").read_text())["params"]
    host = lambda A, S: host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=A, bar_size=S,
                                       axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=r0))
    H, H0, Ha = host(p["amp"], p["size"]), host(1., 1.), host(0., 1.)
    ic = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"], "gravity21")[0]
    t, o = agama.orbit(potential=H, ic=ic, timestart=T, time=-700./AGAMA_T_MYR, trajsize=7001, accuracy=1e-10)
    x = o[:, :3]; R = np.hypot(x[:, 0], x[:, 1])
    print(f"R {R.min():.2f}-{R.max():.2f} kpc; time fraction R>3/4/5: " + " ".join(f"{np.mean(R > c):.2f}" for c in (3, 4, 5)))
    Fa = np.linalg.norm(Ha.force(x), axis=1)
    tau = lambda h: (lambda F: x[:, 0]*F[:, 1]-x[:, 1]*F[:, 0])(np.array([h.force(xi, t=ti)-Ha.force(xi) for xi, ti in zip(x, t)]))
    tf, tp = tau(H), tau(H0)
    for lab, q in (("strong", tf), ("published", tp), ("extra (strong - published)", tf-tp)):
        u = q/(R*Fa)
        print(f"{lab:28s} rms {np.sqrt(np.mean(u**2)):.4f}  mean {np.mean(u):+.4f}  share of mean square from R>4: {np.sum(q[R > 4]**2)/np.sum(q**2):.2f}")
    for lab, h in (("strong", H), ("published", H0)):
        _, o2 = agama.orbit(potential=h, ic=ic, timestart=T, time=-700./AGAMA_T_MYR, trajsize=3001, accuracy=1e-10)
        lz = o2[:, 0]*o2[:, 4]-o2[:, 1]*o2[:, 3]; print(f"L_z range ({lab}) over 700 Myr: {lz.max()-lz.min():.0f} kpc km/s")


if __name__ == "__main__":
    main()
