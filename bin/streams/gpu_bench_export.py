#!/usr/bin/env python3
"""Export the inputs of the GPU integrator benchmark (2026-10-25): host tables, progenitor, particle initial conditions, references.

Point: the best free-fit spray final_c0_2 (fit_knot_nm.py chain c0; its seed and epoch offset), so the JAX sprays can be compared
particle by particle with the stored AGAMA spray and scored with stream_model_compare.
- Host: the full time-dependent host (axisymmetric part + A x [bar(x/S) - axisymmetrised bar(x/S)], rotating at Omega_b) evaluated
  at t = T on an (R, phi', z) grid in the bar frame (phi' = phi - theta(T)), Fourier-decomposed in phi'; only the cos(m phi') terms
  with even m <= 8 are kept (the others are checked to be negligible). At time t: Phi = sum_m Phi_m(R, z) cos(m (phi - theta(t))),
  theta(t) = theta_T + Omega_b (t - T). Grid uniform in u = asinh(R / 0.5 kpc) and w = asinh(z / 0.2 kpc) (Phi is even in R and z,
  so a mirror boundary at 0 is exact); cubic B-spline coefficients (scipy.ndimage.spline_filter, mode 'mirror').
- Progenitor: Phi_sat(r) of the frozen A_nodm profile (restricted.FrozenCore) on a grid uniform in ln r, B-spline coefficients;
  its centre orbit (t, x, v) every 0.05 Myr, AGAMA 1e-12 (as spray.spray_unwrapped).
- Particles: release times and initial conditions of the trailing particles < 700 Myr of final_c0_2 (captured from
  grid4_common.make_spray), the AGAMA 1e-8 final states (== the stored spray) and AGAMA 1e-10 final states; chi and age for the
  stream_model_compare selection.
Usage: PYTHONPATH=src python bin/streams/gpu_bench_export.py -> results/gpu/bench_inputs.npz
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import spline_filter
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
from grid4_common import make_spray, T, r_kpc, M
import ocen_stream.spray as SP
from ocen_stream.restricted import host_potential, agama_kpc, satellite_potential, FrozenCore, AGAMA_T_MYR
from ocen_stream.frames import frame_r0

RUN = json.loads((ROOT/"results/streams/fitknot_20261025/runs/final_c0_2.json").read_text()); P = RUN["params"]
NU, NW, NPHI, MMAX = 256, 128, 32, 8                                 # grid nodes in u, w, phi'; highest Fourier mode kept
RS, ZS, RMAX, ZMAX = 0.5, 0.2, 30., 10.                              # asinh scales and extent [kpc]
OUT = ROOT/"results/gpu/bench_inputs.npz"


def main():
    agama = agama_kpc(); t1 = time.time()
    host_kw = dict(bar_size=P["size"], axi_variant=dict(halo_q=P["halo_q"], vc_target=P["vc"], r0_kpc=frame_r0("gravity21")))
    host = host_potential("x", bar_omega=P["omega"], bar_angle_deg=P["angle"], t_today=T, bar_amp=P["amp"], **host_kw)
    theta_T = np.deg2rad(P["angle"])
    # host tables
    u = np.linspace(0., np.arcsinh(RMAX/RS), NU); w = np.linspace(0., np.arcsinh(ZMAX/ZS), NW); ph = 2*np.pi*np.arange(NPHI)/NPHI
    R, Z, PH = np.meshgrid(RS*np.sinh(u), ZS*np.sinh(w), ph, indexing="ij")
    X = np.column_stack(((R*np.cos(PH+theta_T)).ravel(), (R*np.sin(PH+theta_T)).ravel(), Z.ravel()))
    phi = host.potential(X, t=T).reshape(R.shape); F = np.fft.rfft(phi, axis=2)/NPHI
    modes = np.stack([F[..., 0].real]+[2*F[..., m].real for m in range(2, MMAX+1, 2)])           # cos(m phi'), even m
    drop = max(np.abs(F[..., m]).max() for m in range(1, MMAX+1, 2))+max(np.abs(F[..., m].imag).max() for m in range(2, MMAX+1, 2))
    zsym = np.abs(host.potential(X*[1, 1, -1], t=T).reshape(R.shape)-phi).max()
    print(f"host grid {R.shape}: |Phi| max {np.abs(phi).max():.3e}; dropped odd/sin modes max {drop:.2e}; z-asymmetry max {zsym:.2e}; "
          f"highest kept mode m={MMAX} amp {np.abs(modes[-1]).max():.2e}, aliasing check m=10..16 {np.abs(F[..., 10:]).max():.2e}", flush=True)
    coef = np.stack([spline_filter(m, order=3, mode="mirror") for m in modes])
    # progenitor potential and centre orbit (as spray.spray_unwrapped)
    sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=FrozenCore(r_kpc, M))
    lr = np.linspace(np.log(1e-5), np.log(50.), 2048); rr = np.exp(lr)
    sat_phi = sat.potential(np.column_stack((rr, 0*rr, 0*rr))); sat_coef = spline_filter(sat_phi, order=3, mode="nearest")
    sat_mtot = -sat_phi[-1]*rr[-1]/agama.G
    # particles: capture the particle call of spray_unwrapped (ic, t0) and keep AGAMA 1e-8's final states
    cap = {}; real = agama.orbit

    def spy(potential, ic, timestart, time, trajsize, accuracy, **kw):
        if np.ndim(timestart) == 0:
            if trajsize > 2:
                cap["tc"], cap["orb"] = res = real(potential=potential, ic=ic, timestart=timestart, time=time, trajsize=trajsize, accuracy=accuracy, **kw)
                return res
            return real(potential=potential, ic=ic, timestart=timestart, time=time, trajsize=trajsize, accuracy=accuracy, **kw)
        cap.setdefault("ic", []).append(np.array(ic)); cap.setdefault("t0", []).append(np.array(timestart)); cap["total"] = potential
        res = real(potential=potential, ic=ic, timestart=timestart, time=time, trajsize=trajsize, accuracy=accuracy, **kw)
        cap.setdefault("xv8", []).append(np.vstack([r[-1] for r in res[:, 1]])); return res
    agama.orbit = spy
    try:
        m = make_spray(P["omega"], P["angle"], P["amp"], P["dist"], nrel=RUN["nrel"], seed=RUN["seed"], pm=(P["pmra"], P["pmdec"]),
                       host_kw=host_kw, max_age_myr=700., epoch_offset=RUN["epoch_offset"], frame="gravity21")
    finally:
        agama.orbit = real
    st = np.load(ROOT/"results/streams/fitknot_20261025/runs/final_c0_2.npz")
    print("spray reproduces final_c0_2.npz:", all(np.array_equal(m[k], st[k]) for k in ("l", "b", "pmra", "pmdec", "chi", "age")), flush=True)
    ic, t0, xv8 = (np.vstack(cap[k]) if k != "t0" else np.concatenate(cap[k]) for k in ("ic", "t0", "xv8"))
    xv10 = np.vstack(real(potential=cap["total"], ic=ic, timestart=t0, time=T-t0, trajsize=1, accuracy=1e-10, verbose=False)[:, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, host_coef=coef, host_m=np.arange(0, MMAX+1, 2), u0=u[0], du=u[1]-u[0], w0=w[0], dw=w[1]-w[0], rs=RS, zs=ZS,
             theta_T=theta_T, omega_b=P["omega"], T=T, t_unit_myr=AGAMA_T_MYR,
             sat_coef=sat_coef, lr0=lr[0], dlr=lr[1]-lr[0], sat_mtot=sat_mtot, G=agama.G,
             tc=cap["tc"], orb=cap["orb"], ic=ic, t0=t0, xv_agama8=xv8, xv_agama10=xv10, chi=np.asarray(m["chi"]), age=np.asarray(m["age"]),
             params=json.dumps(P), source="results/streams/fitknot_20261025/runs/final_c0_2.npz")
    print(f"{OUT.relative_to(ROOT)}: {ic.shape[0]} particles, {OUT.stat().st_size/1e6:.1f} MB, {time.time()-t1:.0f} s")


if __name__ == "__main__":
    main()
