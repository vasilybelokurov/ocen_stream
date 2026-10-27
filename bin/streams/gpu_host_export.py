#!/usr/bin/env python3
"""Export parameter-independent host tables for the JAX spray (GPU pipeline stage 1, 2026-10-25).

The host of restricted.host_potential(bar_omega, bar_angle_deg, t_today, bar_amp=A, bar_size=S, axi_variant=dict(halo_q=q,
vc_target=v_c, r0_kpc=R0)) is
    Phi = Phi_b(R, z) + c(q, v_c) Phi_h(R, z; q) + A sum_m B_m(R/S, z/S) cos(m (phi - theta(t))),  theta(t) = angle + Omega_b (t - T)
with Phi_b the axisymmetrised Hunter+2024 baryons, Phi_h the unit-normalisation Einasto halo with density flattening q
(mw_variants.PARAMS_DARK, Multipole lmax 12), B_m the cos(m phi) modes (bar along x) of [baryon_full - baryon_axi], and
c = (v_c^2 - v_b^2(R0)) / v_h^2(R0; q) (mw_variants.axisymmetric_variant). Tables: grid uniform in u = asinh(R / 0.5) and
w = asinh(z / 0.2) (as gpu_bench_export.py), cubic B-spline coefficients (mirror at 0); halo tables on a q stack with not-a-knot
B-spline coefficients along q (and for v_h^2(R0; q)), so a model's halo is a fixed linear combination of 4 stack tables.
Grid: R <= 250, |z| <= 250 kpc (2026-10-27): the kernel stops particles beyond 100 kpc, well inside (the mirror boundary of the
spline prefilter spoils the outer ~20% of the range: 10% force errors at the old 30 kpc edge).
Usage: PYTHONPATH=src python bin/streams/gpu_host_export.py [OUT] -> results/gpu/host_tables.npz
"""
import os, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import spline_filter
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
from ocen_stream.restricted import agama_kpc, HUNTER24_DIR
from ocen_stream.mw_variants import PARAMS_DARK, MUL, _vc2
from ocen_stream.frames import frame_r0

NPHI, MMAX = 32, 8
RS, ZS, RMAX, ZMAX = 0.5, 0.2, 250., 250.   # 2026-10-27: was 30, 10 kpc (256 x 128 nodes); escapers left the tables and the
                                            # extrapolated spline flung them to 1e4 km/s (docs/reviews/trapped_particles_fable_codex_20261027.md)
DU, DW = np.arcsinh(30./RS)/255, np.arcsinh(10./ZS)/127      # node spacing of the 2026-10-25 tables (kept; old nodes are a subset)
NU, NW = int(np.ceil(np.arcsinh(RMAX/RS)/DU))+1, int(np.ceil(np.arcsinh(ZMAX/ZS)/DW))+1
QS = np.round(np.arange(0.55, 1.3501, 0.05), 3)                      # usable q 0.65-1.25 (two nodes of margin)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results/gpu/host_tables.npz"


def bcoef_nak(f, axis=0):
    """cubic B-spline coefficients c_{-1..n} (n+2 along axis) of the not-a-knot interpolating spline of samples f on a uniform grid:
    (c_{i-1} + 4 c_i + c_{i+1}) / 6 = f_i, plus continuous third derivatives at the second and second-last nodes. Unlike a mirror
    boundary this is exact for non-even functions (the halo is monotonic in q)."""
    f = np.moveaxis(f, axis, 0); n = f.shape[0]; A = np.zeros((n+2, n+2))
    for i in range(n):
        A[i, i:i+3] = (1/6, 4/6, 1/6)
    A[n, :5] = (1, -4, 6, -4, 1); A[n+1, -5:] = (1, -4, 6, -4, 1)
    c = np.linalg.solve(A, np.concatenate((f.reshape(n, -1), np.zeros((2, f[0].size)))))
    return np.moveaxis(c.reshape((n+2,)+f.shape[1:]), 0, axis)


def main():
    agama = agama_kpc(); t1 = time.time(); r0 = frame_r0("gravity21")
    u = DU*np.arange(NU); w = DW*np.arange(NW)
    R, Z = np.meshgrid(RS*np.sinh(u), ZS*np.sinh(w), indexing="ij"); XZ = np.column_stack((R.ravel(), 0*R.ravel(), Z.ravel()))
    filt = lambda a: spline_filter(a, order=3, mode="mirror")
    bary = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_baryon_axi.ini"))
    phib = bary.potential(XZ).reshape(R.shape); vb2 = _vc2(bary, r0)
    # bar term modes in the file frame (bar along x)
    full = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_baryon_full.ini"))
    ph = 2*np.pi*np.arange(NPHI)/NPHI; R3, Z3, P3 = (a[..., None]*np.ones(NPHI) for a in (R, Z, np.zeros_like(R))); P3 = P3+ph
    X3 = np.column_stack(((R3*np.cos(P3)).ravel(), (R3*np.sin(P3)).ravel(), Z3.ravel()))
    d = (full.potential(X3)-bary.potential(X3)).reshape(R3.shape); F = np.fft.rfft(d, axis=2)/NPHI
    bar = np.stack([F[..., 0].real]+[2*F[..., m].real for m in range(2, MMAX+1, 2)])
    drop = max(np.abs(F[..., m]).max() for m in range(1, MMAX+1, 2))+max(np.abs(F[..., m].imag).max() for m in range(2, MMAX+1, 2))
    print(f"bar term: modes m=0..{MMAX} even, |B| max {np.abs(bar).max():.3e}; dropped odd/sin {drop:.1e}; m>=10 {np.abs(F[..., 10:]).max():.1e}", flush=True)
    # halo stack (unit normalisation), its v_c^2(R0)
    halo, vh2 = [], []
    for q in QS:
        h = agama.Potential(density=agama.Density(**dict(PARAMS_DARK, axisRatioZ=q)), **dict(MUL, symmetry="Axisymmetric"))
        halo.append(h.potential(XZ).reshape(R.shape)); vh2.append(_vc2(h, r0))
    halo = np.array(halo); print(f"halo stack q {QS[0]}-{QS[-1]} ({len(QS)}); v_h^2(R0) {vh2[0]:.0f}-{vh2[-1]:.0f}; v_b^2(R0) {vb2:.0f}", flush=True)
    hc = bcoef_nak(np.stack([filt(a) for a in halo]))                              # (nq + 2, ...): c_{-1..nq}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, axi_b=filt(phib), bar=np.stack([filt(b) for b in bar]), bar_m=np.arange(0, MMAX+1, 2), halo=hc, q=QS, vh2=bcoef_nak(np.array(vh2)),
             vb2=vb2, r0=r0, du=u[1]-u[0], dw=w[1]-w[0], rs=RS, zs=ZS)
    print(f"{OUT}: {NU} x {NW} nodes, {OUT.stat().st_size/1e6:.1f} MB, {time.time()-t1:.0f} s")


if __name__ == "__main__":
    main()
