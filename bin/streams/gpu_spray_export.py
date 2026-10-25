#!/usr/bin/env python3
"""Export the inputs of the JAX spray model (GPU pipeline stage 2, 2026-10-25): frame constants, omega Cen, progenitor, random bank.

- Frame (gravity21): the astropy ICRS -> Galactocentric map is affine; A, b and v_sun are read off by transforming the origin and
  unit vectors (x_gc = A x_icrs + b, v_gc = A v_icrs + v_sun; model frame = diag(-1, 1, 1) x_gc as frames.to_model); ICRS ->
  Galactic rotation Agal likewise.
- omega Cen: ra, dec, v_los (frames.OCEN_OBS); progenitor: Phi_sat(ln r) B-spline of the frozen A_nodm profile and its M(<r) table
  (grid4_common r_kpc, M); release set-up of the fits: window 1000 Myr, nrel, max age 700 Myr, T.
- Bank: the normal draws of spray.release_ic for numpy default_rng(SEED) (rx, rz, rvy, rvz; 2 nrel each), so a JAX spray with
  this bank and the same epoch offset reproduces grid4_common.make_spray(seed=SEED, epoch_offset=...) particle by particle.
Usage: PYTHONPATH=src python bin/streams/gpu_spray_export.py [SEED] [NREL] -> results/gpu/spray_inputs.npz
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import spline_filter
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import astropy.coordinates as coord
import astropy.units as u
from grid4_common import T, r_kpc, M
from ocen_stream.frames import galcen, OCEN_OBS
from ocen_stream.restricted import agama_kpc, satellite_potential, FrozenCore

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3150002                 # final_c0_2 (fit_knot_nm.py chain c0, final 2)
NREL = int(sys.argv[2]) if len(sys.argv) > 2 else 256000
OUT = ROOT/"results/gpu/spray_inputs.npz"


def affine(frame="gravity21"):
    P = np.vstack((np.zeros(3), np.eye(3)))
    c = coord.SkyCoord(x=P[:, 0]*u.kpc, y=P[:, 1]*u.kpc, z=P[:, 2]*u.kpc, v_x=P[:, 0]*u.km/u.s, v_y=P[:, 1]*u.km/u.s, v_z=P[:, 2]*u.km/u.s,
                       frame="icrs", representation_type="cartesian", differential_type="cartesian").transform_to(galcen(frame))
    x = np.column_stack((c.x.to_value(u.kpc), c.y.to_value(u.kpc), c.z.to_value(u.kpc)))
    v = np.column_stack((c.v_x.to_value(u.km/u.s), c.v_y.to_value(u.km/u.s), c.v_z.to_value(u.km/u.s)))
    g = coord.SkyCoord(x=P[1:, 0], y=P[1:, 1], z=P[1:, 2], frame="icrs", representation_type="cartesian").transform_to(coord.Galactic())
    Agal = np.column_stack([g.cartesian.xyz.value[:, i] for i in range(3)])
    return (x[1:]-x[0]).T, x[0], v[0], Agal


def main():
    agama = agama_kpc(); A, b, vsun, Agal = affine()
    sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=FrozenCore(r_kpc, M))
    lr = np.linspace(np.log(1e-5), np.log(50.), 2048); rr = np.exp(lr); sphi = sat.potential(np.column_stack((rr, 0*rr, 0*rr)))
    rng = np.random.default_rng(SEED); bank = np.stack([rng.normal(size=2*NREL) for _ in range(4)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, A=A, b=b, vsun=vsun, Agal=Agal, ra=OCEN_OBS["ra"], dec=OCEN_OBS["dec"], vlos=OCEN_OBS["vlos"],
             sat_coef=spline_filter(sphi, order=3, mode="nearest"), lr0=lr[0], dlr=lr[1]-lr[0], sat_mtot=-sphi[-1]*rr[-1]/agama.G, G=agama.G,
             r_kpc=r_kpc, M_enc=M, bank=bank, seed=SEED, nrel=NREL, window_myr=1000., max_age_myr=700., T=T)
    print(f"{OUT.relative_to(ROOT)}: seed {SEED}, nrel {NREL}, {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
