"""Axisymmetric Milky Way variants of the Hunter+2024 model (../oCen_bar/agama_potentials/example_mw_potential_hunter24.py) for the
potential-shape test (docs/STREAM_DIAGNOSTIC_PLAN.md, Phase 3 item 1). The bar's non-axisymmetric part is NOT rebuilt (host_potential
adds A x (baryon_full - baryon_axi) as before); only the axisymmetric part is replaced by
    baryon_axi (file) + (disc_scale - 1) x [thin + thick stellar discs]  +  halo(q) x c,
where halo(q) is the Hunter+2024 Einasto halo (Spheroid parameters copied from the construction script) with axisRatioZ = q, and c is
chosen so that the circular velocity at R0 = 8.178 kpc equals that of the original model (keep_vc=True). r0_kpc moves the
radius at which v_c is normalised (vc_target; frames.frame_r0 for a frame other than baumgardt; added 2026-10-20). disc_scale = 1, q = 1
reproduces the original axisymmetric potential (tested in tests/test_mw_variants.py)."""
from __future__ import annotations

import os

import numpy as np

R0_KPC = 8.178
# copied from example_mw_potential_hunter24.py (makePotentialModel)
PARAMS_DARK = dict(type="Spheroid", densitynorm=2.774e11, gamma=0, beta=0, alpha=1, outerCutoffRadius=8.682e-6, cutoffStrength=0.1704)
PARAMS_DISK = [dict(type="Disk", surfaceDensity=1.332e9, scaleRadius=2.0, scaleHeight=0.3, innerCutoffRadius=2.7, sersicIndex=1),
               dict(type="Disk", surfaceDensity=8.97e8, scaleRadius=2.8, scaleHeight=0.9, innerCutoffRadius=2.7, sersicIndex=1)]
MUL = dict(type="Multipole", lmax=12, gridSizeR=36, rmin=1e-4, rmax=1000)
CYL = dict(type="CylSpline", gridSizeR=30, gridSizez=25, Rmin=0.1, Rmax=50, zmin=0.05, zmax=20, mmax=0)


def _vc2(pot, R=R0_KPC):
    return float(-R*pot.force(np.array([[R, 0., 0.]]))[0, 0])


def axisymmetric_variant(hunter_dir, halo_q=1.0, disc_scale=1.0, keep_vc=True, vc_target=None, r0_kpc=R0_KPC):
    """Return (axisymmetric agama.Potential, info dict). See module docstring. vc_target [km/s] (optional): choose the halo
    normalisation c so that v_c(R0) = vc_target instead (overrides keep_vc; baryons unchanged)."""
    from .restricted import agama_kpc
    agama = agama_kpc()
    bary = agama.Potential(file=os.path.join(hunter_dir, "MWPotentialHunter24_baryon_axi.ini"))
    comps = [bary]
    if disc_scale != 1.0:
        dd = [dict(p, surfaceDensity=p["surfaceDensity"]*(disc_scale-1.0)) for p in PARAMS_DISK]
        comps.append(agama.Potential(density=agama.Density(*dd), **CYL))
    base = agama.Potential(*comps)
    halo1 = agama.Potential(density=agama.Density(**dict(PARAMS_DARK, axisRatioZ=halo_q)), **dict(MUL, symmetry="Axisymmetric"))
    c = 1.0
    if vc_target is not None:
        c = (vc_target**2-_vc2(base, r0_kpc))/_vc2(halo1, r0_kpc)
        if c <= 0:
            raise ValueError("cannot reach vc_target: baryons alone exceed it")
    elif keep_vc:
        orig = agama.Potential(file=os.path.join(hunter_dir, "MWPotentialHunter24_axi.ini"))
        c = (_vc2(orig, r0_kpc)-_vc2(base, r0_kpc))/_vc2(halo1, r0_kpc)
        if c <= 0:
            raise ValueError("cannot keep v_c(R0): baryons alone exceed it")
    halo = agama.Potential(density=agama.Density(**dict(PARAMS_DARK, axisRatioZ=halo_q, densitynorm=PARAMS_DARK["densitynorm"]*c)),
                           **dict(MUL, symmetry="Axisymmetric"))
    pot = agama.Potential(base, halo)
    return pot, dict(halo_q=halo_q, disc_scale=disc_scale, halo_norm_factor=c, vc_R0=np.sqrt(_vc2(pot, r0_kpc)), r0_kpc=r0_kpc)
