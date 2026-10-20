"""gravity21 solar frame, v_c normalisation radius and the step2 design (2026-10-20)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]


def test_gravity21_matches_sgra_proper_motion():
    from ocen_dm.streams.frames import FRAMES
    f = FRAMES["gravity21"]
    mu = f["v_sun"][1]/(4.74047*f["r0_kpc"])
    assert abs(mu-6.411) < 1e-3                                   # Reid & Brunthaler 2020
    assert abs(FRAMES["baumgardt"]["v_sun"][1]/(4.74047*FRAMES["baumgardt"]["r0_kpc"])-6.507) < 1e-3   # the old frame's defect


def test_sgra_at_rest_in_gravity21():
    """Sgr A* (Galactocentric origin, at rest) seen from the gravity21 Sun: mu_l = -6.411 mas/yr."""
    import astropy.coordinates as coord, astropy.units as u
    from ocen_dm.streams.frames import galcen
    g = coord.SkyCoord(x=0*u.kpc, y=0*u.kpc, z=0*u.kpc, v_x=0*u.km/u.s, v_y=0*u.km/u.s, v_z=0*u.km/u.s, frame=galcen("gravity21"))
    gal = g.transform_to(coord.Galactic())
    assert abs(gal.pm_l_cosb.to_value(u.mas/u.yr)+6.411) < 2e-3
    assert abs(gal.distance.to_value(u.kpc)-8.275) < 1e-6


def test_vc_normalised_at_frame_r0():
    from ocen_dm.streams.mw_variants import axisymmetric_variant, _vc2
    from ocen_dm.streams.restricted import HUNTER24_DIR
    from ocen_dm.streams.frames import frame_r0
    r0 = frame_r0("gravity21")
    pot, info = axisymmetric_variant(HUNTER24_DIR, halo_q=0.9, vc_target=233., r0_kpc=r0)
    assert abs(np.sqrt(_vc2(pot, r0))-233.) < 1e-3 and info["r0_kpc"] == r0
    pot0, _ = axisymmetric_variant(HUNTER24_DIR, halo_q=0.9, vc_target=233.)      # default radius unchanged
    assert abs(np.sqrt(_vc2(pot0, 8.178))-233.) < 1e-3


def test_design_maps():
    import emulator_design as ED
    rng = np.random.default_rng(3)
    for u in rng.uniform(size=(3, 9)):
        p1 = ED.unit_to_params(u, "v1"); p0 = ED.unit_to_params(u)
        assert p1 == p0
        for design in ("v1", "step2"):
            q = ED.unit_to_params(u, design)
            assert np.allclose(ED.params_to_unit(q, design), u, atol=1e-9)
    q = ED.unit_to_params(np.full(9, 0.5), "step2")
    assert abs(q["dist"]-5.43) < 1e-12 and abs(q["vc"]-235.) < 1e-12


def test_vc_prior():
    from profile_nm import ln_prior
    from ocen_dm.cluster import OCEN_DISTANCE_KPC, MU_SYS
    p = dict(dist=OCEN_DISTANCE_KPC, pmra=MU_SYS[0], pmdec=MU_SYS[1], vc=219.)
    assert ln_prior(p) == 0. and ln_prior(p, True) == pytest.approx(-2.0)


def test_mw_variants_default_unchanged():
    """Existing behaviour (keep_vc at 8.178) unchanged."""
    from ocen_dm.streams.mw_variants import axisymmetric_variant, _vc2
    from ocen_dm.streams.restricted import HUNTER24_DIR, agama_kpc
    agama = agama_kpc()
    import os
    orig = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
    pot, _ = axisymmetric_variant(HUNTER24_DIR, halo_q=0.8)
    assert abs(_vc2(pot)-_vc2(orig)) < 1e-6*_vc2(orig)
