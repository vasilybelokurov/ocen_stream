"""Tests for the axisymmetric MW variants (src/ocen_dm/streams/mw_variants.py)."""
import os
import numpy as np
from ocen_dm.streams.mw_variants import axisymmetric_variant, R0_KPC
from ocen_dm.streams.restricted import HUNTER24_DIR, agama_kpc

PTS = np.array([[1., 0, 0.2], [3., 1., 0.5], [5., 0, 2.], [8.178, 0, 0], [6., 2., 3.], [15., 0, 5.]])


def test_default_variant_reproduces_original():
    agama = agama_kpc(); orig = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
    pot, info = axisymmetric_variant(HUNTER24_DIR, 1.0, 1.0, keep_vc=False)
    assert np.allclose(pot.force(PTS), orig.force(PTS), rtol=2e-3, atol=1e-3)


def test_keep_vc_and_flattening():
    agama = agama_kpc(); orig = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
    vc0 = np.sqrt(-R0_KPC*orig.force(np.array([[R0_KPC, 0, 0]]))[0, 0])
    for q, f in ((0.8, 1.0), (1.0, 1.2), (1.2, 0.8)):
        pot, info = axisymmetric_variant(HUNTER24_DIR, q, f)
        assert abs(info["vc_R0"]-vc0) < 0.05
    flat = axisymmetric_variant(HUNTER24_DIR, 0.8, 1.0)[0]; rnd = axisymmetric_variant(HUNTER24_DIR, 1.0, 1.0)[0]
    x = np.array([[5., 0, 3.]])
    assert abs(flat.force(x)[0, 2]) > abs(rnd.force(x)[0, 2])     # flattened halo pulls harder towards the plane at z = 3 kpc


def test_vc_target():
    for vc in (220., 240.):
        pot, info = axisymmetric_variant(HUNTER24_DIR, vc_target=vc)
        assert abs(info["vc_R0"]-vc) < 0.05
