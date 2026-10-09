"""Tests for the optional cluster-rotation term in the spray release (src/ocen_dm/streams/spray.py release_ic)."""
import numpy as np
from ocen_dm.streams.spray import release_ic


def _setup(n=50):
    rng = np.random.default_rng(3)
    orbit = np.column_stack((rng.normal(size=(n, 3))*5, rng.normal(size=(n, 3))*200))
    rj = np.full(n, 0.1); vj = np.full(n, 3.)
    R = np.array([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])
    return orbit, rj, vj, R


def test_zero_rotation_is_identity():
    orbit, rj, vj, R = _setup()
    a, _ = release_ic(orbit, rj, vj, R, np.random.default_rng(1))
    b, _ = release_ic(orbit, rj, vj, R, np.random.default_rng(1), spin=dict(r_kpc=np.array([0., 1.]), vrot_kms=np.zeros(2), s=np.array([0, 0, 1.])))
    assert np.array_equal(a, b)


def test_added_velocity_perpendicular_and_sized():
    orbit, rj, vj, R = _setup()
    s = np.array([0.3, -0.2, 0.9]); s /= np.linalg.norm(s)
    a, _ = release_ic(orbit, rj, vj, R, np.random.default_rng(1))
    b, _ = release_ic(orbit, rj, vj, R, np.random.default_rng(1), spin=dict(r_kpc=np.array([0., 1.]), vrot_kms=np.array([2., 2.]), s=s))
    dv = b[:, 3:]-a[:, 3:]; off = b[:, :3]-np.repeat(orbit[:, :3], 2, axis=0)
    assert np.allclose(np.linalg.norm(dv, axis=1), 2.)
    assert np.allclose(dv @ s, 0., atol=1e-10)
    assert np.allclose(np.einsum("ij,ij->i", dv, off), 0., atol=1e-10)
    assert np.all(np.einsum("ij,ij->i", np.cross(off, dv), np.tile(s, (len(dv), 1))) > 0)   # rotation sense +s for v_rot > 0
