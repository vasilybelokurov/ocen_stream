"""Tests for the barred host options in host_potential (constant and slowing bar, static size factor)."""
import numpy as np
from ocen_dm.streams.restricted import host_potential

T = 2.0


def _bar_axis_deg(pot, t, R=2.0):
    phi = np.radians(np.arange(0, 180, 0.25))
    rho = pot.density(np.column_stack((R*np.cos(phi), R*np.sin(phi), 0*phi)), t=t)
    return np.degrees(phi[np.argmax(rho)])


def test_new_code_path_matches_old_constant_bar():
    a = host_potential("x", bar_omega=35., bar_angle_deg=28., t_today=T, bar_amp=1.2)
    b = host_potential("x", bar_omega=35., bar_angle_deg=28., t_today=T, bar_amp=1.2, bar_eta=None, bar_size=1.0)
    x = np.array([[3., 1., 0.2], [-1., 2., 0.], [6., -2., 1.]])
    for t in (0.3, 1.5, T):
        assert np.allclose(a.potential(x, t=t), b.potential(x, t=t), rtol=1e-12)


def test_slowing_bar_today_equals_constant_bar():
    a = host_potential("x", bar_omega=35., bar_angle_deg=28., t_today=T, bar_amp=1.2)
    s = host_potential("x", bar_omega=35., bar_angle_deg=28., t_today=T, bar_amp=1.2, bar_eta=0.004)
    x = np.array([[3., 1., 0.2], [-1., 2., 0.], [6., -2., 1.]])
    assert np.allclose(a.potential(x, t=T), s.potential(x, t=T), rtol=1e-6)


def test_slowing_bar_angle_history():
    eta, Om = 0.004, 35.
    s = host_potential("x", bar_omega=Om, bar_angle_deg=28., t_today=T, bar_amp=1.2, bar_eta=eta)
    for dt in (0.05, 0.1):
        expect = (28. + np.degrees(np.log(1.-eta*Om*dt)/eta)) % 180.
        got = _bar_axis_deg(s, T-dt)
        assert abs(((got-expect+90.) % 180.)-90.) < 0.6


def test_portail_bar_orientation_today():
    p = host_potential("x", bar_omega=35., bar_angle_deg=28., t_today=T, bar_amp=1.0, bar_model="portail17")
    assert abs(_bar_axis_deg(p, T)-28.) < 0.6


def test_bar_size_lengthens_bar():
    """Non-axisymmetric potential amplitude at R = 5 kpc grows with the static size factor."""
    phi = np.radians(np.arange(0, 180, 1.))
    pts = np.column_stack((5*np.cos(phi), 5*np.sin(phi), 0*phi))
    amp = []
    for s in (1.0, 1.3):
        p = host_potential("x", bar_omega=35., bar_angle_deg=0., t_today=T, bar_amp=1.0, bar_size=s)
        v = p.potential(pts, t=T); amp.append(v.max()-v.min())
    assert amp[1] > amp[0]
