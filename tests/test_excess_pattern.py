"""The 'excess' bar pattern (restricted.host_potential, 2026-10-21): co-rotating and in phase it reproduces the inflated-bar host."""
import numpy as np


def test_excess_corotating_equals_inflated_bar():
    from ocen_dm.streams.restricted import host_potential
    T, om, ang, A, S = 2.0, 33.14, 12.5, 1.37, 1.33
    infl = host_potential("x", bar_omega=om, bar_angle_deg=ang, t_today=T, bar_amp=A, bar_size=S)
    pub = host_potential("x", bar_omega=om, bar_angle_deg=ang, t_today=T, bar_amp=1.0, bar_size=1.0,
                         excess=dict(amp=A, size=S, omega=om, angle_deg=ang))
    rng = np.random.default_rng(0); P = np.column_stack((rng.uniform(-8, 8, (50, 2)), rng.uniform(-2, 2, 50)))
    for t in (T, T-0.3, T-0.7):
        np.testing.assert_allclose(pub.force(P, t=t), infl.force(P, t=t), rtol=1e-6, atol=1e-6)


def test_excess_has_no_axisymmetric_part():
    from ocen_dm.streams.restricted import host_potential
    T = 2.0
    a = host_potential("x", bar_omega=33., bar_angle_deg=25., t_today=T, bar_amp=1.0, bar_size=1.0)
    b = host_potential("x", bar_omega=33., bar_angle_deg=25., t_today=T, bar_amp=1.0, bar_size=1.0,
                       excess=dict(amp=1.4, size=1.25, omega=0., angle_deg=40.))
    phi = np.linspace(0, 2*np.pi, 145)[:-1]
    for R in (3., 6.):
        P = np.column_stack((R*np.cos(phi), R*np.sin(phi), np.full_like(phi, 0.5)))
        d = b.potential(P, t=T)-a.potential(P, t=T)
        assert abs(d.mean()) < 2e-3*np.abs(a.potential(P, t=T)).mean()
