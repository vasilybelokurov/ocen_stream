"""Unit tests for score_conditional (density-free along u)."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from ocen_dm.streams.score import score_conditional


def _data(rng, n=300):
    u = rng.uniform(0, 20, n); x = rng.normal(0, 0.5, n)
    pm = np.column_stack((-3-0.3*u+rng.normal(0, 0.3, n), -6-0.1*u+rng.normal(0, 0.3, n)))
    cov = np.zeros((n, 2, 2)); cov[:, 0, 0] = cov[:, 1, 1] = 0.04
    v = np.full(n, np.nan); e = np.full(n, np.nan)
    return u, np.column_stack((x, pm)), v, e, cov


def _model(rng, u):
    m = len(u); x = rng.normal(0, 0.5, m)
    w = np.column_stack((x, -3-0.3*u+rng.normal(0, 0.3, m), -6-0.1*u+rng.normal(0, 0.3, m)))
    chi = 5*u+1e-3                                   # chi increases with u: 1 chi bin of 5 = 1 deg in u
    return w, np.full(m, 230.), chi, np.full(m, 100.)


def test_no_bins_gives_background_only():
    rng = np.random.default_rng(1); du, dw, dv, de, dc = _data(rng)
    r = score_conditional(du, dw, dv, de, dc, np.zeros(5), np.zeros((5, 3)), np.zeros(5), -np.ones(5), np.zeros(5), u_window=(0, 20))
    assert r["K"] == 0 and np.allclose(r["per_star"], np.log(1/(12*30*30)))


def test_eps_one_is_background():
    rng = np.random.default_rng(2); du, dw, dv, de, dc = _data(rng); mu = rng.uniform(0, 20, 4000)
    mw, mv, chi, age = _model(rng, mu)
    r = score_conditional(du, dw, dv, de, dc, mu, mw, mv, chi, age, eps=1.0, u_window=(0, 20))
    assert np.allclose(r["per_star"], np.log(1/(12*30*30)))


def test_density_along_u_cancels():
    """Same conditional distribution, very different along-u density of the model (uniform vs steeply rising): the conditional
    score must agree within Monte Carlo noise, while the along-u marginal (not used) differs by a factor ~ 10."""
    rng = np.random.default_rng(3); du, dw, dv, de, dc = _data(rng)
    u1 = rng.uniform(0, 20, 20000); u2 = 20*np.sqrt(rng.uniform(0, 1, 20000))      # p(u) ~ u
    s = []
    for mu in (u1, u2):
        mw, mv, chi, age = _model(np.random.default_rng(4), mu)
        s.append(score_conditional(du, dw, dv, de, dc, mu, mw, mv, chi, age, nmin=5, u_window=(0, 20))["total"])
    assert abs(s[0]-s[1]) < 0.02*abs(s[0]), s


def test_duplicating_a_chi_bin_changes_nothing():
    rng = np.random.default_rng(5); du, dw, dv, de, dc = _data(rng); mu = rng.uniform(0, 20, 6000)
    mw, mv, chi, age = _model(rng, mu)
    a = score_conditional(du, dw, dv, de, dc, mu, mw, mv, chi, age, u_window=(0, 20))["total"]
    k = (chi >= 20) & (chi < 25)
    b = score_conditional(du, dw, dv, de, dc, np.concatenate((mu, mu[k])), np.vstack((mw, mw[k])), np.concatenate((mv, mv[k])),
                          np.concatenate((chi, chi[k])), np.concatenate((age, age[k])), u_window=(0, 20))["total"]
    assert abs(a-b) < 1e-8*abs(a)


def test_wrong_kinematics_scores_worse():
    rng = np.random.default_rng(6); du, dw, dv, de, dc = _data(rng); mu = rng.uniform(0, 20, 6000)
    mw, mv, chi, age = _model(rng, mu)
    good = score_conditional(du, dw, dv, de, dc, mu, mw, mv, chi, age, u_window=(0, 20))["total"]
    mw2 = mw.copy(); mw2[:, 1] += 1.0
    bad = score_conditional(du, dw, dv, de, dc, mu, mw2, mv, chi, age, u_window=(0, 20))["total"]
    assert bad < good-100


def test_explicit_background_bounds_normalise_and_reject_outside_members():
    """With explicit w_bounds the background is a normalised uniform density over that box; members outside it raise."""
    import pytest
    from ocen_dm.streams.score import score_conditional
    rng = np.random.default_rng(3)
    n = 50
    du = rng.uniform(0, 10, n); dw = np.column_stack((rng.uniform(-2, 2, n), rng.uniform(-5, -3, n), rng.uniform(-8, -6, n)))
    dv = np.full(n, np.nan); dev = np.full(n, np.nan); cov = np.tile(np.eye(2)*1e-4, (n, 1, 1))
    # no model particles -> background only: L_i = g_w
    empty = np.zeros(0)
    bounds = ((-3., 3.), (-6., -2.), (-9., -5.))
    r = score_conditional(du, dw, dv, dev, cov, empty, np.zeros((0, 3)), empty, empty, empty, u_window=(0., 10.), w_bounds=bounds)
    assert np.allclose(r["per_star"], -np.log(6.*4.*4.))
    # the legacy widths give the same value when they equal the box widths
    r0 = score_conditional(du, dw, dv, dev, cov, empty, np.zeros((0, 3)), empty, empty, empty, u_window=(0., 10.), w_box=(6., 4., 4.))
    assert np.allclose(r0["per_star"], r["per_star"])
    dw_out = dw.copy(); dw_out[0, 0] = 5.
    with pytest.raises(ValueError):
        score_conditional(du, dw_out, dv, dev, cov, empty, np.zeros((0, 3)), empty, empty, empty, u_window=(0., 10.), w_bounds=bounds)


def _toy_model(rng, n):
    """Particles of one fixed stream distribution (u, x, pmra, pmdec), chi spread over a long tail."""
    u = rng.uniform(0, 20, n); x = rng.normal(0, 0.5, n)
    pm = np.column_stack((-3-0.3*u+rng.normal(0, 0.3, n), -6-0.1*u+rng.normal(0, 0.3, n)))
    chi = rng.exponential(60., n)+1e-3
    return u, np.column_stack((x, pm)), np.full(n, np.nan), chi, np.zeros(n)


def test_pooled_equals_single_bin_and_fixed_grid_keeps_k_constant():
    rng = np.random.default_rng(11)
    data = _data(np.random.default_rng(1))
    du, dw, dcov = data[0], data[1], data[4]
    n = len(du); dv = np.full(n, np.nan); dev = np.full(n, np.nan)
    mu, mw, mv, chi, age = _toy_model(rng, 20000)
    one = np.full_like(chi, 2.)                       # every particle in the first legacy bin
    a = score_conditional(du, dw, dv, dev, dcov, mu, mw, mv, one, age, u_window=(0., 20.))
    b = score_conditional(du, dw, dv, dev, dcov, mu, mw, mv, chi, age, u_window=(0., 20.), pooled=True)
    assert a["K"] == 1 and b["K"] == 1 and np.isclose(a["total"], b["total"])
    edges = np.arange(0., 205., 5.)
    ks = [score_conditional(du, dw, dv, dev, dcov, mu[:m], mw[:m], mv[:m], chi[:m], age[:m], u_window=(0., 20.),
                            chi_edges=edges)["K"] for m in (500, 5000, 20000)]
    assert ks == [40, 40, 40]


def test_pooled_score_converges_with_particle_number_and_legacy_k_grows():
    """Two independent samples of the same model: the pooled lnL difference shrinks with N; legacy K grows with N."""
    data = _data(np.random.default_rng(2))
    du, dw, dcov = data[0], data[1], data[4]
    n = len(du); dv = np.full(n, np.nan); dev = np.full(n, np.nan)
    diffs, kl = [], []
    for m in (2000, 32000):
        s1 = _toy_model(np.random.default_rng(100+m), m); s2 = _toy_model(np.random.default_rng(200+m), m)
        t = [score_conditional(du, dw, dv, dev, dcov, *s, u_window=(0., 20.), pooled=True)["total"] for s in (s1, s2)]
        diffs.append(abs(t[0]-t[1]))
        kl.append(score_conditional(du, dw, dv, dev, dcov, *s1, u_window=(0., 20.))["K"])
    assert diffs[1] < diffs[0]
    assert kl[1] > kl[0]
