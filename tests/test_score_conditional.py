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
