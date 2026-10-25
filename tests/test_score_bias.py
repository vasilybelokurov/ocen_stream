"""score_bias.chi_bin_bias: reproduces the fixed-chi-bin score and removes most of its small-N bias (toy, i.i.d. particles)."""
import numpy as np

from ocen_stream.score import score_conditional
from ocen_stream.score_bias import chi_bin_bias, jackknife_total

E = np.arange(0., 205., 5.)


def _data(rng, n=300):
    u = rng.uniform(0, 20, n); x = rng.normal(0, 0.5, n)
    pm = np.column_stack((-3-0.3*u+rng.normal(0, 0.3, n), -6-0.1*u+rng.normal(0, 0.3, n)))
    cov = np.zeros((n, 2, 2)); cov[:, 0, 0] = cov[:, 1, 1] = 0.04
    return u, np.column_stack((x, pm)), np.full(n, np.nan), np.full(n, np.nan), cov


def _model(rng, n):
    u = rng.uniform(0, 20, n); x = rng.normal(0, 0.5, n)
    pm = np.column_stack((-3-0.3*u+rng.normal(0, 0.3, n), -6-0.1*u+rng.normal(0, 0.3, n)))
    return u, np.column_stack((x, pm)), np.full(n, np.nan), rng.exponential(60., n)+1e-3, np.zeros(n)


def test_reproduces_score_conditional():
    du, dw, dv, de, dc = _data(np.random.default_rng(1)); mu, mw, mv, chi, age = _model(np.random.default_rng(2), 3000)
    r = chi_bin_bias(du, dw, dv, de, dc, mu, mw, mv, chi, age, E, u_window=(0., 20.))
    s = score_conditional(du, dw, dv, de, dc, mu, mw, mv, chi, age, u_window=(0., 20.), chi_edges=E)
    np.testing.assert_allclose(r["per_star"], s["per_star"], rtol=0, atol=1e-10)


def test_jackknife_removes_the_bias_and_delta_method_part_of_it():
    du, dw, dv, de, dc = _data(np.random.default_rng(1))
    sc = lambda m: score_conditional(du, dw, dv, de, dc, m["u"], m["w"], m["v"], m["chi"], m["age"], u_window=(0., 20.), chi_edges=E)
    as_dict = lambda t: dict(u=t[0], w=t[1], v=t[2], chi=t[3], age=t[4])
    ref = np.mean([sc(as_dict(_model(np.random.default_rng(90+i), 300000)))["total"] for i in range(3)])
    raw, jk, dm = [], [], []
    for s in range(40):
        m = as_dict(_model(np.random.default_rng(5000+s), 1500))
        j = jackknife_total(sc, m); raw.append(j["total"]); jk.append(j["total_corrected"])
        dm.append(chi_bin_bias(du, dw, dv, de, dc, m["u"], m["w"], m["v"], m["chi"], m["age"], E, u_window=(0., 20.))["total_corrected"])
    se = np.std(jk, ddof=1)/np.sqrt(len(jk))
    b_raw, b_jk, b_dm = np.mean(raw)-ref, np.mean(jk)-ref, np.mean(dm)-ref
    assert b_raw < -4*se                       # raw score biased low at this N
    assert abs(b_jk) < max(3*se, 0.25*abs(b_raw))
    assert b_raw < b_dm < 0                     # delta method: right sign, partial
