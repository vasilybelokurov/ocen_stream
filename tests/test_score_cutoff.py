"""score_conditional(cutoff, jackknife_groups) (2026-10-18): cutoff 7 reproduces the exact score; the one-pass jackknife equals
the brute-force score_bias.jackknife_total; legacy bins refuse the one-pass jackknife."""
import json
from pathlib import Path

import numpy as np
import pytest

from ocen_stream.score import score_conditional
from ocen_stream.score_bias import jackknife_total

ROOT = Path(__file__).resolve().parents[1]
E = np.arange(0., 205., 5.)


def _case(seed=1, n=400, m=6000):
    rng = np.random.default_rng(seed)
    du = rng.uniform(0, 20, n); x = rng.normal(0, 0.8, n)
    dw = np.column_stack((x, -3-0.3*du+rng.normal(0, 0.3, n), -6-0.1*du+rng.normal(0, 0.3, n)))
    e1, e2, rho = rng.uniform(0.02, 0.4, n), rng.uniform(0.02, 0.4, n), rng.uniform(-0.6, 0.6, n)
    cov = np.zeros((n, 2, 2)); cov[:, 0, 0] = e1**2; cov[:, 1, 1] = e2**2; cov[:, 0, 1] = cov[:, 1, 0] = rho*e1*e2   # correlated
    dv = np.where(rng.uniform(size=n) < 0.1, 230+rng.normal(0, 10, n), np.nan); dev = np.where(np.isfinite(dv), 2., np.nan)
    mu = rng.uniform(-2, 22, m); mw = np.column_stack((rng.normal(0, 0.6, m), -3-0.3*mu+rng.normal(0, 0.3, m), -6-0.1*mu+rng.normal(0, 0.3, m)))
    mv = 230+rng.normal(0, 8, m); chi = rng.exponential(40., m)+1e-3      # sparse tail bins, some empty in 0-200
    chi[rng.uniform(size=m) < 0.05] *= -1                                  # some leading-arm particles (chi < 0, excluded)
    return du, dw, dv, dev, cov, mu, mw, mv, chi, np.full(m, 100.)


@pytest.mark.parametrize("kw", [dict(chi_edges=E), dict(pooled=True)])
def test_cutoff7_matches_exact(kw):
    a = _case()
    ex = score_conditional(*a, u_window=(0., 20.), return_neff=True, return_components=True, **kw)
    cu = score_conditional(*a, u_window=(0., 20.), return_neff=True, return_components=True, cutoff=7., chunk=37, **kw)
    assert np.max(np.abs(cu["per_star"]-ex["per_star"])) < 1e-6
    np.testing.assert_allclose(cu["foreground_u"], ex["foreground_u"], rtol=1e-9, atol=1e-300)
    fg = ex["foreground_joint"] > 1e-6*ex["foreground_joint"].max()
    np.testing.assert_allclose(cu["n_eff"][fg], ex["n_eff"][fg], rtol=1e-4)


def test_one_pass_jackknife_equals_brute_force():
    du, dw, dv, dev, cov, mu, mw, mv, chi, age = _case(seed=2)
    model = dict(u=mu, w=mw, v=mv, chi=chi, age=age)
    fn = lambda md: score_conditional(du, dw, dv, dev, cov, md["u"], md["w"], md["v"], md["chi"], md["age"], u_window=(0., 20.), chi_edges=E)
    ref = jackknife_total(fn, model)
    one = score_conditional(du, dw, dv, dev, cov, mu, mw, mv, chi, age, u_window=(0., 20.), chi_edges=E, cutoff=7., jackknife_groups=10)
    assert abs(one["total_jk"]-ref["total_corrected"]) < 1e-4
    np.testing.assert_allclose(one["leave_out_totals"], ref["totals_leave_out"], atol=1e-5)


def test_legacy_bins_refuse_one_pass_jackknife():
    a = _case()
    with pytest.raises(ValueError):
        score_conditional(*a, u_window=(0., 20.), cutoff=7., jackknife_groups=10)
    with pytest.raises(ValueError):
        score_conditional(*a, u_window=(0., 20.), chi_edges=E, jackknife_groups=10)     # needs cutoff


def test_real_spray_jackknife_matches_stored():
    import sys
    sys.path.insert(0, str(ROOT/"bin/streams"))
    f = ROOT/"results/streams/training_9d_20261017/runs/p176_r0_n256k.npz"
    if not f.exists():
        pytest.skip("training spray not present")
    G = pytest.importorskip("grid4_common")
    m = dict(np.load(f)); stored = json.loads(f.with_name("p176_r0_n256k_jk.json").read_text())
    s = G.score(m, chi_edges=E, chunk=50, cutoff=7., jackknife_groups=10)
    assert abs(s["total"]-stored["total"]) < 1e-3 and abs(s["total_jk"]-stored["total_jk"]) < 1e-3
