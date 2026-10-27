"""Production SMC + emcee polish on analytic targets with deterministic jitter like ours (pilots.fake_chi2): the posterior moments,
the evidence and the extent of a curved valley are recovered; a resumed run retraces the first bit for bit."""
import numpy as np
import pytest
from scipy.special import ndtr
from ocen_stream import pilots as PL, smc as S

D = PL.D


def exact(mode, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    if mode == "gauss":
        u = rng.multivariate_normal(np.full(D, 0.5), PL.FAKE_COV, n)
    else:
        u0 = rng.normal(0.5, 0.2, 4*n); u0 = u0[(u0 > 0) & (u0 < 1)][:n]
        u = np.column_stack([u0, 0.4+1.5*(u0-0.5)**2+0.01*rng.standard_normal(n), 0.5+0.03*rng.standard_normal((n, D-2))])
    return u


def jitter_logmean(mode):
    """log E_p0[exp(-jitter / 2)]: the jitter's (deterministic) contribution to log Z, by Monte Carlo over the exact posterior."""
    u = exact(mode); c = PL.fake_chi2(u, mode)[0]; d = u-0.5
    if mode == "gauss":
        c0 = 166+np.einsum("ni,ij,nj->n", d, np.linalg.inv(PL.FAKE_COV), d)
    else:
        c0 = 166+((u[:, 0]-0.5)/0.2)**2+((u[:, 1]-0.4-1.5*(u[:, 0]-0.5)**2)/0.01)**2+np.sum(((u[:, 2:]-0.5)/0.03)**2, 1)
    return float(np.log(np.mean(np.exp(-0.5*(c-c0)))))


def run(tmp_path, mode, n=600, **kw):   # production defaults (smc.smc); log Z scatter between seeds ~0.4 (measured)
    ev = PL.Evaluator(None, tmp_path/mode, fake=mode); x, c, logz = S.smc(ev, tmp_path/mode, n=n, log=lambda *a: None, **kw)
    return ev, x, c, logz


def test_gauss_moments_and_evidence(tmp_path):
    ev, x, c, logz = run(tmp_path, "gauss")
    sd = np.sqrt(np.diag(PL.FAKE_COV)); m = x.mean(0)
    assert np.all(np.abs(m-0.5)/sd < 0.25), (m-0.5)/sd
    assert np.all(np.abs(x.std(0)/sd-1) < 0.2), x.std(0)/sd
    # Z = int exp(-chi2/2) du (box >> posterior): exp(-83) sqrt((2 pi)^d |C|) x the jitter term
    lz = -83+0.5*D*np.log(2*np.pi)+0.5*np.linalg.slogdet(PL.FAKE_COV)[1]+jitter_logmean("gauss")
    assert abs(logz-lz) < 1.0, (logz, lz)


def test_curved_valley_extent(tmp_path):
    ev, x, c, logz = run(tmp_path, "valley")
    # truth: u0 ~ N(0.5, 0.2) truncated to [0, 1]; u1 = 0.4 + 1.5 (u0 - 0.5)^2 +- 0.01
    q = np.quantile(x[:, 0], [0.1, 0.5, 0.9]); assert np.all(np.abs(q-np.array([0.26, 0.5, 0.74])) < 0.1), q
    assert np.median(np.abs(x[:, 1]-0.4-1.5*(x[:, 0]-0.5)**2)) < 0.012
    lz = -83+0.5*D*np.log(2*np.pi)+np.log(0.2*0.01*0.03**7)+np.log(ndtr(2.5)-ndtr(-2.5))+jitter_logmean("valley")
    assert abs(logz-lz) < 1.0, (logz, lz)


def test_resume_identical_and_polish(tmp_path):
    ev, x, c, logz = run(tmp_path, "gauss", n=300)
    (tmp_path/"gauss"/"smc_final.npz").unlink(); st = np.load(tmp_path/"gauss"/"smc_state.npz")
    ev2 = PL.Evaluator(None, tmp_path/"gauss", fake="gauss"); x2, c2, l2 = S.smc(ev2, tmp_path/"gauss", n=300, log=lambda *a: None)
    assert ev2.nev == 0 and np.array_equal(x, x2) and logz == l2
    s = S.polish(ev2, tmp_path/"gauss", x, nwalkers=40, nsteps=600, log=lambda *a: None)
    ch = s.get_chain(discard=200, flat=True); sd = np.sqrt(np.diag(PL.FAKE_COV))
    assert np.all(np.abs(ch.mean(0)-0.5)/sd < 0.3) and np.all(np.abs(ch.std(0)/sd-1) < 0.25), (ch.mean(0)-0.5)/sd


def test_interrupted_resume_identical(tmp_path):
    """a run killed in the middle of a sweep and resumed gives the same population, chi2 and log Z as an uninterrupted run."""
    ev, x, c, logz = run(tmp_path, "gauss", n=200)

    class Boom(Exception):
        pass

    class Killer(PL.Evaluator):
        def __call__(self, U, tag, seed=7):
            if self.nev > 3000:
                raise Boom
            return super().__call__(U, tag, seed)
    o = tmp_path/"killed"; k = Killer(None, o, fake="gauss")
    with pytest.raises(Boom):
        S.smc(k, o, n=200, log=lambda *a: None)
    ev2 = PL.Evaluator(None, o, fake="gauss"); x2, c2, l2 = S.smc(ev2, o, n=200, log=lambda *a: None)
    assert np.array_equal(x, x2) and np.array_equal(c, c2) and logz == l2


def test_kept_beta_and_stationarity(tmp_path):
    """the T = 10 population is saved exactly at beta 0.1; started from exact posterior draws, the moves keep the Gaussian (no
    variance contraction: sd within 5% of the jittered target's, whose jitter grows with chi2, after 40 sweeps of 2000 particles)."""
    ev, x, c, logz = run(tmp_path, "gauss", n=300)
    b = sorted(float(np.load(f)["beta"]) for f in (tmp_path/"gauss").glob("smc_stage*.npz")); assert 0.1 in b and b[-1] == 1.0
    rng = np.random.default_rng(7); x0 = rng.multivariate_normal(np.full(D, 0.5), PL.FAKE_COV, 2000)
    o = tmp_path/"stat"; o.mkdir(); ev2 = PL.Evaluator(None, o, fake="gauss"); c0 = ev2(x0, "init")
    S._save(o/"smc_state.npz", x=x0, c=c0, beta=1., stage=1, sweep=0, acc_sum=0., logz=0.)
    xs, cs, _ = S.smc(ev2, o, n=2000, min_sweeps=40, moves=1e9, max_sweeps=40, log=lambda *a: None)
    u = exact("gauss"); d = u-0.5; c0 = 166+np.einsum("ni,ij,nj->n", d, np.linalg.inv(PL.FAKE_COV), d)
    w = np.exp(-0.5*(PL.fake_chi2(u, "gauss")[0]-c0)); sd_t = np.sqrt(np.average((u-np.average(u, 0, w))**2, 0, w))   # jittered target
    r = xs.std(0)/sd_t; assert np.all(np.abs(r-1) < 0.05), r


def test_changed_box_refused(tmp_path):
    o = tmp_path/"box"; PL.Evaluator(None, o, fake="gauss")
    lim = dict(PL.LIM)
    try:
        PL.LIM["dist"] = (5.0, 6.0)
        with pytest.raises(RuntimeError):
            PL.Evaluator(None, o, fake="gauss")
    finally:
        PL.LIM.clear(); PL.LIM.update(lim)
