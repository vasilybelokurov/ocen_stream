"""Production sampler (2026-10-27): adaptive tempered SMC to beta = 1, then an emcee polish (design: Fable, Codex and the pilots,
docs/reviews/sampler_fable_codex_20261026.md; JOURNAL 2026-10-27).

Target: p(u | data) proportional to exp(-chi2(u) / 2) on the unit box (pilots.to_params maps u to the 9 parameters; box set by --box).
  SMC       N particles from the prior (scrambled Sobol; flat box, optionally a Gaussian on the distance, untempered); next beta by
            bisection so that the ESS of the incremental weights is N / 2, stopping exactly at KEEP_BETAS (0.1: the T = 10 population);
            systematic resampling; differential-evolution Metropolis moves in two half-ensembles with the other half as frozen donors
            (gamma 2.38 / sqrt(2 d), gamma 1 every 10th sweep, 1e-4 jitter); proposals outside the box rejected (Codex 2026-10-27:
            reflection is not symmetric for correlated DE steps; simultaneous updates with moving donors are not invariant); per stage at
            least MIN_SWEEPS sweeps
            and more until the summed acceptance >= MOVES (each particle moved ~MOVES times) or MAX_SWEEPS. Defaults (N 600, 2 sweeps,
            moves 2) from tests on analytic targets with our jitter: Gaussian sds within 10%, log Z within +-0.6, a curved valley's extent
            recovered; 60k (Gaussian) - 110k (curved valley) evaluations to beta = 1. Evidence: log Z = sum over
            stages of log mean(incremental weights) (likelihood exp(-chi2 / 2), normalised prior on the box). Every stage saved after
            its moves (OUT/smc_stageNN.npz); state after every sweep (OUT/smc_state.npz) -> exact resume; a changed configuration
            (box, prior, sampler settings, sources) is refused.
  polish    emcee (80% DEMove, 20% StretchMove), walkers drawn from the final SMC population, vectorised: each half-ensemble is one
            evaluator batch; HDF5 backend OUT/polish.h5 (resumes); integrated autocorrelation times in OUT/polish_tau.json.
Usage: python -m ocen_stream.smc --inputs DIR --out DIR [--n 600] [--beta-end 1] [--box dist=5.28,5.78] [--dist-prior 5.43,0.05]
       [--polish-steps 1000]
       [--fake gauss|valley]  (analytic test targets with deterministic jitter, no GPU)
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from . import pilots as PL

D = PL.D


def _save(path, **s):
    tmp = Path(str(path)+".tmp.npz"); np.savez(tmp, **s); os.replace(tmp, path)


def _logmeanexp(a):
    m = np.max(a); return m+np.log(np.mean(np.exp(a-m)))


def log_prior(u, dist_prior=None):
    """log prior on the unit box (up to a constant): flat, or a Gaussian on the distance (mu, sigma) [kpc] in physical units
    (the distance is linear in u, so no Jacobian); -inf outside the box."""
    u = np.atleast_2d(u); inside = np.all((u >= 0) & (u <= 1), 1); lp = np.where(inside, 0., -np.inf)
    if dist_prior is not None:
        d = PL.LIM["dist"][0]+(PL.LIM["dist"][1]-PL.LIM["dist"][0])*u[:, 4]; lp = lp-0.5*((d-dist_prior[0])/dist_prior[1])**2
    return lp


def _init(n, seed, dist_prior):
    """N points from the prior: scrambled Sobol; the distance coordinate by the inverse CDF of the truncated Gaussian prior if any."""
    from scipy.stats import qmc, truncnorm
    x = qmc.Sobol(D, scramble=True, seed=seed).random(n)
    if dist_prior is not None:
        lo, hi = PL.LIM["dist"]; m, s = dist_prior; d = truncnorm.ppf(x[:, 4], (lo-m)/s, (hi-m)/s, loc=m, scale=s); x[:, 4] = (d-lo)/(hi-lo)
    return x


def smc(ev, out, n=600, beta_end=1., min_sweeps=2, moves=2., max_sweeps=25, seed=1, keep_betas=(0.1,), dist_prior=None, log=print):
    """target prior(u) x exp(-beta chi2 / 2); returns the final population, chi2 and log Z (likelihood exp(-chi2 / 2))."""
    out = Path(out); sp = out/"smc_state.npz"; lpr = lambda u: log_prior(u, dist_prior)
    if sp.exists():
        z = np.load(sp); x, c, beta, stage, sw, acc_sum, logz = z["x"], z["c"], float(z["beta"]), int(z["stage"]), int(z["sweep"]), float(z["acc_sum"]), float(z["logz"])
    else:
        x = _init(n, seed, dist_prior); c = ev(x, "smc_init"); beta, stage, sw, acc_sum, logz = 0., 0, max_sweeps, moves, 0.
        _save(sp, x=x, c=c, beta=beta, stage=stage, sweep=sw, acc_sum=acc_sum, logz=logz)
    ess = lambda w: 1./np.sum(w**2)
    while True:
        if sw >= max_sweeps or (sw >= min_sweeps and acc_sum >= moves):                   # stage done: save it; next beta, resample
            if stage > 0 and not (out/f"smc_stage{stage:02d}.npz").exists():
                np.savez(out/f"smc_stage{stage:02d}.npz", x=x, c=c, beta=beta, logz=logz)   # population after its moves
            if beta >= beta_end:
                break
            cc = np.where(np.isfinite(c), c, np.inf); c0 = cc[np.isfinite(cc)].min()

            def lw(b):
                return -0.5*(b-beta)*(cc-c0)
            wts = lambda b: (lambda w: w/w.sum())(np.exp(lw(b)))
            b_new = beta_end
            if ess(wts(beta_end)) < n/2:                                                  # largest beta with ESS >= N / 2
                lo, hi = beta, beta_end
                for _ in range(60):
                    mid = 0.5*(lo+hi); lo, hi = (mid, hi) if ess(wts(mid)) >= n/2 else (lo, mid)
                b_new = lo if lo > beta else hi
            b_new = min([b_new]+[k for k in keep_betas if beta < k < b_new])                # stop exactly at the kept temperatures
            w = wts(b_new)
            logz += _logmeanexp(lw(b_new))-0.5*(b_new-beta)*c0                              # log mean exp(-(db) chi2 / 2)
            rng = np.random.default_rng([seed, stage, 0]); pos = (rng.random()+np.arange(n))/n
            idx = np.minimum(np.searchsorted(np.cumsum(w), pos), n-1)
            x, c, beta, stage, sw, acc_sum = x[idx], c[idx], b_new, stage+1, 0, 0.
            log(f"smc stage {stage}: beta {beta:.4g} (T {1/beta:.3g}), ESS {ess(w):.0f}, log Z {logz:.2f}, unique {len(np.unique(idx))}")
        # one sweep = two half-ensemble updates; each half moves with differential-evolution steps between two points of the other
        # (frozen) half, so every proposal is symmetric given its donors; proposals outside the box are rejected (not evaluated)
        rng = np.random.default_rng([seed, stage, sw+1]); g = 1.0 if (stage*max_sweeps+sw) % 10 == 9 else 2.38/np.sqrt(2*D)
        perm = rng.permutation(n); halves = (perm[:n//2], perm[n//2:]); acc = np.zeros(n, bool)
        for h, (A, B) in enumerate((halves, halves[::-1])):
            ab = np.array([rng.choice(B, 2, replace=False) for _ in A])
            prop = x[A]+g*(x[ab[:, 0]]-x[ab[:, 1]])+1e-4*rng.standard_normal((len(A), D)); ok = np.all((prop >= 0) & (prop <= 1), 1)
            cp = np.full(len(A), np.inf)
            if ok.any():
                cp[ok] = ev(prop[ok], f"smc_s{stage}_w{sw}_h{h}")
            la = lpr(prop)-lpr(x[A])-0.5*beta*(cp-c[A]); a_ = ok & np.isfinite(cp) & (np.log(rng.random(len(A))) < la)
            x[A[a_]], c[A[a_]] = prop[a_], cp[a_]; acc[A] = a_
        sw += 1; acc_sum += acc.mean()
        log(f"smc stage {stage} sweep {sw}: acceptance {acc.mean():.2f} (sum {acc_sum:.2f}), best chi2 {np.min(c):.1f}, evaluations {ev.nev}")
        _save(sp, x=x, c=c, beta=beta, stage=stage, sweep=sw, acc_sum=acc_sum, logz=logz)
    np.savez(out/"smc_final.npz", x=x, c=c, beta=beta, stage=stage, logz=logz)
    return x, c, logz


def polish(ev, out, x0, beta=1., nwalkers=40, nsteps=1000, seed=3, log=print):
    """emcee from walkers drawn (without replacement where possible) from the final SMC population x0."""
    import emcee
    out = Path(out); be = emcee.backends.HDFBackend(str(out/"polish.h5"))

    def logp(U):
        U = np.atleast_2d(U); r = np.full(len(U), -np.inf); ok = np.all((U >= 0) & (U <= 1), 1)
        if ok.any():
            r[ok] = -0.5*beta*ev(U[ok], "polish")
        return np.where(np.isfinite(r), r, -np.inf)
    s = emcee.EnsembleSampler(nwalkers, D, logp, vectorize=True, backend=be, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.StretchMove(), 0.2)])
    done = be.iteration if os.path.exists(out/"polish.h5") else 0
    if done == 0:
        rng = np.random.default_rng(seed); u0 = np.unique(x0, axis=0); p0 = u0[rng.choice(len(u0), nwalkers, replace=len(u0) < nwalkers)]
        p0 = np.clip(p0+1e-6*rng.standard_normal(p0.shape), 0., 1.); be.reset(nwalkers, D); s._random.seed(seed)
    else:
        p0 = None
    for k, _ in enumerate(s.sample(p0, iterations=nsteps-done, store=True)):
        if (done+k+1) % 50 == 0:
            tau = s.get_autocorr_time(tol=0); log(f"polish step {done+k+1}: acceptance {np.mean(s.acceptance_fraction):.2f}, tau median {np.median(tau):.0f}, max {np.max(tau):.0f}")
    tau = s.get_autocorr_time(tol=0); (out/"polish_tau.json").write_text(json.dumps(dict(steps=int(be.iteration), tau=tau.tolist(), n_over_tau=float(be.iteration/np.max(tau)))))
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--beta-end", type=float, default=1.); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--box", default="", help="name=lo,hi;... e.g. dist=5.28,5.78;vc=200,250")
    ap.add_argument("--dist-prior", default=None, help="mu,sigma [kpc]: Gaussian prior on the distance (untempered)")
    ap.add_argument("--polish-steps", type=int, default=0); ap.add_argument("--walkers", type=int, default=40)
    ap.add_argument("--fake", default=None, choices=[None, "gauss", "valley"]); a = ap.parse_args()
    for item in filter(None, a.box.split(";")):
        k, v = item.split("="); PL.LIM[k] = tuple(float(q) for q in v.split(","))
    dp = tuple(float(q) for q in a.dist_prior.split(",")) if a.dist_prior else None
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True); cfg = dict(n=a.n, beta_end=a.beta_end, seed=a.seed, box=PL.LIM, dist_prior=dp, fake=a.fake)
    cf = O/"config.json"
    if cf.exists() and json.loads(cf.read_text()) != json.loads(json.dumps(cfg)):
        raise SystemExit(f"{cf}: this run was started with another configuration ({cf.read_text()}); use a new --out")
    cf.write_text(json.dumps(cfg))
    if not a.fake:
        import jax
        jax.config.update("jax_enable_x64", True)
    ev = PL.Evaluator(a.inputs, O, fake=a.fake or False); t0 = time.time()
    x, c, logz = smc(ev, O, n=a.n, beta_end=a.beta_end, seed=a.seed, dist_prior=dp)
    if a.polish_steps:
        polish(ev, O, x, beta=a.beta_end, nwalkers=a.walkers, nsteps=a.polish_steps)
    print("done", json.dumps(dict(evaluations=ev.nev, gpu_s=ev.t, wall_s=time.time()-t0, logz=logz, best_chi2=float(np.min(c)))), flush=True)


if __name__ == "__main__":
    main()
