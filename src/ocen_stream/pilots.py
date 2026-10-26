"""Two exploration pilots for the sampler choice (2026-10-26; docs/reviews/sampler_fable_codex_20261026.md), same evaluator and box.

Box (user, 2026-10-26): angle 0-50 deg, Omega_b 20-50 km/s/kpc, A 0.6-2.0, S from 0.8 to corotation (R_CR / 4.25 kpc), q 0.65-1.25,
v_c(R0) 200-250 km/s, distance 5.28-5.78 kpc, PMs within +-4 x 0.022 mas/yr of (-3.250, -6.746). Unit cube u in [0, 1]^9; S is
0.8 + u_S (S_max - 0.8) (conditionally uniform). Target for tempering: exp(-beta chi2 / 2) x uniform(u).
  jitter   chi2 along one line through the best point (offsets 1e-6..1e-1 in u, both signs): is the jitter white on proposal scales?
  smc      Fable's pilot: 500 Sobol points, adaptive tempering (next beta: ESS = N/2) to beta = 0.1, systematic resampling, 5 sweeps
           of differential-evolution Metropolis moves per stage (gamma 2.38/sqrt(18), every 10th sweep gamma 1); budget 16000
  scan     Codex's pilot: 4000 Sobol points; local Nelder-Mead searches (60 dispersed starts, 200 evaluations each)
  diagnose the same for both pilots, on their own points: paths between their 10 best distinct minima (45 pairs x 60 points) and the
           10 best points near each minimum rescored with 8 independent score banks (Fable: budget 16000 so the SMC reaches beta 0.1;
           Codex/Fable: identical diagnostics for both)
Every evaluation (u, chi2, chi2 + log|C| term, summaries, errors, seconds) is appended to OUT/log/*.npz and reused (cache keyed by
u and bank): a restart repeats nothing; the pilots' random numbers are seeded by (pilot, stage, sweep), so a restart retraces them.
Usage: python -m ocen_stream.pilots --inputs DIR --out DIR --run jitter,smc,scan     (--fake: analytic test likelihood, no GPU)
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np

NAMES = ("angle", "omega", "amp", "size", "dist", "pmra", "pmdec", "vc", "halo_q")
LIM = dict(angle=(0., 50.), omega=(20., 50.), amp=(0.6, 2.0), dist=(5.28, 5.78), vc=(200., 250.), halo_q=(0.65, 1.25))
MU, MU_ERR, RBAR, D = (-3.250, -6.746), 0.022, 4.25, 9


def r_cr(ht, omega, vc, q):
    """corotation radius [kpc] of the axisymmetric host (tables of jaxspray.host_for, bar off): root of v_c(R)/R = omega on a 0.005-kpc
    grid, linear between nodes (vs AGAMA training_set.r_cr: 0.005 kpc at Omega 20-50)."""
    R = np.linspace(1., 25., 4801); f = np.sqrt(R*np.asarray(_dphidR(ht)(float(vc), float(q))))/R-omega
    i = np.nonzero(f < 0)[0]
    if i.size == 0 or i[0] == 0:
        raise ValueError(f"corotation not bracketed for omega {omega}")
    i = i[0]; return R[i-1]+(R[i]-R[i-1])*f[i-1]/(f[i-1]-f[i])


_DPHI = {}


def _dphidR(ht):
    """jitted dPhi/dR of the axisymmetric host at z = 0 on the r_cr grid, as a function of (v_c, q) (one compilation per ht)."""
    if id(ht) not in _DPHI:
        import jax, jax.numpy as jnp
        from . import jaxspray as J
        R = jnp.linspace(1., 25., 4801); x = jnp.stack([R, 0*R, 0*R], 1)

        def f(vc, q):
            tb = dict(J.host_for(ht, 0., 0., 0., 1., q, vc, 0.), A=0.)
            return jax.grad(lambda x: jnp.sum(J.phi_host(x, jnp.zeros(R.size), tb)))(x)[:, 0]
        _DPHI[id(ht)] = jax.jit(f)
    return _DPHI[id(ht)]


def to_params(u, ht):
    lin = lambda k, x: LIM[k][0]+(LIM[k][1]-LIM[k][0])*x
    p = dict(angle=lin("angle", u[0]), omega=lin("omega", u[1]), amp=lin("amp", u[2]), dist=lin("dist", u[4]),
             pmra=MU[0]+MU_ERR*(8.*u[5]-4.), pmdec=MU[1]+MU_ERR*(8.*u[6]-4.), vc=lin("vc", u[7]), halo_q=lin("halo_q", u[8]))
    p["size"] = 0.8+u[3]*(r_cr(ht, p["omega"], p["vc"], p["halo_q"])/RBAR-0.8)
    return {k: float(p[k]) for k in NAMES}


def to_unit(p, ht):
    """inverse of to_params (for starting points given in physical parameters)."""
    f = lambda k: (p[k]-LIM[k][0])/(LIM[k][1]-LIM[k][0])
    smax = r_cr(ht, p["omega"], p["vc"], p["halo_q"])/RBAR
    return np.array([f("angle"), f("omega"), f("amp"), (p["size"]-0.8)/(smax-0.8), f("dist"), (p["pmra"]-MU[0])/MU_ERR/8+0.5,
                     (p["pmdec"]-MU[1])/MU_ERR/8+0.5, f("vc"), f("halo_q")])


def provenance(inputs, fake):
    """hashes of the inputs and of the evaluator's sources (a log is reused only with the same provenance)."""
    import hashlib
    h = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]; src = Path(__file__).parent
    r = dict(fake=bool(fake), **{f.name: h(f) for f in [src/"pilots.py", src/"gpupipe.py", src/"jaxscore.py", src/"jaxmodel.py",
                                                       src/"jaxspray.py", src/"cuspray.py", src/"cspray.py", src/"csrc"/"spray.c"]})
    if not fake:
        r.update({f: h(Path(inputs)/f) for f in ("host_tables.npz", "spray_inputs.npz", "score_inputs.npz", "points.json")})
    return r


class Log:
    """append-only evaluation log in OUT/log (one npz per batch, written atomically) + an in-memory cache keyed by (u bytes, bank).
    OUT/manifest.json holds the provenance; a log written with other inputs or sources is refused."""
    def __init__(self, out, prov=None):
        self.d = Path(out)/"log"; self.d.mkdir(parents=True, exist_ok=True); self.cache = {}; self.n = 0; mf = Path(out)/"manifest.json"
        if prov is not None:
            if mf.exists() and json.loads(mf.read_text()) != prov:
                raise RuntimeError(f"{mf}: log written with other inputs/sources; use a new --out")
            mf.write_text(json.dumps(prov, indent=1))
        for f in sorted(self.d.glob("*.npz")):
            z = np.load(f)
            for k in range(len(z["chi2"])):
                self.cache[(z["u"][k].tobytes(), int(z["bank"][k]))] = (float(z["chi2"][k]), float(z["chi2ld"][k]))
            self.n += 1

    def write(self, rec):
        tmp = self.d/f".{self.n:06d}.tmp.npz"; np.savez(tmp, **rec); os.replace(tmp, self.d/f"{self.n:06d}.npz"); self.n += 1
        for k in range(len(rec["chi2"])):
            self.cache[(rec["u"][k].tobytes(), int(rec["bank"][k]))] = (float(rec["chi2"][k]), float(rec["chi2ld"][k]))

    def records(self, prefix, bank=7, exclude="_diag"):
        """all logged (u, chi2, chi2ld, seconds) with tags starting with prefix (and not containing exclude), in log order."""
        U, C, L, S = [], [], [], []
        for f in sorted(self.d.glob("[0-9]*.npz")):
            z = np.load(f); k = np.array([str(t).startswith(prefix) and exclude not in str(t) for t in z["tag"]]) & (z["bank"] == bank)
            U.append(z["u"][k]); C.append(z["chi2"][k]); L.append(z["chi2ld"][k]); S.append(z["s"][k])
        return tuple(np.concatenate(x) if x else np.zeros(0) for x in (U, C, L, S))


class Evaluator:
    """chi2 of unit-cube points (B, 9) in batches of BATCH through gpupipe (or an analytic fake); every evaluation logged."""
    BATCH = 20

    def __init__(self, inputs, out, fake=False):
        self.log, self.fake, self.nev, self.t = Log(out, provenance(inputs, fake)), fake, 0, 0.
        if fake:
            return
        import jax, jax.numpy as jnp
        from . import jaxspray as J, jaxmodel as JM, jaxscore as JS
        I = Path(inputs); self.ht, self.inp, self.si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
        pts = json.loads((I/"points.json").read_text()); off = pts.pop("_epoch_offset"); self.trel, self.keep = JM.schedule(self.inp, off)
        self.points = {k: {q: float(v) for q, v in p.items()} for k, p in pts.items()}; self.banks = {}
        self.obs = jax.jit(lambda y: JM.observables(y, self.inp)); self.c2 = jax.jit(lambda m, e: (JS.chi2(m, e, self.si), JS.chi2(m, e, self.si, logdet=True)))

    def bank(self, seed):
        import jax.numpy as jnp
        from . import jaxscore as JS
        if seed not in self.banks:
            self.banks[seed] = {k: jnp.asarray(v) for k, v in JS.make_bank(len(self.keep), 8, seed).items()}
        return self.banks[seed]

    def _models(self, U, seed):
        """chi2, chi2 + log-det term, summaries m, errors e for the rows of U (one launch for the batch). A model whose parameters,
        release or particles fail gets chi2 = inf; the other models of the batch are unaffected."""
        if self.fake:
            return fake_chi2(U)
        import jax.numpy as jnp
        from concurrent.futures import ThreadPoolExecutor
        from . import jaxmodel as JM, jaxscore as JS, cuspray
        bk = self.bank(seed); n = len(U); c2, c2l = np.full(n, np.inf), np.full(n, np.inf); M, E = np.full((n, 25), np.nan), np.full((n, 25), np.nan)

        def prep(u):
            try:
                return JM.release_prepare(to_params(u, self.ht), self.ht, self.inp, self.trel)
            except Exception:
                return None
        with ThreadPoolExecutor(os.cpu_count()) as ex:                                    # CPU part in threads (as gpupipe)
            pre = list(ex.map(prep, U))
        ok, rel = [], []
        for k, pr in enumerate(pre):
            if pr is None:
                continue
            try:
                rel.append(JM.release_finish(*pr, self.trel, self.inp, self.keep)); ok.append(k)
            except Exception:
                pass
        if not ok:
            return c2, c2l, M, E
        t0 = np.asarray(rel[0][2]); out = cuspray.integrate([(np.asarray(r[1]), t0, r[0]) for r in rel], JM.SPRAY_ACC)
        age = jnp.asarray((float(self.inp["T"])-t0)*JM.T_MYR)
        for k, (y, chi, natt) in zip(ok, out):
            if np.any(natt < 0):
                continue
            m, e = JS.measure_capped(self.obs(jnp.asarray(y)), jnp.asarray(chi*JM.T_MYR), age, self.si, bk); a, b = self.c2(m, e)
            c2[k], c2l[k], M[k], E[k] = float(a), float(b), np.asarray(m), np.asarray(e)
        return c2, c2l, M, E

    def __call__(self, U, tag, seed=7):
        """chi2 (len(U),) of unit points U; cached points are not re-evaluated."""
        U = np.ascontiguousarray(np.clip(np.atleast_2d(np.asarray(U, np.float64)), 0., 1.)); res = np.full(len(U), np.nan)
        todo = []
        for k, u in enumerate(U):
            c = self.log.cache.get((u.tobytes(), seed))
            if c is None:
                todo.append(k)
            else:
                res[k] = c[0]
        for a in range(0, len(todo), self.BATCH):
            idx = todo[a:a+self.BATCH]; t1 = time.perf_counter(); c2, c2l, M, E = self._models(U[idx], seed); dt = time.perf_counter()-t1
            self.log.write(dict(u=U[idx], chi2=c2, chi2ld=c2l, m=M, e=E, bank=np.full(len(idx), seed), tag=np.array([tag]*len(idx)), s=np.full(len(idx), dt/len(idx))))
            res[idx] = c2; self.nev += len(idx); self.t += dt
        return res


def fake_chi2(U):
    """analytic test likelihood: two separated quadratic basins (chi2 166 and 351) + deterministic jitter (+-2 near the minimum)."""
    U = np.atleast_2d(U); c1, c2 = np.full(D, 0.4), np.array([0.66, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    q1 = 166+np.sum(((U-c1)/0.03)**2, 1); q2 = 351+np.sum(((U-c2)/0.03)**2, 1); c = np.minimum(q1, q2)
    jit = (1+0.05*(c-166))*np.sin(1e5*U@np.arange(1., D+1)); c = c+jit
    return c, c+1., np.zeros((len(U), 25)), np.ones((len(U), 25))


def jitter(ev, u0, out):
    """chi2 along a random direction through u0 at offsets +-logspace(-6, -1, 50) (100 evaluations)."""
    rng = np.random.default_rng(11); d = rng.standard_normal(D); d /= np.linalg.norm(d)
    off = np.concatenate((-np.logspace(-6, -1, 50)[::-1], np.logspace(-6, -1, 50)))
    c = ev(np.clip(u0+off[:, None]*d, 0, 1), "jitter"); c0 = ev(u0[None], "jitter")[0]
    np.savez(Path(out)/"jitter.npz", off=off, chi2=c, chi2_0=c0, u0=u0, d=d); return off, c, c0


def _state_save(path, **s):
    tmp = Path(str(path)+".tmp.npz"); np.savez(tmp, **s); os.replace(tmp, path)


def smc(ev, out, n=500, beta_end=0.1, sweeps=5, budget=16000, seed=1):
    """Fable's pilot (adaptive tempering SMC with DE moves). State saved after every sweep (OUT/smc_state.npz); history of populations
    per stage in OUT/smc_stages.npz."""
    from scipy.stats import qmc
    sp = Path(out)/"smc_state.npz"
    if sp.exists():
        z = np.load(sp); x, c, beta, stage, sw, nev = z["x"], z["c"], float(z["beta"]), int(z["stage"]), int(z["sweep"]), int(z["nev"])
    else:
        x = qmc.Sobol(D, scramble=True, seed=seed).random(n); c = ev(x, "smc_init"); beta, stage, sw, nev = 0., 0, sweeps, n
    ess = lambda w: 1./np.sum(w**2)
    while nev < budget:
        if sw >= sweeps:                                                                    # new stage: next beta, resample
            if beta >= beta_end:
                break
            fin = np.isfinite(c); cc = np.where(fin, c, np.inf)

            def wts(b):
                lw = -0.5*(b-beta)*(cc-cc[fin].min()); w = np.exp(lw); return w/w.sum()
            lo, hi = beta, beta_end
            if ess(wts(hi)) < n/2:
                for _ in range(60):
                    mid = 0.5*(lo+hi); lo, hi = (mid, hi) if ess(wts(mid)) >= n/2 else (lo, mid)
            b_new = hi if ess(wts(hi)) >= n/2 else lo
            if b_new <= beta:
                b_new = min(beta_end, beta+1e-6)
            w = wts(b_new); rng = np.random.default_rng([seed, stage, 0])
            pos = (rng.random()+np.arange(n))/n; idx = np.minimum(np.searchsorted(np.cumsum(w), pos), n-1)
            x, c, beta, stage, sw = x[idx], c[idx], b_new, stage+1, 0
            np.savez(Path(out)/f"smc_stage{stage:02d}.npz", x=x, c=c, beta=beta, ess=ess(w))
        rng = np.random.default_rng([seed, stage, sw+1]); g = 1.0 if (stage*sweeps+sw) % 10 == 9 else 2.38/np.sqrt(2*D)
        ab = np.array([(lambda q: q+(q >= i))(rng.choice(n-1, 2, replace=False)) for i in range(n)])     # two others per particle
        prop = x+g*(x[ab[:, 0]]-x[ab[:, 1]])+1e-4*rng.standard_normal((n, D)); ok = np.all((prop >= 0) & (prop <= 1), 1)
        cp = np.full(n, np.inf); cp[ok] = ev(prop[ok], f"smc_s{stage}_w{sw}"); nev += int(ok.sum())
        acc = np.log(rng.random(n)) < -0.5*beta*(cp-c); acc &= ok & np.isfinite(cp)
        x, c = np.where(acc[:, None], prop, x), np.where(acc, cp, c); sw += 1
        print(f"smc stage {stage} beta {beta:.4g} sweep {sw}: acceptance {acc.mean():.2f}, best chi2 {np.min(c):.1f}, evaluations {ev.nev}", flush=True)
        _state_save(sp, x=x, c=c, beta=beta, stage=stage, sweep=sw, nev=nev)
    np.savez(Path(out)/"smc_final.npz", x=x, c=c, beta=beta, stage=stage, nev=nev)
    return x, c, beta


class NM:
    """bounded Nelder-Mead (reflection 1, expansion 2, contraction 0.5, shrink 0.5; points clipped to [0, 1]) as an ask/tell state machine."""
    def __init__(self, x0, step=0.1):
        self.S = [np.clip(x0, 0, 1)]+[np.clip(x0+step*np.eye(D)[i]*(1 if x0[i] < 0.9 else -1), 0, 1) for i in range(D)]
        self.F = [None]*(D+1); self.pend = list(range(D+1)); self.phase = "init"; self.n = 0

    def ask(self):
        if self.phase in ("init", "shrink"):
            return [self.S[i] for i in self.pend]
        o = np.argsort(self.F); self.S = [self.S[i] for i in o]; self.F = [self.F[i] for i in o]
        self.cen = np.mean(self.S[:-1], 0)
        if self.phase == "reflect":
            self.xr = np.clip(self.cen+(self.cen-self.S[-1]), 0, 1); return [self.xr]
        if self.phase == "expand":
            return [np.clip(self.cen+2*(self.cen-self.S[-1]), 0, 1)]
        return [self._xc()]                                                                 # contract (outside / inside)

    def _xc(self):
        return np.clip(self.cen+0.5*(self.xr-self.cen), 0, 1) if self.phase == "contract_out" else np.clip(self.cen+0.5*(self.S[-1]-self.cen), 0, 1)

    def tell(self, f):
        self.n += len(f)
        if self.phase in ("init", "shrink"):
            for i, v in zip(self.pend, f):
                self.F[i] = v
            self.pend = []; self.phase = "reflect"; return
        if self.phase == "reflect":
            self.fr = f[0]
            if self.fr < self.F[0]:
                self.phase = "expand"
            elif self.fr < self.F[-2]:
                self.S[-1], self.F[-1] = self.xr, self.fr
            else:
                self.phase = "contract_out" if self.fr < self.F[-1] else "contract_in"
            return
        if self.phase == "expand":
            xe = np.clip(self.cen+2*(self.cen-self.S[-1]), 0, 1)
            self.S[-1], self.F[-1] = (xe, f[0]) if f[0] < self.fr else (self.xr, self.fr); self.phase = "reflect"; return
        xc = self._xc(); ref = self.fr if self.phase == "contract_out" else self.F[-1]          # contract: accept if better than ref
        if f[0] <= ref:
            self.S[-1], self.F[-1] = xc, f[0]; self.phase = "reflect"
        else:
            self.S = [self.S[0]]+[self.S[0]+0.5*(s-self.S[0]) for s in self.S[1:]]; self.pend = list(range(1, D+1)); self.phase = "shrink"

    def best(self):
        i = int(np.nanargmin([np.inf if v is None else v for v in self.F])); return self.S[i], self.F[i]

    def size(self):
        return max(np.max(np.abs(s-self.S[0])) for s in self.S)


def scan(ev, out, n_sobol=4000, n_starts=60, nm_budget=200, seed=2):
    """Codex's pilot: Sobol scan and dispersed local Nelder-Mead searches (then the shared diagnostics)."""
    from scipy.stats import qmc
    X = qmc.Sobol(D, scramble=True, seed=seed).random(n_sobol); C = ev(X, "scan_sobol")
    print(f"scan: Sobol {n_sobol} done, best chi2 {np.nanmin(C):.1f}, evaluations {ev.nev}", flush=True)
    order = np.argsort(np.where(np.isfinite(C), C, np.inf)); starts = []
    for i in order:                                                                         # dispersed starts: >= 0.25 apart in u
        if all(np.max(np.abs(X[i]-X[j])) >= 0.25 for j in starts):
            starts.append(i)
        if len(starts) == n_starts:
            break
    chains = [NM(X[i]) for i in starts]
    while True:
        act = [ch for ch in chains if ch.n < nm_budget and ch.size() > 1e-3]
        if not act:
            break
        asks = [ch.ask() for ch in act]; flat = np.array([u for a in asks for u in a]); f = ev(flat, "scan_nm"); k = 0
        for ch, a in zip(act, asks):
            ch.tell(list(f[k:k+len(a)])); k += len(a)
    ends = sorted([ch.best() for ch in chains], key=lambda b: b[1])
    np.savez(Path(out)/"scan_nm.npz", x=np.array([e[0] for e in ends]), c=np.array([e[1] for e in ends]), starts=X[starts], cstart=C[starts])
    print(f"scan: local searches done, best chi2 {ends[0][1]:.1f}, evaluations {ev.nev}", flush=True)


def diagnose(ev, out, name, n_min=10, n_path=60, n_near=10, banks=tuple(range(101, 109))):
    """the same diagnostics for each pilot, from its own logged points (tags name*): up to n_min local minima (points with >= 5 points of
    the pilot and none better within 0.1 in u, best first), chi2 along the straight paths between them (45 pairs x n_path), and the n_near best
    points near each minimum rescored with independent score banks. Results in OUT/<name>_diag.npz."""
    U, C, _, _ = ev.log.records(name); U, i = np.unique(U, axis=0, return_index=True); C = np.where(np.isfinite(C[i]), C[i], np.inf)
    o = np.argsort(C); mins = []
    for k in o:
        nb = np.max(np.abs(U-U[k]), 1) < 0.1
        if np.isfinite(C[k]) and nb.sum() >= 5 and C[k] <= C[nb].min():                    # explored neighbourhood, nothing better in it
            mins.append(k)
        if len(mins) == n_min:
            break
    t = np.linspace(0, 1, n_path+2)[1:-1]; paths = {}
    for i in range(len(mins)):
        for j in range(i+1, len(mins)):
            P = U[mins[i]][None]+t[:, None]*(U[mins[j]]-U[mins[i]])[None]; paths[f"p{i}_{j}"] = ev(P, f"{name}_diag_path_{i}_{j}")
    near = []
    for m in mins:
        d = np.max(np.abs(U-U[m]), 1); k = [q for q in o if d[q] < 0.1][:n_near]; near += k
    R = np.array([ev(U[near], f"{name}_diag_rescore_{b}", seed=b) for b in banks])
    np.savez(Path(out)/f"{name}_diag.npz", t=t, umin=U[mins], cmin=C[mins], unear=U[near], cnear=C[near], rescore=R, banks=np.array(banks), **paths)
    print(f"{name} diagnostics: {len(mins)} minima, chi2 {np.round(C[mins], 1).tolist()}, evaluations {ev.nev}", flush=True)


def cost(ev, name):
    """evaluations and A100 seconds of a pilot from the log (all tags name*, including its diagnostics; across restarts)."""
    U, C, _, S = ev.log.records(name, exclude="\0"); R = [ev.log.records(name, bank=b, exclude="\0") for b in range(101, 109)]
    return dict(evaluations=int(len(C)+sum(len(r[1]) for r in R)), gpu_s=float(S.sum()+sum(r[3].sum() for r in R)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="jitter,smc,scan"); ap.add_argument("--fake", action="store_true"); a = ap.parse_args()
    if not a.fake:
        import jax
        jax.config.update("jax_enable_x64", True)
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True); ev = Evaluator(a.inputs, O, fake=a.fake); t0 = time.time()
    for job in a.run.split(","):
        n0, s0, w0 = ev.nev, ev.t, time.time()
        if job == "jitter":
            u0 = np.full(D, 0.4) if a.fake else to_unit(ev.points["c0_final"], ev.ht)
            off, c, c0 = jitter(ev, u0, O); print("jitter: chi2_0", c0, "chi2 - chi2_0 at |offset| 1e-6, 1e-4, 1e-2:",
                                                  [float(np.median(np.abs(c[np.isclose(np.abs(off), v, rtol=0.2)]-c0))) for v in (1e-6, 1e-4, 1e-2)], flush=True)
        elif job == "smc":
            smc(ev, O); diagnose(ev, O, "smc")
        elif job == "scan":
            scan(ev, O); diagnose(ev, O, "scan")
        rec = dict(job=job, this_session=dict(evaluations=ev.nev-n0, gpu_s=ev.t-s0, wall_s=time.time()-w0), **(cost(ev, job) if job != "jitter" else {}))
        print("done", rec, flush=True); (O/f"done_{job}.json").write_text(json.dumps(rec))

if __name__ == "__main__":
    main()
