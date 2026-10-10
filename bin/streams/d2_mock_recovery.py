#!/usr/bin/env python3
"""D2 (docs/STREAM_DIAGNOSTIC_PLAN.md): does the method itself pull the bar angle low? Mock-injection / recovery.
Truths: sprays at (35.5, 28, 1.2, 5.6) and (34.5, 20, 1.2, 5.6), 32000 release epochs, seed 7 (independent of the candidates).
Mock catalogue (per truth, NMOCK times): each real member keeps its phi1 (u), Gaia PM covariance and v_los availability/error; its
(dphi2, pmra, pmdec, v_los) are drawn from the truth's conditional at u: a truth particle j is picked with probability
prop. to w_j N(u - U_j; 0.5 deg) (w_j = chi-bin weight, same cuts as score_conditional), then kernel noise (0.5 deg, 0.2 mas/yr,
5 km/s) and the member's measurement errors are added. No contaminants.
Candidates: grid4 sprays (8000 epochs, seed 1) at d = 5.6, amp 1.2 and 1.4, all Omega_b (7) and angles (4): 56 models.
Output: results/plot_data/d2_mock_recovery.json (per mock: lnL of all candidates); prints the recovered-angle distribution and
Delta lnL(best 16 - best 28) for each truth. Usage: python bin/streams/d2_mock_recovery.py [--nmock 100] [--nproc 8] [--time]
"""
import json, sys, itertools, time
from multiprocessing import Pool
import numpy as np
from grid4_common import ROOT, make_spray, project_gc, GC, du, d, cov, win, score_conditional

NMOCK = int(sys.argv[sys.argv.index("--nmock")+1]) if "--nmock" in sys.argv else 100
NPROC = int(sys.argv[sys.argv.index("--nproc")+1]) if "--nproc" in sys.argv else 8
TRUTHS = [(35.5, 28., 1.2, 5.6), (34.5, 20., 1.2, 5.6)]
CANDS = [(om, an, am, 5.6) for om, an, am in itertools.product([33., 34., 34.5, 35., 35.5, 36., 37.], [16., 20., 24., 28.], [1.2, 1.4])]
hasv = np.isfinite(d["v"]); n = len(du)


def prep(m):
    mu, mx = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700)
    edges = np.arange(0., m["chi"][k].max()+5., 5.)
    bins = [np.where(k & (m["chi"] >= e) & (m["chi"] < e+5.))[0] for e in edges]; bins = [b for b in bins if len(b) >= 20]; K = len(bins)
    idx = np.concatenate(bins); wt = np.concatenate([np.full(len(b), 1./(K*len(b))) for b in bins])
    return dict(u=mu[idx], w=np.column_stack((mx[idx], m["pmra"][idx], m["pmdec"][idx])), v=m["vlos"][idx], wt=wt)


def make_mock(T, rng):
    W = np.zeros((n, 3)); V = np.full(n, np.nan)
    for i0 in range(0, n, 500):
        s = slice(i0, min(n, i0+500))
        p = T["wt"]*np.exp(-0.5*((du[s, None]-T["u"])/0.5)**2); p /= p.sum(1, keepdims=True)
        c = np.cumsum(p, 1); j = (c < rng.random(c.shape[0])[:, None]).sum(1)
        W[s] = T["w"][j]; V[s] = T["v"][j]
    W[:, 0] += rng.normal(0, 0.5, n); W[:, 1:] += rng.normal(0, 0.2, (n, 2))
    W[:, 1:] += np.array([rng.multivariate_normal([0, 0], cov[i]) for i in range(n)])
    V = np.where(hasv, V+rng.normal(0, 5., n)+rng.normal(0, 1., n)*np.nan_to_num(d["e_v"]), np.nan)
    return W, V


CM = None


def init():
    global CM
    CM = [dict(np.load(ROOT/"results/streams/spray_grid4/om{:g}_an{:g}_am{:g}_d{:g}.npz".format(*c))) for c in CANDS]
    for m in CM:
        m["u"], m["x"] = project_gc(m["l"], m["b"], GC)


def score_mock(args):
    W, V = args
    return [score_conditional(du, W, V, d["e_v"], cov, m["u"], np.column_stack((m["x"], m["pmra"], m["pmdec"])), m["vlos"], m["chi"], m["age"],
                              u_window=win)["total"] for m in CM]


if __name__ == "__main__":
    out = dict(cands=CANDS, truths={})
    for tr in TRUTHS:
        f = ROOT/"results/streams/d2/om{:g}_an{:g}_am{:g}_d{:g}_n32000_s7.npz".format(*tr)
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **make_spray(*tr, nrel=32000, seed=7))
        T = prep(dict(np.load(f))); rng = np.random.default_rng(11)
        mocks = [make_mock(T, rng) for _ in range(NMOCK if "--time" not in sys.argv else 2)]
        t0 = time.time()
        with Pool(NPROC, initializer=init) as pool:
            L = np.array(pool.map(score_mock, mocks))
        print(f"truth {tr}: {len(mocks)} mocks scored in {time.time()-t0:.0f} s", flush=True)
        if "--time" in sys.argv:
            continue
        C = np.array(CANDS); best = C[np.argmax(L, 1)]
        ang = {f"{a:g}": float(np.mean(best[:, 1] == a)) for a in (16., 20., 24., 28.)}
        b16 = L[:, C[:, 1] == 16].max(1); b28 = L[:, C[:, 1] == 28].max(1); b20 = L[:, C[:, 1] == 20].max(1)
        print(f"  recovered angle fractions {ang}; recovered Omega_b {dict(zip(*np.unique(best[:, 0], return_counts=True)))}")
        print(f"  Delta lnL(best16 - best28): median {np.median(b16-b28):.0f}, 16-84% {np.percentile(b16-b28, [16, 84]).round(0)}; "
              f"(best16 - best20): median {np.median(b16-b20):.0f}", flush=True)
        out["truths"][str(tr)] = dict(lnL=L.tolist(), angle_frac=ang)
    if "--time" not in sys.argv:
        (ROOT/"results/plot_data/d2_mock_recovery.json").write_text(json.dumps(out))
