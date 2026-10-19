#!/usr/bin/env python3
"""Angle-stratified multi-start trust-region search for the 9-D stream likelihood, one wave at a time (2026-10-18; plan
reviewed by Codex, docs/reviews/codex_search_plan_2026-10-18.md).

Observations for a catalogue (real data or a mock): the stored sprays re-scored against it (rescore_batch.py output,
--base) plus every earlier wave of this search (stream_batch.py batch dirs scored against the same catalogue; checked).
Emulator: replicate means, replicate-fitted noise surface, constant-mean GP (emulator.py), refitted every wave.
Search: bar angle in 5 bins of 5 deg (unit coordinate 0..1 in 0.2 steps); per bin 2 trust regions (TR): box of side L
(unit cube; angle limited to the bin) around a centre. Wave 1 centres: per bin the highest mu + 2 sd candidate and the
candidate among the bin's top 300 farthest from it in the other 8 coordinates. After each wave a TR succeeds if its best
new replicate mean exceeds its incumbent (best mean inside the box before the wave) by more than 2 noise sd: the centre
moves there and L grows x1.5 (max 0.8); otherwise L halves; below 0.05 the TR restarts at the bin's best mu + 2 sd
candidate at least 0.3 from the other TR (L 0.4). Screening is reversible: nothing is ever excluded from the global points.
Per wave (default 100 sprays): 6 points per TR (60; greedy mu + 2 sd inside the box, sd updated with fixed-kernel
believer fantasies), 20 whole-domain maximin points against all existing locations, 10 random held-out validation
points (never fitted), and 2 independent seeds at each bin's best measured location (10; replicates of that location).
Writes OUTDIR/{jobs.json, catalogue.json, state.json, design_info.json}; run with stream_batch.py run --outdir OUTDIR.
Usage: OMP_NUM_THREADS=4 PYTHONPATH=src python bin/streams/search_wave.py --catalogue CAT --box wide --base RESCORE_DIR \
          --outdir OUT_WAVEk [--prev WAVE_DIR ...] --seed-base 700000
"""
import argparse, hashlib, json, sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
import emulator_design as ED      # noqa: E402
from ocen_dm.streams.emulator import group_replicates, MeanEmulator   # noqa: E402

HELD = "validation (held out)"
NB, NTR, NLOC, NGLOB, NVAL, NCHK = 5, 2, 6, 20, 10, 2
L0, LMAX, LMIN, KAPPA = 0.4, 0.8, 0.05, 2.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def observations(base, prev, csha, prior=False):
    rows = []
    R = json.loads((base/"rescore.json").read_text())
    if R["manifest"]["catalogue_sha256"] != csha:
        raise SystemExit(f"{base}: re-scored against a different catalogue")
    from profile_nm import ln_prior
    lp = (lambda p: ln_prior(p)) if prior else (lambda p: 0.)
    rows += [dict(key=r["key"], unit=r["unit"], y=r["total_jk"]+lp(r["params"]), v=r["var_jk"], purpose=r["purpose"]) for r in R["rows"]]
    for w in prev:
        man = json.loads((w/"manifest.json").read_text())
        if man.get("catalogue", {}).get("sha256") != csha:
            raise SystemExit(f"{w}: scored against a different catalogue")
        for f in (w/"runs").glob("*.json"):
            if f.name.endswith(".error.json"):
                continue
            j = json.loads(f.read_text())
            rows.append(dict(key=j["key"], unit=j["unit"], y=j["total_jk"]+lp(j["params"]), v=j["var_jk"], purpose=j["purpose"]))
    return rows


def fit(rows):
    fr = [r for r in rows if r["purpose"] != HELD]
    g = group_replicates(np.array([r["unit"] for r in fr], float), np.array([r["y"] for r in fr]), np.array([r["v"] for r in fr]),
                         np.array([r["key"] for r in fr]))
    s2 = ED.noise_var(g); em = MeanEmulator(n_restarts=3).fit(g["X"], g["ybar"], s2/g["r"])
    return g, s2, em


def box(c, L, b):
    lo = np.clip(c-L/2, 0, 1); hi = np.clip(c+L/2, 0, 1)
    lo[0], hi[0] = max(lo[0], b/NB), min(hi[0], (b+1)/NB)
    if hi[0] <= lo[0]:
        lo[0], hi[0] = b/NB, (b+1)/NB
    return lo, hi


def in_box(X, lo, hi):
    return np.all((X >= lo) & (X <= hi), axis=1)


def init_centres(em, b, rng, avoid=None):
    from scipy.stats import qmc
    U = qmc.Sobol(9, scramble=True, seed=int(rng.integers(1 << 30))).random_base2(13); U[:, 0] = (b+U[:, 0])/NB
    mu, sd = em.predict(U); o = np.argsort(-(mu+KAPPA*sd))[:300]; T = U[o]
    if avoid is not None:
        far = np.linalg.norm(T[:, 1:]-avoid[1:], axis=1) >= 0.3
        T = T[far] if far.any() else T
        return T[0]
    c1 = T[0]; c2 = T[int(np.argmax(np.linalg.norm(T[:, 1:]-c1[1:], axis=1)))]
    return c1, c2


def greedy(em, g, s2, C, n, Xextra):
    """n distinct candidates from C by greedy mu + kappa sd, sd updated by fixed-kernel believer fantasies."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    mu, sd = em.predict(C); pick = []
    Xb = np.vstack((g["X"], Xextra)) if len(Xextra) else g["X"].copy()
    yb = np.append((g["ybar"]-em.ymu)/em.ysd, (em.predict(Xextra, return_std=False)-em.ymu)/em.ysd if len(Xextra) else [])
    nb = np.append((s2/g["r"])/em.ysd**2, np.full(len(Xextra), np.median(s2)/em.ysd**2))
    for _ in range(n):
        a = mu+KAPPA*sd; a[pick] = -np.inf; i = int(np.argmax(a)); pick.append(i)
        Xb = np.vstack((Xb, C[i])); yb = np.append(yb, (mu[i]-em.ymu)/em.ysd); nb = np.append(nb, np.median(s2)/em.ysd**2)
        sd = em.ysd*GaussianProcessRegressor(em.gp.kernel_, alpha=nb, optimizer=None).fit(Xb, yb).predict(C, return_std=True)[1]
    return [C[i] for i in pick], [float(mu[i]) for i in pick]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True); ap.add_argument("--box", default="wide", choices=("wide", "data"))
    ap.add_argument("--base", required=True); ap.add_argument("--outdir", required=True); ap.add_argument("--prev", nargs="*", default=[])
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--prior", action="store_true", help="add the distance and PM measurement priors to the scores")
    a = ap.parse_args()
    P = lambda x: Path(x) if Path(x).is_absolute() else ROOT/x
    od, base, prev = P(a.outdir), P(a.base), [P(x) for x in a.prev]
    if (od/"jobs.json").exists():
        raise SystemExit("jobs.json exists; refusing to overwrite")
    csha = "real" if a.catalogue == "real" else sha(P(a.catalogue))
    rows = observations(base, prev, csha, a.prior); g, s2, em = fit(rows)
    rng = np.random.default_rng(a.seed_base); wave = len(prev)+1
    noise_sd = float(np.sqrt(np.median(s2)))
    # trust-region state
    if prev:
        state = json.loads((prev[-1]/"state.json").read_text())
        last = [json.loads(f.read_text()) for f in (prev[-1]/"runs").glob("*.json") if not f.name.endswith(".error.json")]
        for tr in state["trs"]:
            lo, hi = np.array(tr["lo"]), np.array(tr["hi"])
            new = [r for r in last if r["purpose"] == f"local TR{tr['id']}"]
            k = in_box(g["X"], lo, hi)
            if new:
                keys = {r["key"] for r in new}; idx = [i for i, kk in enumerate(g["keys"]) if kk in keys]
                bn = max(idx, key=lambda i: g["ybar"][i]); bnew = g["ybar"][bn]
                if bnew > tr["incumbent"]+2*noise_sd:
                    tr["centre"] = g["X"][bn].tolist(); tr["L"] = min(LMAX, 1.5*tr["L"]); tr["status"] = "success"
                else:
                    tr["L"] *= 0.5; tr["status"] = "failure"
            if tr["L"] < LMIN:
                other = [t for t in state["trs"] if t["bin"] == tr["bin"] and t["id"] != tr["id"]][0]
                tr["centre"] = init_centres(em, tr["bin"], rng, avoid=np.array(other["centre"])).tolist(); tr["L"] = L0
                tr["status"] = "restart"
    else:
        state = dict(trs=[])
        for b in range(NB):
            c1, c2 = init_centres(em, b, rng)
            for j, c in enumerate((c1, c2)):
                state["trs"].append(dict(id=f"{b}{j}", bin=b, centre=c.tolist(), L=L0, status="init"))
    # incumbents and proposals
    from scipy.stats import qmc
    jobs, Xnew = [], np.zeros((0, 9))
    for tr in state["trs"]:
        lo, hi = box(np.array(tr["centre"]), tr["L"], tr["bin"]); tr["lo"], tr["hi"] = lo.tolist(), hi.tolist()
        k = in_box(g["X"], lo, hi); tr["incumbent"] = float(g["ybar"][k].max()) if k.any() else -np.inf
        C = lo+(hi-lo)*qmc.Sobol(9, scramble=True, seed=int(rng.integers(1 << 30))).random_base2(11)
        pts, mus = greedy(em, g, s2, C, NLOC, Xnew)
        for x, m in zip(pts, mus):
            jobs.append(dict(key=f"w{wave}_{tr['id']}_{len(jobs)}", unit=x.tolist(), purpose=f"local TR{tr['id']}", mu=m))
        Xnew = np.vstack((Xnew, pts))
    allX = np.vstack((g["X"], Xnew)); C = qmc.Sobol(9, scramble=True, seed=int(rng.integers(1 << 30))).random_base2(14)
    d = np.min(np.linalg.norm(C[:, None, :]-allX[None, ::1, :], axis=2), axis=1)
    for _ in range(NGLOB):                      # whole-domain maximin (screening never excludes these)
        i = int(np.argmax(d)); jobs.append(dict(key=f"w{wave}_glob_{len(jobs)}", unit=C[i].tolist(), purpose="global maximin"))
        d = np.minimum(d, np.linalg.norm(C-C[i], axis=1))
    V = qmc.Sobol(9, scramble=True, seed=int(rng.integers(1 << 30))).random(NVAL)
    jobs += [dict(key=f"w{wave}_val_{len(jobs)+i}", unit=v.tolist(), purpose=HELD) for i, v in enumerate(V)]
    for b in range(NB):                         # independent checks at each bin's best measured location
        k = (g["X"][:, 0] >= b/NB) & (g["X"][:, 0] <= (b+1)/NB)
        i = np.flatnonzero(k)[int(np.argmax(g["ybar"][k]))]
        jobs += [dict(key=g["keys"][i], unit=g["X"][i].tolist(), purpose=f"check bin{b}", measured=float(g["ybar"][i]))
                 for _ in range(NCHK)]
    out = []
    for n, j in enumerate(jobs):
        seed = a.seed_base+1000*wave+n
        out.append(dict(j, run_id=f"w{wave}_{n:03d}", point=-1, replicate=n, seed=seed, nrel=256000,
                        epoch_offset=float(np.random.default_rng([seed, 777]).uniform()), params=ED.unit_to_params(np.array(j["unit"]))))
    od.mkdir(parents=True, exist_ok=True)
    (od/"jobs.json").write_text(json.dumps(out, indent=1))
    (od/"catalogue.json").write_text(json.dumps(dict(catalogue=a.catalogue, box=a.box), indent=1))
    (od/"state.json").write_text(json.dumps(state, indent=1))
    best = {b: float(np.max(g["ybar"][(g["X"][:, 0] >= b/NB) & (g["X"][:, 0] <= (b+1)/NB)])) for b in range(NB)}
    (od/"design_info.json").write_text(json.dumps(dict(wave=wave, n_jobs=len(out), noise_sd=noise_sd, kernel=str(em.gp.kernel_),
                                                       best_per_bin=best, n_points=len(g["ybar"])), indent=1))
    print(f"wave {wave}: {len(out)} jobs; best measured per angle bin: " + ", ".join(f"{10+5*b}-{15+5*b}: {v:.0f}" for b, v in best.items()))
    for tr in state["trs"]:
        print(f"  TR{tr['id']} {tr['status']:8s} L {tr['L']:.2f} incumbent {tr['incumbent']:.0f} centre angle {10+25*tr['centre'][0]:.1f}")


if __name__ == "__main__":
    main()
