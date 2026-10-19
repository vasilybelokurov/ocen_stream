#!/usr/bin/env python3
"""Direct bar-angle profile by Nelder-Mead on sprays (2026-10-19; plan reviewed by Codex,
docs/reviews/codex_profile_plan_2026-10-19.md).

At each fixed bar angle, maximise the jackknife-corrected ln L (fixed chi bins 0-200, cutoff 7) over the other 8
parameters (unit-cube coordinates of the 9-D design; bar size as a fraction of its corotation-allowed range) with a bounded
ask/tell Nelder-Mead (reflection 1, expansion 2, contraction 0.5, shrink 0.5; points clipped to [0, 1]; initial simplex
step 0.1). Two chains per angle start from distinct earlier runs: the best measured run within +-2.5 deg of the angle and
the best one whose other 8 coordinates are >= 0.3 away from it (both moved to the angle). Each chain uses common random
numbers (one seed and epoch offset for all its evaluations; diagnostic 2026-10-19: differences reliable for steps >~0.05).
Search at NREL_SEARCH (128k; same ranking as 256k, ~40-125 lower). A chain stops when its simplex is smaller than 0.05
(max vertex distance from the best) or after BUDGET evaluations. Then each angle's best point gets 4 independent 256k
seeds ('final').
All chains run concurrently on a process pool; every evaluation is stored (runs/<id>.{npz,json}) and reused on restart
(identity = point, seed, offset, nrel); sources are hashed in manifest.json and re-checked before each evaluation.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/profile_nm.py --catalogue CAT --box wide --start-from DIR ... \
           --outdir OUT --angles 12.5 17.5 22.5 27 27.5 32.5 --workers 8 [--budget 30] [--nrel 128000]
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
STEP0, TOL, NFINAL = 0.1, 0.05, 4
SOURCES = ["bin/streams/profile_nm.py", "bin/streams/grid4_common.py", "bin/streams/rescore_batch.py", "bin/streams/mock_data.py",
           "bin/streams/emulator_design.py", "src/ocen_dm/streams/score.py", "src/ocen_dm/streams/spray.py",
           "src/ocen_dm/streams/restricted.py", "src/ocen_dm/streams/mw_variants.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(args):
    """Spray at unit point u (angle included) and score against the catalogue. Returns the record."""
    u, seed, off, nrel, spec, outdir, rid, sources = args
    rec_f = Path(outdir)/"runs"/f"{rid}.json"
    if rec_f.exists():
        return json.loads(rec_f.read_text())
    changed = [s for s, h in sources.items() if sha(ROOT/s) != h]
    if changed:
        raise RuntimeError(f"sources changed: {changed}")
    import emulator_design as ED, rescore_batch as RB
    from grid4_common import make_spray, win, project_gc, GC
    from ocen_dm.streams.score import score_conditional
    p = ED.unit_to_params(np.asarray(u)); t0 = time.time()
    m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=nrel, seed=seed, pm=(p["pmra"], p["pmdec"]),
                   host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"])), max_age_myr=700., epoch_offset=off)
    keys = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")
    ms = {k: np.asarray(m[k], np.float64) for k in keys}; ms["age_cut_myr"] = np.float64(700.)
    tmp = Path(outdir)/"runs"/f"{rid}_tmp.npz"; np.savez_compressed(tmp, **ms); os.replace(tmp, Path(outdir)/"runs"/f"{rid}.npz")
    c, csha, ob = RB.catalogue("real" if spec["catalogue"] == "real" else str(ROOT/spec["catalogue"]))
    if csha != spec["sha256"]:
        raise RuntimeError("catalogue changed")
    BG, VB = RB.bounds(spec["box"]); ob = "raise" if spec["box"] == "wide" else ob
    mu, mx = project_gc(ms["l"], ms["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, ms["pmra"], ms["pmdec"])), ms["vlos"],
                          ms["chi"], ms["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    rec = dict(run_id=rid, unit=list(map(float, u)), params=p, seed=seed, epoch_offset=off, nrel=nrel, total=float(s["total"]),
               total_jk=float(s["total_jk"]), var_jk=float(s["var_jk"]), catalogue=spec, seconds=time.time()-t0,
               spray_sha256=sha(Path(outdir)/"runs"/f"{rid}.npz"))
    tmpj = rec_f.with_suffix(".json.tmp"); tmpj.write_text(json.dumps(rec, indent=1)); os.replace(tmpj, rec_f)
    return rec


class NM:
    """Ask/tell Nelder-Mead minimising f over the 8 free unit coordinates (angle fixed)."""

    def __init__(self, cid, angle_u, x0, seed, off, budget):
        self.cid, self.au, self.seed, self.off, self.budget = cid, angle_u, seed, off, budget
        x0 = np.clip(np.asarray(x0, float), 0, 1); S = [x0]
        for i in range(8):
            v = x0.copy(); v[i] = v[i]+STEP0 if v[i]+STEP0 <= 1 else v[i]-STEP0; S.append(v)
        self.S, self.F = np.array(S), np.full(9, np.nan); self.n_eval = 0; self.state = "init"; self.pending = list(range(9))
        self.best_hist = []

    def full(self, z):
        return np.insert(np.clip(z, 0, 1), 0, self.au)

    def done(self):
        if self.state == "init":
            return False
        o = np.argsort(self.F); size = np.max(np.linalg.norm(self.S-self.S[o[0]], axis=1))
        return size < TOL or self.n_eval >= self.budget

    def ask(self):
        """List of (tag, z) points to evaluate now."""
        if self.state == "init":
            return [(("v", i), self.S[i]) for i in self.pending]
        if self.state == "shrink":
            return [(("v", i), self.S[i]) for i in self.pending]
        return [(self.state, self.trial)]

    def _order(self):
        o = np.argsort(self.F); self.S, self.F = self.S[o], self.F[o]; self.best_hist.append(float(self.F[0]))

    def _reflect(self):
        self._order(); self.c = self.S[:-1].mean(0)
        self.trial = np.clip(self.c+(self.c-self.S[-1]), 0, 1); self.state = "reflect"

    def tell(self, tag, z, f):
        self.n_eval += 1
        if tag[0] == "v":
            self.S[tag[1]] = z; self.F[tag[1]] = f; self.pending.remove(tag[1])
            if not self.pending:
                self._reflect()
            return
        st = tag
        if st == "reflect":
            self.fr, self.xr = f, z
            if self.F[0] <= f < self.F[-2]:
                self.S[-1], self.F[-1] = z, f; self._reflect()
            elif f < self.F[0]:
                self.trial = np.clip(self.c+2*(self.xr-self.c), 0, 1); self.state = "expand"
            else:
                outside = f < self.F[-1]
                self.trial = np.clip(self.c+0.5*((self.xr if outside else self.S[-1])-self.c), 0, 1)
                self.state = "contract_out" if outside else "contract_in"
        elif st == "expand":
            if f < self.fr:
                self.S[-1], self.F[-1] = z, f
            else:
                self.S[-1], self.F[-1] = self.xr, self.fr
            self._reflect()
        elif st in ("contract_out", "contract_in"):
            ref = self.fr if st == "contract_out" else self.F[-1]
            if f <= ref:
                self.S[-1], self.F[-1] = z, f; self._reflect()
            else:                                     # shrink towards the best vertex
                self.S[1:] = self.S[0]+0.5*(self.S[1:]-self.S[0]); self.F[1:] = np.nan
                self.pending = list(range(1, 9)); self.state = "shrink"
        if self.state == "shrink" and not self.pending:
            self._reflect()

    def tell_vertex_batch_done(self):
        if self.state == "shrink" and not self.pending:
            self._reflect()


def starts(start_dirs, angle):
    """Two distinct start points (8 free unit coords) for a fixed angle from earlier runs."""
    R = []
    for d in start_dirs:
        for f in (Path(d)/"runs").glob("*.json"):
            if f.name.endswith(".error.json") or f.name.endswith("_jk.json"):
                continue
            j = json.loads(f.read_text())
            if "total_jk" in j and "unit" in j and j.get("purpose") != "validation (held out)":
                R.append((j["total_jk"], np.array(j["unit"])))
    au = (angle-10)/25
    near = [r for r in R if abs(r[1][0]-au) <= 0.1] or R
    near.sort(key=lambda r: -r[0]); a = near[0][1]
    far = [r for r in near if np.linalg.norm(r[1][1:]-a[1:]) >= 0.3]
    b = (far[0] if far else near[1])[1]
    return a[1:], b[1:], near[0][0], (far[0] if far else near[1])[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True); ap.add_argument("--box", default="wide"); ap.add_argument("--outdir", required=True)
    ap.add_argument("--start-from", nargs="+", required=True); ap.add_argument("--angles", nargs="+", type=float, required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--nrel", type=int, default=128000); ap.add_argument("--nrel-final", type=int, default=256000)
    ap.add_argument("--seed-base", type=int, default=900000)
    a = ap.parse_args()
    P = lambda x: Path(x) if Path(x).is_absolute() else ROOT/x
    od = P(a.outdir); (od/"runs").mkdir(parents=True, exist_ok=True)
    spec = dict(catalogue=a.catalogue, box=a.box); spec["sha256"] = "real" if a.catalogue == "real" else sha(P(a.catalogue))
    sources = {s: sha(ROOT/s) for s in SOURCES}
    man = dict(spec=spec, sources=sources, angles=a.angles, budget=a.budget, nrel=a.nrel, nrel_final=a.nrel_final, seed_base=a.seed_base,
               start_from=a.start_from, step0=STEP0, tol=TOL)
    mf = od/"manifest.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        if {k: old[k] for k in man} != man:
            raise SystemExit("manifest differs from an earlier run in this directory; refusing to resume")
    else:
        mf.write_text(json.dumps(man, indent=1))
    chains = []
    for ia, ang in enumerate(a.angles):
        za, zb, fa, fb = starts([P(d) for d in a.start_from], ang)
        for j, z in enumerate((za, zb)):
            seed = a.seed_base+100*ia+j
            chains.append(NM(f"a{ang:g}_{j}", (ang-10)/25, z, seed, float(np.random.default_rng([seed, 777]).uniform()), a.budget))
        print(f"angle {ang:g}: starts from runs with lnL {fa:.0f} and {fb:.0f}", flush=True)
    queue = {}   # future -> (chain, tag, z)

    def submit(ex, ch):
        for tag, z in ch.ask():
            key = (ch.cid, ch.n_eval+len([1 for v in queue.values() if v[0] is ch]))
            rid = f"{ch.cid}_e{key[1]:03d}"
            f = ex.submit(evaluate, (ch.full(z).tolist(), ch.seed, ch.off, a.nrel, spec, str(od), rid, sources))
            queue[f] = (ch, tag, z)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for ch in chains:
            submit(ex, ch)
        while queue:
            doneset, _ = wait(list(queue), return_when=FIRST_COMPLETED)
            for f in doneset:
                ch, tag, z = queue.pop(f); rec = f.result()
                ch.tell(tag, np.asarray(rec["unit"])[1:], -rec["total_jk"])
                busy = any(v[0] is ch for v in queue.values())
                if not busy and not ch.done():
                    submit(ex, ch)
                elif not busy:
                    print(f"{time.strftime('%H:%M:%S')} chain {ch.cid} done: {ch.n_eval} evals, best {-ch.F.min():.0f}", flush=True)
            st = {ch.cid: dict(n_eval=ch.n_eval, best=float(-np.nanmin(ch.F)) if np.isfinite(ch.F).any() else None,
                               best_unit=ch.full(ch.S[int(np.nanargmin(ch.F))]).tolist() if np.isfinite(ch.F).any() else None,
                               state=ch.state) for ch in chains}
            (od/"state.json").write_text(json.dumps(st, indent=1))
        # final independent 256k checks of each angle's best point
        best = {}
        for ch in chains:
            ang = a.angles[[f"a{x:g}" for x in a.angles].index(ch.cid.rsplit("_", 1)[0])]
            v = float(-np.nanmin(ch.F))
            if ang not in best or v > best[ang][0]:
                best[ang] = (v, ch.full(ch.S[int(np.nanargmin(ch.F))]))
        futs = []
        for ia, (ang, (v, u)) in enumerate(sorted(best.items())):
            for k in range(NFINAL):
                seed = a.seed_base+50000+100*ia+k
                futs.append((ang, v, ex.submit(evaluate, (u.tolist(), seed, float(np.random.default_rng([seed, 777]).uniform()), a.nrel_final,
                                                          spec, str(od), f"final_a{ang:g}_{k}", sources))))
        out = {}
        for ang, v, f in futs:
            out.setdefault(ang, dict(search_best_128k=v, final=[]))["final"].append(f.result()["total_jk"])
    for ang, d in out.items():
        y = np.array(d["final"]); d.update(mean=float(y.mean()), se=float(y.std(ddof=1)/np.sqrt(len(y))))
        d["params"] = json.loads((od/"runs"/f"final_a{ang:g}_0.json").read_text())["params"]
    (od/"profile.json").write_text(json.dumps({f"{k:g}": v for k, v in out.items()}, indent=1))
    print(f"done in {(time.time()-t0)/3600:.1f} h")
    for ang in sorted(out):
        d = out[ang]; print(f"angle {ang:5.1f}: 256k {d['mean']:.0f} +- {d['se']:.0f} (search best 128k {d['search_best_128k']:.0f}); "
                            f"Omega {d['params']['omega']:.1f} A {d['params']['amp']:.2f} S {d['params']['size']:.2f} d {d['params']['dist']:.2f} vc {d['params']['vc']:.0f}")


if __name__ == "__main__":
    main()
