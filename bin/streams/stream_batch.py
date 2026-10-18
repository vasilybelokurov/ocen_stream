#!/usr/bin/env python3
"""Batches of stream sprays for the 9-D emulator, with in-worker scoring and jackknife (2026-10-18).

Runs a job list (OUTDIR/jobs.json: run_id, params, seed, nrel, epoch_offset, purpose) with the same spray, storage and
scores as training_set.py (float64 spray; chi 0-200 / pooled / legacy scores with per-star components), plus the
delete-group jackknife of the chi 0-200 score in the same worker (as training_jackknife.py: G = 10, interleaved). Every
job draws its own escape offsets (seed) AND a random start of the release-epoch grid (make_spray epoch_offset), so
replicates are independent in all simulation randomness (stochastic-kriging replicates).
Provenance: OUTDIR/manifest.json (transitive source hashes, input files, data hash, job-list hash); fail closed on resume;
workers re-hash the sources before each job.

Usage:
  python bin/streams/stream_batch.py replicates --outdir results/streams/training_9d_reps_20261018   # writes jobs.json
  OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/stream_batch.py run --outdir <dir> --workers 8
  python bin/streams/stream_batch.py status --outdir <dir>
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import training_set as TS       # noqa: E402  (design, constants, provenance helpers; not modified)

G_JK = 10
TRAIN = ROOT/"results/streams/training_9d_20261017"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def maximin_points(n, forced):
    """Greedy maximin subset (unit-cube distance) of the training design, starting from the forced indices."""
    U = np.array([p["unit"] for p in json.loads((TRAIN/"design.json").read_text())["points"]])
    sel = list(dict.fromkeys(forced))
    d = np.min(np.linalg.norm(U[:, None, :]-U[None, sel, :], axis=2), axis=1)
    while len(sel) < n:
        i = int(np.argmax(d)); sel.append(i)
        d = np.minimum(d, np.linalg.norm(U-U[i], axis=1))
    return sel


def replicate_jobs(npts=64, nrep=2):
    pts = json.loads((TRAIN/"design.json").read_text())["points"]
    P = {k: np.array([p["params"][k] for p in pts]) for k in TS.NAMES}
    U = np.array([p["unit"] for p in pts])
    forced = [int(np.argmin(np.sum((U-0.5)**2, 1))), int(np.argmax(P["amp"]*P["size"])), int(np.argmin(P["amp"])),
              int(np.argmax(P["omega"])), int(np.argmin(P["omega"])), int(np.argmax(P["angle"])), int(np.argmin(P["angle"]))]
    sel = maximin_points(npts, forced)
    jobs = []
    for i in sel:
        for r in range(1, nrep+1):
            seed = 200000+10*i+r
            u = float(np.random.default_rng([seed, 777]).uniform())
            jobs.append(dict(run_id=f"p{i:03d}_s{r}", point=int(i), replicate=r, seed=seed, nrel=TS.NREL, epoch_offset=u,
                             params=pts[i]["params"], purpose="replicate (stochastic kriging)"))
    return jobs


def manifest(jobs):
    from grid4_common import du, dW, d, cov, win, BG_BOUNDS, V_BOUNDS
    from ocen_dm.provenance import environment_record
    h = hashlib.sha256()
    for a in (du, dW, d["v"], d["e_v"], cov, np.asarray(win)):
        h.update(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())
    srcs = TS.project_sources()+["bin/streams/stream_batch.py"]
    m = dict(sources={s: sha(ROOT/s) for s in sorted(set(srcs))}, inputs=TS.input_files(), data_sha256=h.hexdigest(),
             jobs_sha256=hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest(), age_cut_myr=TS.AGE_CUT,
             score_chunk=TS.SCORE_CHUNK, chi_edges=TS.CHI_EDGES.tolist(), jackknife_groups=G_JK,
             scores=dict(common=dict(w_bounds=[list(b) for b in BG_BOUNDS], v_bounds=list(V_BOUNDS), u_window=list(map(float, win)))),
             spray=dict(function="grid4_common.make_spray", frame="baumgardt", release="uniform + epoch_offset", window_myr=1000.,
                        arm_only=1, escape_recipe="fardal"),
             environment=environment_record(),
             git_commit_at_start=os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip())
    m["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k not in ("environment", "git_commit_at_start")},
                                                     sort_keys=True).encode()).hexdigest()
    return m


def task(args):
    job, mid, sources, outdir = args
    rd = Path(outdir)/"runs"; rid = job["run_id"]
    try:
        changed = [s for s, h in sources.items() if sha(ROOT/s) != h]
        if changed:
            raise RuntimeError(f"source files changed since the manifest: {changed}")
        from grid4_common import make_spray, score
        p = job["params"]; t0 = time.time()
        m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=job["nrel"], seed=job["seed"], pm=(p["pmra"], p["pmdec"]),
                       host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"])),
                       max_age_myr=TS.AGE_CUT, epoch_offset=job["epoch_offset"])
        t1 = time.time()
        keys = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")
        ms = {k: np.asarray(m[k], np.float64) for k in keys}; ms["age_cut_myr"] = np.float64(TS.AGE_CUT)
        npz = rd/f"{rid}.npz"; tmp = rd/f"{rid}_tmp.npz"; np.savez_compressed(tmp, **ms); os.replace(tmp, npz)
        E, ck = TS.CHI_EDGES, TS.SCORE_CHUNK
        sc = dict(chi200=score(ms, chi_edges=E, return_neff=True, return_components=True, chunk=ck),
                  pooled=score(ms, pooled=True, return_components=True, chunk=ck), legacy=score(ms, legacy_bins=True, chunk=ck))
        n = len(ms["chi"]); idx = np.arange(n); lo_t, lo_p = [], []
        for g in range(G_JK):
            keep = idx % G_JK != g
            s = score({k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in ms.items()},
                      chi_edges=E, chunk=ck)
            lo_t.append(s["total"]); lo_p.append(s["per_star"])
        lo_t, lo_p = np.array(lo_t), np.array(lo_p); t2 = time.time()
        full = sc["chi200"]
        tot_jk = G_JK*full["total"]-(G_JK-1)*lo_t.mean(); var_jk = (G_JK-1)/G_JK*np.sum((lo_t-lo_t.mean())**2)
        per = {f"lnL_{k}": np.asarray(v["per_star"], np.float64) for k, v in sc.items()}
        per["lnL_chi200_jk"] = G_JK*full["per_star"]-(G_JK-1)*lo_p.mean(axis=0)
        per["n_eff_chi200"] = np.asarray(full["n_eff"], np.float32)
        for k in ("chi200", "pooled"):
            per[f"fg_joint_{k}"] = np.asarray(sc[k]["foreground_joint"]); per[f"fg_u_{k}"] = np.asarray(sc[k]["foreground_u"])
        per["bg_u"] = np.float64(full["background_u"]); per["bg_w"] = np.asarray(full["background_w"])
        per["leave_out_totals"] = lo_t
        ptmp = rd/f"{rid}_perstar_tmp.npz"; np.savez_compressed(ptmp, **per); os.replace(ptmp, rd/f"{rid}_perstar.npz")
        sel = (ms["chi"] > 0) & (ms["age"] < TS.AGE_CUT); nbin = np.histogram(ms["chi"][sel], E)[0]
        rec = dict(job, manifest_sha256=mid, spray_sha256=sha(npz), perstar_sha256=sha(rd/f"{rid}_perstar.npz"),
                   n_particles=int(n), n_selected=int(sel.sum()), empty_chi_bins=int((nbin == 0).sum()),
                   sparse_chi_bins_lt20=int((nbin < 20).sum()),
                   scores={k: dict(total=float(v["total"]), K=int(v["K"]), frac_bg=float(v["frac_bg"]), settings=v["settings"])
                           for k, v in sc.items()},
                   total_jk=float(tot_jk), bias_jk=float(full["total"]-tot_jk), var_jk=float(var_jk),
                   n_eff_chi200_median=float(np.median(full["n_eff"])), t_spray_s=t1-t0, t_score_jk_s=t2-t1)
        TS.atomic_write(rd/f"{rid}.json", json.dumps(rec, indent=1))
        return rid, tot_jk, t1-t0, t2-t1, None
    except Exception as e:
        TS.atomic_write(rd/f"{rid}.error.json", json.dumps(dict(job, manifest_sha256=mid, error=repr(e))))
        return rid, None, 0., 0., repr(e)


def prepare(outdir):
    (outdir/"runs").mkdir(parents=True, exist_ok=True)
    jobs = json.loads((outdir/"jobs.json").read_text())
    man = manifest(jobs); mf = outdir/"manifest.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        if old["manifest_sha256"] != man["manifest_sha256"]:
            diff = [k for k in man if k not in ("environment", "git_commit_at_start", "manifest_sha256") and old.get(k) != man[k]]
            raise SystemExit(f"manifest changed ({diff}); refusing to resume into {outdir}")
    else:
        TS.atomic_write(mf, json.dumps(man, indent=1))
    return jobs, man["manifest_sha256"], man["sources"]


def done(outdir, mid):
    out = set()
    for f in (outdir/"runs").glob("*.json"):
        if f.name.endswith(".error.json"):
            continue
        r = json.loads(f.read_text())
        if r.get("manifest_sha256") != mid:
            raise SystemExit(f"{f.name}: manifest mismatch; refusing to mix")
        rd = outdir/"runs"
        npz, per = rd/f"{r['run_id']}.npz", rd/f"{r['run_id']}_perstar.npz"
        if npz.exists() and per.exists() and sha(npz) == r["spray_sha256"] and sha(per) == r["perstar_sha256"]:
            out.add(r["run_id"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("replicates", "run", "status"))
    ap.add_argument("--outdir", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--nrel", type=int, default=0, help="smoke tests only")
    a = ap.parse_args(); outdir = Path(a.outdir) if Path(a.outdir).is_absolute() else ROOT/a.outdir
    if a.action == "replicates":
        outdir.mkdir(parents=True, exist_ok=True)
        jobs = replicate_jobs()
        if a.nrel:
            jobs = [dict(j, nrel=a.nrel) for j in jobs]
        if (outdir/"jobs.json").exists():
            raise SystemExit("jobs.json exists; refusing to overwrite")
        TS.atomic_write(outdir/"jobs.json", json.dumps(jobs, indent=1))
        print(f"{len(jobs)} jobs over {len({j['point'] for j in jobs})} points -> {outdir/'jobs.json'}")
        return
    jobs, mid, sources = prepare(outdir); dn = done(outdir, mid)
    if a.action == "status":
        print(f"{len(dn)}/{len(jobs)} done, {len(list((outdir/'runs').glob('*.error.json')))} errors"); return
    todo = [j for j in jobs if j["run_id"] not in dn][:a.limit or None]
    print(f"{len(todo)} jobs to do ({len(dn)} done), {a.workers} workers", flush=True)
    nerr = 0
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for k, (rid, tot, ts, tj, err) in enumerate(pool.imap_unordered(task, [(j, mid, sources, str(outdir)) for j in todo]), 1):
            msg = f"ERROR {err}" if err else f"lnL_jk {tot:.1f} spray {ts:.0f}s score+jk {tj:.0f}s"
            nerr += bool(err) and "source files changed" not in err
            print(f"{time.strftime('%H:%M:%S')} {k}/{len(todo)} {rid} {msg}", flush=True)
    if nerr:                                  # unattended runs: failures must not look like success (Codex 2026-10-18)
        print(f"{nerr} jobs FAILED", flush=True); sys.exit(1)


if __name__ == "__main__":
    main()

# 2026-10-18: the first replicate batch (training_9d_reps_20261018) was stopped after 64 of 128 jobs by editing this file
# (workers re-hash sources and refuse later jobs); see the project journal.
