#!/usr/bin/env python3
"""Emulator training set over the 9-D model space (2026-10-17; ranges agreed with the user, JOURNAL 2026-10-17).

Parameters and ranges:
  angle   10-35 deg          bar angle today
  omega   30-40 km/s/kpc     bar pattern speed
  amp     0.6-2.0            bar amplitude A
  size    0.8-min(1.5, R_CR/4.25)   bar size S; S x 4.25 kpc (A2 half-peak bar length at S = 1) <= corotation radius of the
                             point's own axisymmetric host; mapped from the unit cube without rejection
  dist    5.25-5.95 kpc      omega Cen distance
  pmra, pmdec  MU_SYS +- 4 x 0.022 mas/yr
  vc      220-240 km/s       v_c(R0) via the halo normalisation (axi_variant vc_target)
  halo_q  0.8-1.2            halo flattening (axi_variant halo_q)
Design: scrambled Sobol (scipy.stats.qmc, d = 9, seed 20261017), 256 points. Independent seed per run (no common random
numbers). Run order (Codex 2026-10-17: calibration tranche first): points 0-31; calibration: two points (nearest the cube
centre; largest amp x size) with 7 extra seeds at 256k and 8 seeds at 128k; replicates: points 0, 4, ..., 28 with 3 extra
seeds; points 32-127; points 128-255.

Generation is independent of the likelihood definition: every run saves its full spray in float64 (every returned particle:
l, b, pmra, pmdec, vlos, d, chi, age; trailing arm, age < 700 Myr), so any score can be recomputed later (~20 s per run).
Three scores are stored for monitoring: fixed chi bins 0-200 (width 5, empty bins kept), pooled, legacy chi bins (S3), each
with per-star ln L, the per-star foreground joint and u-marginal densities (float64; numerator and denominator follow with
the background constants in the manifest), K, frac_bg, settings; per-member n_eff (chi-bin score); chi histogram 0-1000.
Provenance: OUTDIR/manifest.json: sha256 of every project source module the generator imports (transitively, found by a
trial spray), of the input files (potential .ini files, mass profile, frame, member catalogue and Gaia errors), of the
member data arrays and of the design; particle numbers, age cut, score settings, environment, git commit. A restart with a
different manifest aborts; each worker re-hashes the sources before a run and refuses to run if any changed (fail closed);
a run counts as done only if its record matches the manifest and the hashes of its files.

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/training_set.py run --workers 8
  python bin/streams/training_set.py status
Output: results/streams/training_9d_20261017/{manifest.json, design.json, runs/<run_id>.npz, runs/<run_id>.json}
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUTDIR = ROOT/"results/streams/training_9d_20261017"
NREL = 256000
NREL_CAL = 128000
AGE_CUT = 700.
RBAR_KPC = 4.25
SOBOL_SEED = 20261017
NPTS = 256
REPLICATE_POINTS = list(range(0, 32, 4))
N_EXTRA_SEEDS = 3
CHI_EDGES = np.arange(0., 205., 5.)
MU_ERR = 0.022
NAMES = ("angle", "omega", "amp", "size", "dist", "pmra", "pmdec", "vc", "halo_q")
CAL_EXTRA_SEEDS = 7          # calibration points: seeds r = 1..7 at NREL (r = 0 is in the main design) and r = 0..7 at NREL_CAL
CHI_HIST = np.arange(0., 1005., 5.)
SCORE_CHUNK = 50             # members per kernel block: peak ~0.9 GB per worker at 256k (400 -> 6.8 GB); results bit-identical
HUNTER_FILES = ("MWPotentialHunter24_axi.ini", "MWPotentialHunter24_baryon_full.ini", "MWPotentialHunter24_baryon_axi.ini")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r_cr(omega, vc, halo_q):
    """Corotation radius [kpc] of the point's axisymmetric host: root of v_c(R)/R = omega [km/s/kpc] (bracketed, bisection)."""
    from ocen_dm.streams.mw_variants import axisymmetric_variant
    from ocen_dm.streams.restricted import HUNTER24_DIR
    pot = axisymmetric_variant(HUNTER24_DIR, halo_q=halo_q, vc_target=vc)[0]
    om = lambda R: np.sqrt(-R*pot.force(np.array([[R, 0., 0.]]))[0, 0])/R - omega
    lo, hi = 1.0, 20.0
    if not (om(lo) > 0 > om(hi)):
        raise ValueError(f"corotation for omega {omega} not bracketed in {lo}-{hi} kpc")
    for _ in range(60):
        mid = 0.5*(lo+hi)
        lo, hi = (mid, hi) if om(mid) > 0 else (lo, mid)
    return 0.5*(lo+hi)


def design():
    from scipy.stats import qmc
    from ocen_dm.cluster import MU_SYS
    U = qmc.Sobol(d=9, scramble=True, seed=SOBOL_SEED).random_base2(m=int(np.log2(NPTS)))
    pts = []
    for i, u in enumerate(U):
        p = dict(angle=10.+25.*u[0], omega=30.+10.*u[1], amp=0.6+1.4*u[2], dist=5.25+0.7*u[4],
                 pmra=MU_SYS[0]+MU_ERR*(8.*u[5]-4.), pmdec=MU_SYS[1]+MU_ERR*(8.*u[6]-4.), vc=220.+20.*u[7], halo_q=0.8+0.4*u[8])
        rcr = r_cr(p["omega"], p["vc"], p["halo_q"])
        smax = min(1.5, rcr/RBAR_KPC)
        if smax <= 0.8:
            raise ValueError(f"point {i}: corotation {rcr:.2f} kpc leaves no allowed bar size")
        p["size"] = 0.8+u[3]*(smax-0.8)
        pts.append(dict(index=i, unit=u.tolist(), params={k: float(p[k]) for k in NAMES}, r_cr_kpc=rcr, size_max=smax))
    return pts


def calibration_points(pts):
    """Two calibration points among 0-31: nearest the unit-cube centre, and the largest amp x size."""
    U = np.array([p["unit"] for p in pts[:32]])
    c = int(np.argmin(np.sum((U-0.5)**2, axis=1)))
    e = int(np.argmax([p["params"]["amp"]*p["params"]["size"] for p in pts[:32]]))
    return [c, e] if e != c else [c, int(np.argsort(np.sum((U-0.5)**2, axis=1))[1])]


def runs(pts):
    seed = lambda i, r, n: 100000+100*i+r+(50 if n == NREL_CAL else 0)
    order = [(i, 0, NREL) for i in range(32)]
    cal = calibration_points(pts)
    order += [(i, r, NREL) for i in cal for r in range(1, CAL_EXTRA_SEEDS+1)]
    order += [(i, r, NREL_CAL) for i in cal for r in range(CAL_EXTRA_SEEDS+1)]
    order += [(i, r, NREL) for i in REPLICATE_POINTS if i not in cal for r in range(1, N_EXTRA_SEEDS+1)]
    order += [(i, 0, NREL) for i in range(32, 128)] + [(i, 0, NREL) for i in range(128, NPTS)]
    return [dict(run_id=f"p{i:03d}_r{r}_n{n//1000}k", point=i, replicate=r, nrel=n, seed=seed(i, r, n), params=pts[i]["params"])
            for i, r, n in order]


def project_sources():
    """Project source files imported by a trial spray + score (transitive), as repository-relative paths."""
    from grid4_common import make_spray, score
    m = make_spray(35., 25., 1.2, 5.6, nrel=40, seed=1, host_kw=dict(bar_size=1.1, axi_variant=dict(halo_q=0.9, vc_target=230.)),
                   max_age_myr=AGE_CUT)
    score(m, chi_edges=CHI_EDGES, return_neff=True, return_components=True)
    out = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().is_relative_to(ROOT) and f.endswith(".py"):
            out.add(str(Path(f).resolve().relative_to(ROOT)))
    out.add("bin/streams/training_set.py")
    return sorted(out)


def input_files():
    from ocen_dm.streams.restricted import HUNTER24_DIR
    files = {f"$OCEN_BAR_DIR/agama_potentials/{f}": os.path.join(HUNTER24_DIR, f) for f in HUNTER_FILES}
    files["$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits"] = os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")
    for f in ("results/plot_data/stream5455_gaia_errors.npz", "results/plot_data/stream54_gc_frame_chord.json",
              "results/nbody/A_nodm/model_profiles.json"):
        files[f] = str(ROOT/f)
    return {k: sha256_file(v) for k, v in files.items()}


def manifest(pts):
    from grid4_common import du, dW, d, cov, win, BG_BOUNDS, V_BOUNDS
    from ocen_dm.provenance import environment_record
    h = hashlib.sha256()
    for a in (du, dW, d["v"], d["e_v"], cov, np.asarray(win)):
        h.update(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    m = dict(sources={s: sha256_file(ROOT/s) for s in project_sources()}, inputs=input_files(), data_sha256=h.hexdigest(),
             n_members=int(len(du)),
             design_sha256=hashlib.sha256(json.dumps(pts, sort_keys=True).encode()).hexdigest(),
             nrel=NREL, nrel_calibration=NREL_CAL, age_cut_myr=AGE_CUT, score_chunk=SCORE_CHUNK, rbar_kpc=RBAR_KPC, sobol_seed=SOBOL_SEED, npts=NPTS,
             scores=dict(chi200=dict(chi_edges=CHI_EDGES.tolist()), pooled=dict(pooled=True), legacy=dict(legacy_bins=True),
                         common=dict(w_bounds=[list(b) for b in BG_BOUNDS], v_bounds=list(V_BOUNDS), u_window=list(map(float, win)))),
             spray=dict(function="grid4_common.make_spray", frame="baumgardt", release="uniform", window_myr=1000., arm_only=1,
                        escape_recipe="fardal", host="hunter24 barred, axi_variant(halo_q, vc_target)", fields_float32=True),
             environment=environment_record(), git_commit_at_start=commit)
    m["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k not in ("environment", "git_commit_at_start")},
                                                     sort_keys=True).encode()).hexdigest()
    return m


def atomic_write(path, text):
    tmp = path.with_suffix(path.suffix+".tmp"); tmp.write_text(text); os.replace(tmp, path)


def task(job):
    job, mid, sources, outdir = job
    runs_dir = Path(outdir)/"runs"      # passed explicitly: spawned workers do not see a --outdir override
    try:
        changed = [s for s, h in sources.items() if sha256_file(ROOT/s) != h]
        if changed:
            raise RuntimeError(f"source files changed since the manifest: {changed}")
        from grid4_common import make_spray, score
        p = job["params"]; t0 = time.time()
        m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=job["nrel"], seed=job["seed"], pm=(p["pmra"], p["pmdec"]),
                       host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"])), max_age_myr=AGE_CUT)
        t1 = time.time()
        keys = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")
        ms = {k: np.asarray(m[k], np.float64) for k in keys}; ms["age_cut_myr"] = np.float64(AGE_CUT)
        npz = runs_dir/f"{job['run_id']}.npz"
        tmp = npz.with_name(npz.stem+"_tmp.npz"); np.savez_compressed(tmp, **ms); os.replace(tmp, npz)
        sc = dict(chi200=score(ms, chi_edges=CHI_EDGES, return_neff=True, return_components=True, chunk=SCORE_CHUNK),
                  pooled=score(ms, pooled=True, return_components=True, chunk=SCORE_CHUNK),
                  legacy=score(ms, legacy_bins=True, chunk=SCORE_CHUNK))
        t2 = time.time()
        sel = (ms["chi"] > 0) & (ms["age"] < AGE_CUT)
        nbin = np.histogram(ms["chi"][sel], CHI_EDGES)[0]
        per = {f"lnL_{k}": np.asarray(v["per_star"], np.float32) for k, v in sc.items()}
        per["n_eff_chi200"] = np.asarray(sc["chi200"]["n_eff"], np.float32)
        for k in ("chi200", "pooled"):
            per[f"fg_joint_{k}"] = np.asarray(sc[k]["foreground_joint"], np.float64)
            per[f"fg_u_{k}"] = np.asarray(sc[k]["foreground_u"], np.float64)
        per["bg_u"] = np.float64(sc["chi200"]["background_u"]); per["bg_w"] = np.asarray(sc["chi200"]["background_w"], np.float64)
        ptmp = runs_dir/f"{job['run_id']}_perstar_tmp.npz"; np.savez_compressed(ptmp, **per)
        os.replace(ptmp, runs_dir/f"{job['run_id']}_perstar.npz")
        rec = dict(job, manifest_sha256=mid, spray_sha256=sha256_file(npz),
                   perstar_sha256=sha256_file(runs_dir/f"{job['run_id']}_perstar.npz"),
                   n_particles=int(len(ms["chi"])), n_selected=int(sel.sum()), n_selected_chi200=int(nbin.sum()),
                   empty_chi_bins=int((nbin == 0).sum()), sparse_chi_bins_lt20=int((nbin < 20).sum()),
                   chi_hist_0_1000_step5=np.histogram(ms["chi"][sel], CHI_HIST)[0].tolist(),
                   n_chi_negative=int((ms["chi"] <= 0).sum()),
                   scores={k: dict(total=float(v["total"]), K=int(v["K"]), frac_bg=float(v["frac_bg"]), settings=v["settings"])
                           for k, v in sc.items()},
                   n_eff_chi200_median=float(np.median(sc["chi200"]["n_eff"])), t_spray_s=t1-t0, t_score_s=t2-t1)
        atomic_write(runs_dir/f"{job['run_id']}.json", json.dumps(rec, indent=1))
        return job["run_id"], rec["scores"]["chi200"]["total"], t1-t0, t2-t1, None
    except Exception as e:      # recorded, not fatal for the campaign
        atomic_write(runs_dir/f"{job['run_id']}.error.json", json.dumps(dict(job, manifest_sha256=mid, error=repr(e))))
        return job["run_id"], None, 0., 0., repr(e)


def done_runs(mid):
    done = set()
    for f in (OUTDIR/"runs").glob("*.json"):
        if f.name.endswith(".error.json"):
            continue
        r = json.loads(f.read_text())
        if r.get("manifest_sha256") != mid:
            raise SystemExit(f"{f.name}: manifest {r.get('manifest_sha256')} != current {mid}; refusing to mix (new OUTDIR needed)")
        npz, per = OUTDIR/"runs"/f"{r['run_id']}.npz", OUTDIR/"runs"/f"{r['run_id']}_perstar.npz"
        if npz.exists() and per.exists() and sha256_file(npz) == r["spray_sha256"] and sha256_file(per) == r["perstar_sha256"]:
            done.add(r["run_id"])
    return done


def prepare():
    (OUTDIR/"runs").mkdir(parents=True, exist_ok=True)
    dfile = OUTDIR/"design.json"
    pts = json.loads(dfile.read_text())["points"] if dfile.exists() else design()
    if not dfile.exists():
        atomic_write(dfile, json.dumps(dict(names=NAMES, points=pts), indent=1))
    man = manifest(pts)
    mfile = OUTDIR/"manifest.json"
    if mfile.exists():
        old = json.loads(mfile.read_text())
        if old["manifest_sha256"] != man["manifest_sha256"]:
            diff = [k for k in man if k not in ("environment", "git_commit_at_start", "manifest_sha256") and old.get(k) != man[k]]
            raise SystemExit(f"manifest changed ({diff}); refusing to resume into {OUTDIR}")
    else:
        atomic_write(mfile, json.dumps(man, indent=1))
    return pts, man["manifest_sha256"], man["sources"]


def run(args):
    pts, mid, sources = prepare()
    done = done_runs(mid)
    todo = [r for r in runs(pts) if r["run_id"] not in done][:args.limit or None]
    print(f"{len(todo)} runs to do ({len(done)} done), {args.workers} workers, nrel {NREL} (calibration {NREL_CAL}); "
          f"calibration points {calibration_points(pts)}", flush=True)
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for k, (rid, tot, ts, tsc, err) in enumerate(pool.imap(task, [(r, mid, sources, str(OUTDIR)) for r in todo]), 1):
            msg = f"ERROR {err}" if err else f"lnL_chi200 {tot:.1f} spray {ts:.0f}s score {tsc:.0f}s"
            print(f"{time.strftime('%H:%M:%S')} {k}/{len(todo)} {rid} {msg}", flush=True)


def status(args):
    pts, mid, _ = prepare()
    done = done_runs(mid); allr = runs(pts)
    errs = list((OUTDIR/"runs").glob("*.error.json"))
    print(f"{len(done)}/{len(allr)} runs done, {len(errs)} errors")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("run", "status"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending runs (smoke tests)")
    ap.add_argument("--outdir", default=None, help="override the output directory (smoke tests)")
    ap.add_argument("--nrel", type=int, default=None, help="override both particle numbers (smoke tests only; changes the manifest)")
    a = ap.parse_args()
    global OUTDIR, NREL, NREL_CAL
    if a.outdir:
        OUTDIR = Path(a.outdir)
    if a.nrel:
        NREL = NREL_CAL = a.nrel
    dict(run=run, status=status)[a.action](a)


if __name__ == "__main__":
    main()
