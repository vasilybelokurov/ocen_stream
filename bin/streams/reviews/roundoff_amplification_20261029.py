"""Fable review 2026-10-29 (docs/reviews/cross_machine_fable_codex_20261029.md). Usage: PYTHONPATH=src python bin/streams/reviews/roundoff_amplification_20261029.py
ULP-level sensitivity of (a) the C centre orbit and (b) particle integration: does 1e-16 become macroscopic?"""
import os, sys, time, json, ctypes
os.environ.setdefault("OCEN_CACHE", os.path.join(os.environ.get("TMPDIR", "/tmp"), "ocen_cache_review"))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ocen_stream import jaxspray as J, jaxmodel as JM, cspray
from ocen_stream.pilots import to_params
I = str(__import__("pathlib").Path(__file__).resolve().parents[3]/"results/gpu")
ht, inp = J.load_host(I+"/host_tables.npz"), JM.load_inputs(I+"/spray_inputs.npz")
pts = json.loads(open(I+"/points.json").read()); off = pts.pop("_epoch_offset"); trel, keep = JM.schedule(inp, off)
p = {k: float(v) for k, v in list(pts.values())[0].items()}
tb = J.host_for(ht, p["angle"], p["omega"], p["amp"], p["size"], p["halo_q"], p["vc"], inp["T"])
xv0 = np.asarray(JM.today_xv(p["dist"], p["pmra"], p["pmdec"], inp))
t0_ = time.perf_counter(); tk, yk, ak = cspray.orbit_back(xv0, tb, JM.ORBIT_ACC); print("orbit steps", len(tk), "%.3fs" % (time.perf_counter()-t0_))
x1 = xv0.copy(); x1[0] = np.nextafter(x1[0], np.inf)        # 1 ULP in x
tk1, yk1, ak1 = cspray.orbit_back(x1, tb, JM.ORBIT_ACC)
print("(a) centre orbit, 1-ULP input perturbation: steps", len(tk), len(tk1), "| t=0 position diff [kpc]", np.abs(yk1[0,:3]-yk[0,:3]).max(), "| v diff", np.abs(yk1[0,3:]-yk[0,3:]).max())
H = lambda tk, yk, ak, t: np.concatenate((cspray.hermite(t, tk, yk[:, :3], yk[:, 3:]), cspray.hermite(t, tk, yk[:, 3:], ak)), 1)
r0, r1 = H(tk, yk, ak, trel), H(tk1, yk1, ak1, trel)
print("    at release epochs: max |dx| kpc", np.abs(r0[:, :3]-r1[:, :3]).max(), "median", np.median(np.abs(r0[:, :3]-r1[:, :3])))
# (b) particles: full release, then integrate a subset with two builds (contract off vs on = clang default on arm64)
tbf, ic, t0, rj = JM.release(p, ht, inp, trel, keep); ic = np.asarray(ic); t0 = np.asarray(t0)
rng = np.random.default_rng(0); sel = np.sort(rng.choice(len(ic), 3000, replace=False))
t1 = time.perf_counter(); y_a, chi_a, na = cspray.integrate(ic[sel], t0[sel], tbf, JM.SPRAY_ACC); print("integrate 3000 particles %.1fs" % (time.perf_counter()-t1))
cspray._LIB = None; cspray.FLAGS = ["-O2", "-ffp-contract=on", "-shared", "-fPIC"]
y_b, chi_b, nb = cspray.integrate(ic[sel], t0[sel], tbf, JM.SPRAY_ACC)
d = np.abs(y_b[:, :3]-y_a[:, :3]).max(1); dc = np.abs(chi_b-chi_a)
print("(b) 3000 particles, fp-contract off vs on: frac with |dx|>1e-6 kpc", np.mean(d > 1e-6), "| >1e-3", np.mean(d > 1e-3), "| >0.1", np.mean(d > 0.1), "| max", d.max())
print("    chi sign flips", int(np.sum(np.sign(chi_a) != np.sign(chi_b))), "| attempts differ", int(np.sum(na != nb)), "| median |dchi|", np.median(dc))
# (c) 1-ULP perturbation of the centre orbit table seen by particles (same build)
cspray._LIB = None; cspray.FLAGS = ["-O2", "-ffp-contract=off", "-shared", "-fPIC"]
tb2 = dict(tbf, orb=np.nextafter(np.asarray(tbf["orb"]), np.inf))
y_c, chi_c, nc = cspray.integrate(ic[sel], t0[sel], tb2, JM.SPRAY_ACC)
d = np.abs(y_c[:, :3]-y_a[:, :3]).max(1)
print("(c) 1-ULP shift of the whole centre-orbit table: frac |dx|>1e-6", np.mean(d > 1e-6), "| >1e-3", np.mean(d > 1e-3), "| >0.1", np.mean(d > 0.1), "| max", d.max(), "| chi sign flips", int(np.sum(np.sign(chi_a) != np.sign(chi_c))))
