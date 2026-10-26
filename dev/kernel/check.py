"""C prototype vs JAX: (1) analytic force / potential / dchi vs jax.grad; (2) per-particle DOP853 loop vs jaxmodel.dop853_chi.
Usage: PYTHONPATH=<repo>/src python check.py <repo> [n_particles]"""
import ctypes as C, json, sys, time
from pathlib import Path
import numpy as np
import jax; jax.config.update("jax_enable_x64", True); jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
ROOT = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 256; HERE = Path(__file__).parent
from ocen_stream import jaxspray as J, jaxmodel as JM


class Tables(C.Structure):
    _fields_ = [("axi", C.POINTER(C.c_double)), ("bar", C.POINTER(C.c_double)), ("m", C.POINTER(C.c_double)),
                ("nm", C.c_int), ("nu", C.c_int), ("nw", C.c_int), ("du", C.c_double), ("dw", C.c_double), ("rs", C.c_double), ("zs", C.c_double),
                ("A", C.c_double), ("S", C.c_double), ("theta_T", C.c_double), ("omega", C.c_double), ("T", C.c_double),
                ("sat", C.POINTER(C.c_double)), ("nr", C.c_int), ("lr0", C.c_double), ("dlr", C.c_double), ("mtot", C.c_double), ("G", C.c_double),
                ("orb", C.POINTER(C.c_double)), ("norb", C.c_int), ("tc0", C.c_double), ("dtc", C.c_double)]


def pack(tb, keep):
    arr = lambda k: np.ascontiguousarray(np.asarray(tb[k], np.float64)); keep.extend([arr("axi"), arr("bar"), arr("m"), arr("sat"), arr("orb")])
    P = lambda a: a.ctypes.data_as(C.POINTER(C.c_double)); f = lambda k: float(tb[k])
    return Tables(P(keep[0]), P(keep[1]), P(keep[2]), keep[2].size, int(tb["nu"]), int(tb["nw"]), f("du"), f("dw"), f("rs"), f("zs"),
                  f("A"), f("S"), f("theta_T"), f("omega"), f("T"), P(keep[3]), int(tb["nr"]), f("lr0"), f("dlr"), f("mtot"), f("G"),
                  P(keep[4]), keep[4].shape[0], f("tc0"), f("dtc"))


G = ROOT/"results/gpu"; pts = json.loads((G/"points.json").read_text()); OFFSET = pts["_epoch_offset"]; p = pts["c0_final"]
inp = JM.load_inputs(G/"spray_inputs.npz"); ht = J.load_host(G/"host_tables.npz"); trel, keep_idx = JM.schedule(inp, OFFSET)
t1 = time.time(); tb, ic, t0, rj = JM.release(p, ht, inp, trel, keep_idx); ic = np.asarray(ic); t0 = np.asarray(t0); print(f"release {time.time()-t1:.1f} s, {len(ic)} particles")
hold = []; T = pack(tb, hold)
for lib in ("libkern.dylib", "libkern_fma.dylib"):
    L = C.CDLL(str(HERE/lib)); dp = C.POINTER(C.c_double)
    L.force_many.argtypes = [C.POINTER(Tables), C.c_int, dp, dp, dp, dp, dp]
    L.integrate_many.argtypes = [C.POINTER(Tables), C.c_int, dp, dp, C.c_double, C.c_int, dp, dp, C.POINTER(C.c_int), C.POINTER(C.c_long)]
    P = lambda a: a.ctypes.data_as(dp)
    # ---- (1) force check at the release points, at points along the orbit (near the centre) and random points in the box
    rng = np.random.default_rng(1); orb = np.asarray(tb["orb"]); nt = 4000
    xs = np.concatenate((ic[rng.choice(len(ic), nt), :3], orb[rng.choice(len(orb), nt), :3]+rng.normal(0, 0.05, (nt, 3)),
                         rng.uniform(-1, 1, (nt, 3))*np.array([25, 25, 8])))
    ts = np.concatenate((t0[:nt], rng.uniform(0, float(tb["T"]), 2*nt)))
    a = np.zeros((len(xs), 3)); pot = np.zeros(len(xs)); dchi = np.zeros(len(xs))
    L.force_many(C.byref(T), len(xs), P(np.ascontiguousarray(xs)), P(np.ascontiguousarray(ts)), P(a), P(pot), P(dchi))
    aj = np.asarray(J.accel(jnp.asarray(xs), jnp.asarray(ts), tb)); pj = np.asarray(J.phi(jnp.asarray(xs), jnp.asarray(ts), tb))
    dj = np.asarray(JM._deriv(jnp.asarray(np.concatenate((xs, 0*xs), 1)), jnp.asarray(ts), tb)[1])
    ra = np.abs(a-aj).max(1)/np.linalg.norm(aj, axis=1); rp = np.abs(pot-pj)/np.abs(pj); rd = np.abs(dchi-dj)/np.maximum(np.abs(dj), 1e-3)
    print(f"[{lib}] force: max rel |da|/|a| {ra.max():.2e} (median {np.median(ra):.1e}); phi: max rel {rp.max():.2e}; dchi: max {rd.max():.2e}")
    # ---- (2) integration of a subset: random + the slowest particles (finish counts of the tolerance scan at this point)
    fin = np.load(G/"tolscan/c0_final_acc1e-06.npz")["finish"]; slow = np.argsort(fin)[::-1][:N//8]
    sel = np.concatenate((slow, rng.choice(len(ic), N-len(slow), replace=False)))
    y0 = np.ascontiguousarray(ic[sel]); tt = np.ascontiguousarray(t0[sel]); yout = np.zeros((N, 6)); chi = np.zeros(N); natt = np.zeros(N, np.int32); nf = np.zeros(N, np.int64)
    t1 = time.time(); L.integrate_many(C.byref(T), N, P(y0), P(tt), 1e-6, 400000, P(yout), P(chi), natt.ctypes.data_as(C.POINTER(C.c_int)), nf.ctypes.data_as(C.POINTER(C.c_long))); tc = time.time()-t1
    print(f"[{lib}] C: {N} particles, {natt.sum()} attempts ({nf.sum()} force evals), {tc:.2f} s = {tc/natt.sum()*1e6:.1f} us/attempt, "
          f"{tc/nf.sum()*1e9:.0f} ns/force eval; attempts median {np.median(natt):.0f} max {natt.max()} (tolscan finish of these: median {np.median(fin[sel]):.0f} max {fin[sel].max()})")
    if lib == "libkern.dylib":
        t1 = time.time(); finish = np.zeros(N, int); yj, cj, nj = JM.dop853_chi(jnp.asarray(y0), jnp.asarray(tt), tb, 1e-6, 32, finish=finish); tj = time.time()-t1
        yj = np.asarray(yj); cj = np.asarray(cj); print(f"JAX CPU: {tj:.1f} s (incl. compile), slowest {nj} attempts")
        np.savez(HERE/"jax_ref.npz", yj=yj, cj=cj, finish=finish, sel=sel)
    z = np.load(HERE/"jax_ref.npz"); yj, cj, finish = z["yj"], z["cj"], z["finish"]
    dx = np.linalg.norm(yout[:, :3]-yj[:, :3], axis=1); dv = np.linalg.norm(yout[:, 3:]-yj[:, 3:], axis=1); dc = np.abs(chi-cj*1)  # chi in kpc Myr both
    chi_c = chi*JM.T_MYR
    dc = np.abs(chi_c-cj)
    print(f"[{lib}] vs JAX: |dx| kpc median {np.median(dx):.1e} 90% {np.percentile(dx, 90):.1e} max {dx.max():.1e}; |dv| km/s median {np.median(dv):.1e} max {dv.max():.1e}; "
          f"|dchi| kpc Myr median {np.median(dc):.1e} max {dc.max():.1e}; chi sign agree {np.mean(np.sign(chi_c) == np.sign(cj)):.4f}; "
          f"attempts C vs JAX(finish, 32-res): equal within 32 for {np.mean(np.abs(natt-finish) <= 32):.3f}")
    bad = np.argsort(dx)[::-1][:5]; print("   worst |dx|:", [(int(i), f"{dx[i]:.1e}", int(natt[i]), int(finish[i]), f"rmin? chi {chi_c[i]:.3g}") for i in bad])
