"""C version of the spray integration (GPU pipeline step 1, 2026-10-26): csrc/spray.c through ctypes.

The C code mirrors jaxspray (potential, analytic B-spline forces) and jaxmodel's DOP853 (AGAMA's controller); one particle per call
iteration, so its arithmetic per particle does not depend on the other particles. Used for (i) the host-only centre orbit
(orbit_back: ~6k sequential steps, latency-bound on a GPU and slow in JAX on a CPU) and (ii) finishing the last particles of a GPU
integration (resume), and as a full CPU integrator (integrate). Compiled on first use with fixed flags (-O2 -ffp-contract=off;
results depend on FMA contraction at the 1e-16 level) into ~/.cache/ocen_stream, keyed by the source hash; particle chunks run on
several CPU cores in threads (ctypes releases the GIL).
"""
import ctypes as C, hashlib, os, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

SRC = Path(__file__).parent/"csrc"/"spray.c"
FLAGS = ["-O2", "-ffp-contract=off", "-shared", "-fPIC"]
_LIB = None
RESC = 100.   # kpc: particles beyond R or |z| = RESC are stopped and excluded (chi = -inf); the host tables reach 250 kpc
_LOCK = threading.Lock()
DP = C.POINTER(C.c_double)
IP = C.POINTER(C.c_int)


class Tables(C.Structure):
    _fields_ = [("axi", DP), ("bar", DP), ("m", DP), ("nm", C.c_int), ("nu", C.c_int), ("nw", C.c_int), ("du", C.c_double), ("dw", C.c_double),
                ("rs", C.c_double), ("zs", C.c_double), ("A", C.c_double), ("S", C.c_double), ("theta_T", C.c_double), ("omega", C.c_double),
                ("T", C.c_double), ("sat", DP), ("nr", C.c_int), ("lr0", C.c_double), ("dlr", C.c_double), ("mtot", C.c_double),
                ("G", C.c_double), ("orb", DP), ("norb", C.c_int), ("tc0", C.c_double), ("dtc", C.c_double), ("resc", C.c_double)]


def lib():
    """the compiled library (built once per source version)."""
    global _LIB
    with _LOCK:
        return _LIB or _build()


def _build():
    global _LIB
    src = SRC.read_bytes(); tag = hashlib.sha256(src+" ".join(FLAGS).encode()).hexdigest()[:12]
    d = Path(os.environ.get("OCEN_CACHE", Path.home()/".cache"/"ocen_stream")); d.mkdir(parents=True, exist_ok=True)
    so = d/f"spray_{tag}.so"
    if not so.exists():
        tmp = so.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([os.environ.get("CC", "cc"), *FLAGS, "-o", str(tmp), str(SRC), "-lm"], check=True); os.replace(tmp, so)
    L = C.CDLL(str(so))
    L.integrate_many.argtypes = [C.POINTER(Tables), C.c_int, DP, DP, C.c_double, C.c_int, DP, DP, IP, C.POINTER(C.c_long)]
    L.integrate_resume.argtypes = [C.POINTER(Tables), C.c_int, DP, DP, DP, DP, DP, IP, DP, C.c_double, C.c_int, IP]
    L.orbit_back.argtypes = [C.POINTER(Tables), DP, C.c_double, C.c_int, DP, DP, DP]; L.orbit_back.restype = C.c_int
    L.force_many.argtypes = [C.POINTER(Tables), C.c_int, DP, DP, DP, DP, DP]
    _LIB = L
    return _LIB


def _p(a):
    return a.ctypes.data_as(DP)


def pack(tb):
    """ctypes Tables for one model from a jaxspray/jaxmodel table dict; returns (Tables, list of the arrays it points to)."""
    f64 = lambda k: np.ascontiguousarray(np.asarray(tb[k], np.float64)) if k in tb else np.zeros(6)
    keep = [f64("axi"), f64("bar"), f64("m"), f64("sat"), f64("orb")]
    g = lambda k, d=0.: float(np.asarray(tb[k])) if k in tb else d
    t = Tables(_p(keep[0]), _p(keep[1]), _p(keep[2]), keep[2].size, int(tb["nu"]), int(tb["nw"]), g("du"), g("dw"), g("rs"), g("zs"),
               g("A"), g("S"), g("theta_T"), g("omega"), g("T"), _p(keep[3]), int(tb.get("nr", 0)), g("lr0"), g("dlr"), g("mtot"), g("G"),
               _p(keep[4]), keep[4].shape[0] if keep[4].ndim == 2 else 0, g("tc0"), g("dtc"), g("resc", RESC))
    return t, keep


def _chunks(n, nthreads):
    e = np.linspace(0, n, max(1, min(nthreads, n))+1).astype(int); return list(zip(e[:-1], e[1:]))


def integrate(ic, t0, tb, acc, max_att=200000, nthreads=None):
    """all particles from t0 to T. Returns (final (N, 6), chi [AGAMA time x kpc], attempts (N,); -1 = failed)."""
    L = lib(); T_, keep = pack(tb); ic = np.ascontiguousarray(ic, np.float64); t0 = np.ascontiguousarray(t0, np.float64); n = len(t0)
    y, chi, natt, nf = np.zeros((n, 6)), np.zeros(n), np.zeros(n, np.int32), np.zeros(n, np.int64)

    def run(a, b):
        L.integrate_many(C.byref(T_), b-a, _p(ic[a:b]), _p(t0[a:b]), acc, max_att, _p(y[a:b]), _p(chi[a:b]),
                         natt[a:b].ctypes.data_as(IP), nf[a:b].ctypes.data_as(C.POINTER(C.c_long)))
    with ThreadPoolExecutor(nthreads or os.cpu_count()) as ex:
        list(ex.map(lambda ab: run(*ab), _chunks(n, 8*(nthreads or os.cpu_count()))))
    return y, chi, natt


def resume(state, tb, acc, max_att=200000, nthreads=None):
    """continue particles from mid-integration states dict(t, y, k1, h, chi, nbad, prev) (numpy, updated copies returned) to T."""
    L = lib(); T_, keep = pack(tb); s = {k: np.ascontiguousarray(np.asarray(v), np.int32 if k == "nbad" else np.float64).copy() for k, v in state.items()}
    n = len(s["t"]); natt = np.zeros(n, np.int32)

    def run(a, b):
        L.integrate_resume(C.byref(T_), b-a, _p(s["t"][a:b]), _p(s["y"][a:b]), _p(s["k1"][a:b]), _p(s["h"][a:b]), _p(s["chi"][a:b]),
                           s["nbad"][a:b].ctypes.data_as(IP), _p(s["prev"][a:b]), acc, max_att, natt[a:b].ctypes.data_as(IP))
    with ThreadPoolExecutor(nthreads or os.cpu_count()) as ex:
        list(ex.map(lambda ab: run(*ab), _chunks(n, 4*(nthreads or os.cpu_count()))))
    s["natt"] = natt
    return s


def orbit_back(xv0, tb, acc, nmax=32768):
    """host-only centre orbit from T back to 0 (DOP853 at acc): accepted steps as (t ascending (n,), states (n, 6), accelerations (n, 3))."""
    L = lib(); T_, keep = pack(tb); xv0 = np.ascontiguousarray(xv0, np.float64)
    t, y, a = np.zeros(nmax), np.zeros((nmax, 6)), np.zeros((nmax, 3))
    n = L.orbit_back(C.byref(T_), _p(xv0), acc, nmax, _p(t), _p(y), _p(a))
    if n < 0:
        raise RuntimeError("cspray.orbit_back failed (buffer exhausted or non-finite step)")
    return t[:n][::-1].copy(), y[:n][::-1].copy(), a[:n][::-1].copy()


def hermite(t, tk, f, df):
    """cubic Hermite interpolation of f (n, d) with derivatives df at ascending knots tk, at times t (numpy; as jaxspray._hermite)."""
    k = np.clip(np.searchsorted(tk, t)-1, 0, len(tk)-2); dt = (tk[k+1]-tk[k])[:, None]; h = (np.asarray(t)[:, None]-tk[k][:, None])/dt
    return (2*h**3-3*h**2+1)*f[k]+(h**3-2*h**2+h)*df[k]*dt+(-2*h**3+3*h**2)*f[k+1]+(h**3-h**2)*df[k+1]*dt
