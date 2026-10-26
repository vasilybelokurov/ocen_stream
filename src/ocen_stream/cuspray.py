"""CUDA version of the spray integration (GPU pipeline step 2, 2026-10-26): csrc/spray.c compiled as a kernel with NVRTC (CuPy).

One particle per thread (integrate_k in spray.c): the same source as the C library (cspray), so the same DOP853 controller and
potential; the results differ from C only by the rounding of the GPU's transcendental functions (and fused multiply-adds when
fmad=True). Each particle's arithmetic depends only on its own inputs, so a model gives the same bits alone or in a batch, in any
order. Particles are launched in the order given (default: as stored); sorting them by expected cost (e.g. the attempts of the
previous model in a chain) keeps the threads of a warp busy for similar times.
"""
import ctypes as C
import numpy as np
from . import cspray

_MOD = {}


def module(fmad=False):
    """the compiled kernel module (NVRTC; once per fmad setting)."""
    import cupy as cp
    if fmad not in _MOD:
        _MOD[fmad] = cp.RawModule(code=cspray.SRC.read_text(), options=("--std=c++14", f"--fmad={'true' if fmad else 'false'}"))
    return _MOD[fmad]


def _tables(tbs):
    """device copies of the per-model tables and one device array of Tables structs pointing at them (arrays shared by object)."""
    import cupy as cp
    dev, keep, structs = {}, [], []

    def d(a):
        if id(a) not in dev:
            dev[id(a)] = cp.asarray(np.ascontiguousarray(np.asarray(a, np.float64)))
        return dev[id(a)]
    for tb in tbs:
        t, k = cspray.pack(tb); keep.append(k)
        for f, key in (("axi", "axi"), ("bar", "bar"), ("m", "m"), ("sat", "sat"), ("orb", "orb")):
            setattr(t, f, C.cast(C.c_void_p(d(tb[key] if key in tb else np.zeros(6)).data.ptr), cspray.DP))
        structs.append(bytes(t))
    return cp.asarray(np.frombuffer(b"".join(structs), np.uint8)), list(dev.values())


def integrate(models, acc, max_att=200000, order=None, fmad=False, block=128):
    """models: list of (ic (N_k, 6), t0 (N_k,), tb) for one or more models, integrated in one launch. order (optional): launch order
    of the concatenated particles. Returns a list of (final (N_k, 6), chi [AGAMA time x kpc], attempts (N_k,); -1 = failed)."""
    import cupy as cp
    n_k = [len(np.asarray(m[1])) for m in models]; n = sum(n_k)
    tbs, keep = _tables([m[2] for m in models])
    y0 = cp.asarray(np.concatenate([np.asarray(m[0], np.float64).reshape(-1, 6) for m in models]))
    t0 = cp.asarray(np.concatenate([np.asarray(m[1], np.float64) for m in models]))
    mi = cp.asarray(np.repeat(np.arange(len(models), dtype=np.int32), n_k))
    order = cp.arange(n, dtype=np.int32) if order is None else cp.asarray(np.asarray(order, np.int32))
    y, chi, natt = cp.zeros((n, 6)), cp.zeros(n), cp.zeros(n, np.int32)
    module(fmad).get_function("integrate_k")(((n+block-1)//block,), (block,),
                                             (tbs, mi, order, np.int32(n), y0, t0, np.float64(acc), np.int32(max_att), y, chi, natt))
    y, chi, natt = y.get(), chi.get(), natt.get(); e = np.cumsum([0]+n_k)
    return [(y[a:b], chi[a:b], natt[a:b]) for a, b in zip(e[:-1], e[1:])]


def warp_efficiency(natt, order=None, warp=32):
    """useful fraction of the thread-attempts in warps of the launch order: sum(attempts) / sum over warps of 32 x max(attempts)."""
    a = np.asarray(natt if order is None else np.asarray(natt)[order], np.float64); a = np.pad(a, (0, -len(a) % warp))
    return a.sum()/(warp*a.reshape(-1, warp).max(1).sum())
