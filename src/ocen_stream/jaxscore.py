"""The 25-point knot score of bin/streams/stream_model_compare.py in JAX (GPU pipeline stage 3, 2026-10-25).

Same measurement of the model as the CPU score: particles with chi > 0 and age < 700 Myr, projected to the stream frame; each gets
the Gaia PM errors of a data star at a similar phi1 (error table sorted by phi1, random offset in [-50, 50) in rank) and
error-convolved PMs for the frozen PM masks; sky: per local window the area-weighted, Gaussian-smoothed (sigma 1 x 2 cells) histogram
scaled to the data's stream count, fitted by the same extended likelihood (Asimov: expected counts as data) for (c, s, r) with the
slope fixed; PMs: truncated-Gaussian fit with a linear trend within +-1.5 deg of each knot (intrinsic PMs, |v| < W); model errors =
sd over NBOOT bootstraps; chi2 as stream_model_compare.chi2 (+ optional log|C|).
Differences, by design, for a deterministic likelihood (common random numbers): all draws come from a bank fixed per particle
SLOT (error-rank offset, two normals, Poisson(1) bootstrap counts instead of multinomial resamples); the Nelder-Mead fits are
replaced by fixed-count damped Newton (Levenberg-Marquardt) iterations from the same starts. stream_model_compare.measure(bank=...)
with the same bank is the NumPy reference.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

RAD = np.pi/180


def load(path, dtype=jnp.float64):
    z = np.load(path); f = lambda a: jnp.asarray(a, dtype)
    S = {k[4:]: f(z[k]) for k in z.files if k.startswith("sky_") and k != "sky_drop"}
    k1 = lambda s: (lambda r: jnp.asarray(r/r.sum(), dtype))(np.exp(-0.5*(np.arange(-int(4*s+0.5), int(4*s+0.5)+1)/s)**2))
    return dict(Rm=f(z["Rm"]), coeffs=f(z["coeffs"]), G=f(z["G"]), pmra_c=f(z["pmra_c"]), pmra_s=f(z["pmra_s"]), pmdec_c=f(z["pmdec_c"]),
                pmdec_s=f(z["pmdec_s"]), win_c=f(z["win_c"]), win_hw=f(z["win_hw"]), err_u=f(z["err_u"]), err=f(z["err"]), sky=S,
                keep=np.nonzero(~z["sky_drop"])[0], smin=float(z["smin"]), smax=float(z["smax"]), nboot=int(z["nboot"]),
                D={k[2:]: f(z[k]) for k in z.files if k.startswith("D_")}, ku=k1(1.), kv=k1(2.))


def make_bank(n, nboot, seed):
    """fixed draws per particle slot: rank offsets, two normals, Poisson(1) bootstrap counts."""
    rng = np.random.default_rng(seed)
    return dict(off=rng.integers(-50, 50, n), nrm=rng.standard_normal((2, n)), boot=rng.poisson(1., (nboot, n)))


def project(l, b, si):
    """(l, b) [deg] -> along-stream phi1 and across-stream dphi2 = phi2 - poly(phi1) [deg] (path.project_gc)."""
    l, b = l*RAD, b*RAD; v = jnp.stack((jnp.cos(b)*jnp.cos(l), jnp.cos(b)*jnp.sin(l), jnp.sin(b)), -1)@si["Rm"].T
    f1 = jnp.arctan2(v[:, 1], v[:, 0])/RAD; f2 = jnp.arcsin(jnp.clip(v[:, 2], -1, 1))/RAD
    return f1, f2-jnp.polyval(si["coeffs"], f1)


def particles(obs, chi, age, si, bank, age_cut=700.):
    """per-slot quantities of the model measurement (stream_model_compare.Obs.particles); sel = chi > 0 & age < age_cut."""
    u, x = project(obs["l"], obs["b"], si); ev = lambda k: jnp.interp(u, si["G"], si[k])
    n = si["err_u"].shape[0]; j = jnp.clip(jnp.searchsorted(si["err_u"], u)+bank["off"], 0, n-1); e = si["err"][j]
    P = dict(u=u, sel=(chi > 0) & (age < age_cut), pmra=obs["pmra"], pmdec=obs["pmdec"])
    for i, q in enumerate(("pmra", "pmdec")):
        o = obs[q]+e[:, i]*bank["nrm"][i]
        P["m_"+q] = jnp.abs(o-ev(q+"_c")) < 2*jnp.sqrt(ev(q+"_s")**2+e[:, i]**2)
    P["v"] = x-ev("win_c"); P["inwin"] = jnp.abs(P["v"]) < ev("win_hw")
    return P


def _smooth(H, si):
    """scipy.ndimage.gaussian_filter(H, (1, 2)) (mode 'reflect', truncate 4) on the last two axes."""
    def conv(a, k, ax):
        r = (k.size-1)//2; a = jnp.moveaxis(a, ax, -1); pad = [(0, 0)]*(a.ndim-1)+[(r, r)]
        a = jnp.pad(a, pad, mode="symmetric"); out = sum(k[i]*a[..., i:i+a.shape[-1]-2*r] for i in range(k.size))
        return jnp.moveaxis(out, -1, ax)
    return conv(conv(H, si["ku"], -2), si["kv"], -1)


def _lm(f, p0, n_iter=40):
    """fixed-count Levenberg-Marquardt minimisation of a smooth scalar f(p) (small p): returns (p, f(p))."""
    g, Hf = jax.grad(f), jax.hessian(f)

    def it(_, s):
        p, fp, lam = s; d = jnp.linalg.solve(Hf(p)+lam*jnp.eye(p.size), g(p)); q = p-d; fq = f(q); ok = jnp.isfinite(fq) & (fq < fp)
        return jnp.where(ok, q, p), jnp.where(ok, fq, fp), jnp.where(ok, lam*0.3, lam*10.)
    p, fp, _ = jax.lax.fori_loop(0, n_iter, it, (p0, f(p0), 1e-3))
    return p, fp


def sky(P, w, si):
    """local Asimov sky fits (c, s) in all windows for particle weights w; NaN where the area-weighted count < 20."""
    S = si["sky"]; m = w*(P["m_pmra"] & P["m_pmdec"]); nwin = S["ug"].shape[0]; nu, nv = S["ug"].shape[1], S["vg"].shape[1]
    iu = jnp.floor((P["u"][None, :]-S["eu0"][:, None])/S["du"][:, None]).astype(jnp.int32)
    iv = jnp.floor((P["v"][None, :]-S["ev0"][:, None])/S["dv"][:, None]).astype(jnp.int32)
    ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    H = jnp.zeros((nwin, nu, nv)).at[jnp.arange(nwin)[:, None], jnp.clip(iu, 0, nu-1), jnp.clip(iv, 0, nv-1)].add(jnp.where(ok, m[None, :], 0.))
    Sd = _smooth(H, si)*S["Ag"]; ntot = Sd.sum((1, 2)); dA = S["du"]*S["dv"]
    lam_true = S["lb"]+Sd/jnp.maximum(ntot, 1e-30)[:, None, None]*(S["r"]*(S["a1"]-S["a0"])/dA)[:, None, None]
    smin, smax = si["smin"], si["smax"]

    def one(k):
        def enll(p):
            c, s, r = p[0], smin+(smax-smin)/(1+jnp.exp(-p[1])), jnp.exp(p[2]); mg = c+S["g"][k]*(S["ug"][k]-S["pk"][k])
            lf = S["lb"][k]+r*jnp.exp(-0.5*((S["vg"][k][None, :]-mg[:, None])/s)**2)/(s*jnp.sqrt(2*jnp.pi))*S["Ag"][k]
            return -jnp.sum(lam_true[k]*jnp.log(jnp.maximum(lf, 1e-30))-lf)*dA[k]
        fits = [_lm(enll, jnp.array([S["c"][k], xs0, jnp.log(S["r"][k])])) for xs0 in (-1.5, 0., 1.5)]
        F = jnp.array([f[1] for f in fits]); p = jnp.stack([f[0] for f in fits])[jnp.argmin(F)]
        return p[0], smin+(smax-smin)/(1+jnp.exp(-p[1]))
    c, s = jax.vmap(one)(jnp.arange(nwin))
    bad = ntot < 20
    return jnp.where(bad, jnp.nan, c), jnp.where(bad, jnp.nan, s)


def pm(P, w, si):
    """truncated-Gaussian fits with a linear trend at the PM knots (intrinsic PMs relative to the frozen track, |v| < W)."""
    out = {}
    for q, oth in (("pmra", "pmdec"), ("pmdec", "pmra")):
        W = si["D"][q+"_W"]; vq = P[q]-jnp.interp(P["u"], si["G"], si[q+"_c"])
        base = w*(P["inwin"] & P["m_"+oth] & (jnp.abs(vq) < W))

        def one(c0):
            x = P["u"]-c0; wk = base*(jnp.abs(x) < 1.5); n = wk.sum(); mu = jnp.sum(wk*vq)/jnp.maximum(n, 1)
            sd = jnp.sqrt(jnp.sum(wk*(vq-mu)**2)/jnp.maximum(n, 1))

            def nl(p):
                m = p[0]+p[1]*x; sg = jnp.exp(p[2]); Z = ndtr((W-m)/sg)-ndtr((-W-m)/sg)
                return -jnp.sum(wk*(-0.5*((vq-m)/sg)**2-p[2]-jnp.log(jnp.maximum(Z, 1e-12))))
            p, _ = _lm(nl, jnp.array([mu, 0., jnp.log(jnp.maximum(sd, 0.1))]))
            return jnp.where(n < 20, jnp.nan, p[0])
        out[q] = jax.vmap(one)(si["D"][q+"_k"])
    return out


def measure(obs, chi, age, si, bank):
    """model summaries (sky c, s in the kept windows, pmra, pmdec at the knots) and their bootstrap sd."""
    P = particles(obs, chi, age, si, bank); kw = si["keep"]

    def vec(w):
        c, s = sky(P, w, si); q = pm(P, w, si)
        return jnp.concatenate((c[kw], s[kw], q["pmra"], q["pmdec"]))
    sel = P["sel"].astype(float)
    V = jax.vmap(vec)(jnp.concatenate((sel[None], sel[None]*bank["boot"]), 0))       # all bootstraps in one batch (was lax.map)
    return V[0], jnp.nanstd(V[1:], 0)


def chi2(m, e, si, logdet=False):
    """stream_model_compare.chi2 (missing value: 1e3 penalty); optionally + log|C_data + C_model| - log|C_data|."""
    D = si["D"]; k = D["pk"].shape[0]; na = D["pmra_k"].shape[0]; tot = 0.; ld = 0.
    for sl, q in ((slice(0, k), "sky_c"), (slice(k, 2*k), "sky_s")):
        de = D[q+"e"]
        var = de**2+e[sl]**2; r = (D[q]-m[sl])**2/var; ok = jnp.isfinite(r)
        tot = tot+jnp.sum(jnp.where(ok, r, 0.))+1e3*jnp.sum(~ok); ld = ld+jnp.sum(jnp.where(ok, jnp.log1p(e[sl]**2/de**2), 0.))
    for sl, q in ((slice(2*k, 2*k+na), "pmra"), (slice(2*k+na, None), "pmdec")):
        r = D[q]-m[sl]; ok = jnp.isfinite(r); r0 = jnp.where(ok, r, 0.)
        C = D[q+"_C"]+jnp.diag(jnp.where(ok, e[sl], 0.)**2); C = jnp.where(ok[:, None] & ok[None, :], C, jnp.eye(r.size))
        tot = tot+r0@jnp.linalg.solve(C, r0)+1e3*jnp.sum(~ok); ld = ld+jnp.linalg.slogdet(C)[1]-jnp.linalg.slogdet(D[q+"_C"])[1]
    return tot+(ld if logdet else 0.)
