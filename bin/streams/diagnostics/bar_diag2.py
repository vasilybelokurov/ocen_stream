"""Read-only diagnostic 2: (a) m=2 force ratio of the fitted vs published bars; (b) E_J / E / Lz of the 25-deg 'branch' vs
on-track debris (age 300-350 Myr) from the stored step-2 sprays; (c) orbit divergence (shadow-orbit) of omega Cen's orbit and
of on-track debris in controlled host variants; (d) when the branch and track groups separate (back-integration, 25 deg)."""
import os, sys, time, numpy as np
ROOT = os.getcwd()
sys.path[:0] = [ROOT+"/bin/streams", ROOT+"/src"]
import grid4_common as g
from ocen_stream.restricted import host_potential, agama_kpc, AGAMA_T_MYR, HUNTER24_DIR
from ocen_stream.mw_variants import axisymmetric_variant
from ocen_stream.frames import to_model, OCEN_OBS, frame_r0
from ocen_stream.path import project_gc
import astropy.coordinates as coord, astropy.units as u
agama = agama_kpc()
T = g.T; R0 = frame_r0("gravity21")
MODELS = {
 "25":   dict(angle=25.0, omega=35.411964, amp=1.3842206, size=1.2356600, dist=5.523893, pmra=-3.2616917, pmdec=-6.7332134, vc=223.00973, halo_q=0.9718893),
 "12.5": dict(angle=12.5, omega=33.140546, amp=1.3652125, size=1.3284842, dist=5.526422, pmra=-3.2612456, pmdec=-6.7595030, vc=225.03203, halo_q=0.8960698),
}
NPZ = {"25": "results/streams/profile_step2_20261020/runs/final_a25_1.npz", "12.5": "results/streams/profile_step2_20261020/runs/final_a12.5_3.npz"}

def make_host(p, **over):
    q = dict(p, **over)
    return host_potential("x", bar_omega=q["omega"], bar_angle_deg=q["angle"], t_today=T, bar_amp=q["amp"], bar_size=q["size"],
                          axi_variant=dict(halo_q=q["halo_q"], vc_target=q["vc"], r0_kpc=R0))

def force_ratio(host, axi, label):
    n = len(host); nonaxi = agama.Potential(host[n-2], host[n-1])
    for z in (0.0, 1.5):
        row = []
        for Rr in (1.5, 2, 3, 4, 5, 6, 7):
            ph = np.linspace(0, 2*np.pi, 73)
            pts = np.column_stack((Rr*np.cos(ph), Rr*np.sin(ph), np.full_like(ph, z)))
            fn = np.linalg.norm(nonaxi.force(pts, t=T), axis=1).max(); fa = np.linalg.norm(axi.force(pts), axis=1).mean()
            row.append(round(float(fn/fa), 3))
        print("  %-28s z=%.1f  |F_m>0|max/|F_axi| at R=1.5,2,3,4,5,6,7:" % (label, z), row)

print("=== (a) non-axisymmetric force ratio ===")
axi0 = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
for bm in ("hunter24", "portail17"):
    h = host_potential("x", bar_omega=37.5, bar_angle_deg=25., t_today=T, bar_amp=1.0, bar_model=bm)
    force_ratio(h, axi0, "%s A=1 S=1 (Hunter axi)" % bm)
for key, p in MODELS.items():
    axi = axisymmetric_variant(HUNTER24_DIR, halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=R0)[0]
    force_ratio(make_host(p), axi, "best %s deg (A %.2f S %.2f)" % (key, p["amp"], p["size"]))
    force_ratio(make_host(p, amp=1.0, size=1.0), axi, "same axi, A=1 S=1")

# ---------------------------------------------------------------------------------------------------------------
def load_debris(key):
    z = np.load(NPZ[key]); m = {k: z[k] for k in z.files if z[k].ndim}
    uu, xx = project_gc(m["l"], m["b"], g.GC)
    gal = coord.SkyCoord(l=m["l"]*u.deg, b=m["b"]*u.deg, frame="galactic").icrs
    m["ra"], m["dec"], m["u"], m["x"] = gal.ra.deg, gal.dec.deg, uu, xx
    return m

# members' running medians per 4-deg phi1 bin
edges = np.arange(2, 30, 4)
def med_members(col):
    return np.array([np.median(col[(g.du >= a) & (g.du < b)]) if np.sum((g.du >= a) & (g.du < b)) > 5 else np.nan for a, b in zip(edges[:-1], edges[1:])])
mpm = med_members(g.d["pmra"]); mx = med_members(g.dx); mpd = med_members(g.d["pmdec"])

def classify(m, a0=300., a1=350.):
    ib = np.clip(np.digitize(m["u"], edges)-1, 0, len(edges)-2)
    ok = (m["chi"] > 0) & (m["age"] >= a0) & (m["age"] < a1) & (m["u"] >= 6) & (m["u"] < 22) & (np.abs(m["x"]) < 15)
    branch = ok & (m["pmra"] < mpm[ib]-3.)
    track = ok & (np.abs(m["x"]-mx[ib]) < 3.) & (np.abs(m["pmra"]-mpm[ib]) < 2.) & (np.abs(m["pmdec"]-mpd[ib]) < 2.)
    return ok, branch, track

def xv_of(m, sel, p):
    return to_model(m["ra"][sel], m["dec"][sel], m["d"][sel], m["pmra"][sel], m["pmdec"][sel], m["vlos"][sel], "gravity21")

def EJ_stats(host, xv, omb, label):
    E = 0.5*np.sum(xv[:, 3:]**2, axis=1) + host.potential(xv[:, :3], t=T)
    Lz = xv[:, 0]*xv[:, 4]-xv[:, 1]*xv[:, 3]; EJ = E-omb*Lz
    q = lambda a: np.round(np.percentile(a, [16, 50, 84]), 0)
    print("  %-10s N=%5d  E_J 16/50/84: %s   E: %s   Lz: %s   r_today 16/50/84: %s" % (label, len(xv), q(EJ), q(E), q(Lz), np.round(np.percentile(np.linalg.norm(xv[:, :3], axis=1), [16, 50, 84]), 2)))
    return E, Lz, EJ

print("\n=== (b) Jacobi energy of branch vs on-track debris (age 300-350 Myr, phi1 6-22) ===")
rng = np.random.default_rng(1)
HOSTS = {k: make_host(p) for k, p in MODELS.items()}
today = {k: to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"], "gravity21")[0] for k, p in MODELS.items()}
groups = {}
for key, p in MODELS.items():
    m = load_debris(key); ok, br, tr = classify(m)
    print(" model %s: selected %d, branch %d (%.2f), track %d (%.2f)" % (key, ok.sum(), br.sum(), br.sum()/ok.sum(), tr.sum(), tr.sum()/ok.sum()))
    xo = today[key]; Eo = 0.5*np.sum(xo[3:]**2)+HOSTS[key].potential(xo[:3], t=T); Lo = xo[0]*xo[4]-xo[1]*xo[3]
    print("  omega Cen: E %.0f Lz %.0f E_J %.0f" % (Eo, Lo, Eo-p["omega"]*Lo))
    for lab, sel in (("all", ok), ("branch", br), ("track", tr)):
        xv = xv_of(m, sel, p); EJ_stats(HOSTS[key], xv, p["omega"], lab)
    groups[key] = dict(branch=xv_of(m, br, p), track=xv_of(m, tr, p), ages_b=m["age"][br], ages_t=m["age"][tr])
    # the 'all' sample with ages 250-500 (the sheet population)
    ok2, br2, tr2 = classify(m, 250., 500.)
    print("  age 250-500: selected %d branch %.2f track %.2f" % (ok2.sum(), br2.sum()/ok2.sum(), tr2.sum()/ok2.sum()))

# ---------------------------------------------------------------------------------------------------------------
print("\n=== (c) shadow-orbit divergence over 1 Gyr (ln(d/d0), d0 = 1e-6 kpc) ===")
def divergence(host, ics, tmax_myr=1000., d0=1e-6):
    tb = tmax_myr/AGAMA_T_MYR
    ic2 = ics.copy(); ic2[:, 0] += d0
    o1 = agama.orbit(potential=host, ic=ics, timestart=T, time=-tb, trajsize=2, accuracy=1e-10)
    o2 = agama.orbit(potential=host, ic=ic2, timestart=T, time=-tb, trajsize=2, accuracy=1e-10)
    out = []
    for a, b in zip(o1, o2):
        d = np.linalg.norm(a[1][-1, :3]-b[1][-1, :3]); out.append(np.log(max(d, 1e-300)/d0))
    return np.array(out)
# sample of on-track debris of the 25-deg model (today's phase space) used as test orbits in every host
tr25 = groups["25"]["track"]; sel = rng.choice(len(tr25), size=min(80, len(tr25)), replace=False); test = tr25[sel]
variants = {
 "best 25":                       make_host(MODELS["25"]),
 "best 12.5":                     make_host(MODELS["12.5"]),
 "25 params, angle 12.5":         make_host(MODELS["25"], angle=12.5),
 "12.5 params, angle 25":         make_host(MODELS["12.5"], angle=25.),
 "25 params, omega 33.14":        make_host(MODELS["25"], omega=33.140546),
 "25 params, A=1 S=1":            make_host(MODELS["25"], amp=1.0, size=1.0),
 "25 params, A=1 S=1.236":        make_host(MODELS["25"], amp=1.0),
 "25 params, A=1.384 S=1":        make_host(MODELS["25"], size=1.0),
 "12.5 params, A=1 S=1":          make_host(MODELS["12.5"], amp=1.0, size=1.0),
}
for lab, h in variants.items():
    t0 = time.time()
    dc = divergence(h, today["25"][None, :] if lab.startswith(("best 25", "25")) else today["12.5"][None, :])[0]
    dd = divergence(h, test)
    print("  %-26s omega Cen orbit ln(d/d0)=%5.1f | 80 on-track debris: median %5.1f, 84%% %5.1f, frac>10: %.2f  (%.0fs)"
          % (lab, dc, np.median(dd), np.percentile(dd, 84), np.mean(dd > 10), time.time()-t0))

# ---------------------------------------------------------------------------------------------------------------
print("\n=== (d0) frequencies (axisymmetric part) of branch vs track debris today ===")
for key, p in MODELS.items():
    axi = axisymmetric_variant(HUNTER24_DIR, halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=R0)[0]; af = agama.ActionFinder(axi)
    for lab in ("branch", "track"):
        xv = groups[key][lab]; J, ang, fr = af(xv, angles=True)
        rat = (p["omega"]-fr[:, 2])/fr[:, 0]; ratz = (p["omega"]-fr[:, 2])/fr[:, 1]
        print("  %s %-6s Jr %s Jz %s Jphi %s | Om_r %s Om_phi %s | (Ob-Op)/Or %s  (Ob-Op)/Oz %s" % (
            key, lab, np.round(np.percentile(J[:, 0], [16, 50, 84]), 0), np.round(np.percentile(J[:, 1], [16, 50, 84]), 0),
            np.round(np.percentile(J[:, 2], [16, 50, 84]), 0), np.round(np.percentile(fr[:, 0], [16, 50, 84]), 1),
            np.round(np.percentile(fr[:, 2], [16, 50, 84]), 1), np.round(np.percentile(rat, [16, 50, 84]), 3), np.round(np.percentile(ratz, [16, 50, 84]), 3)))

print("\n=== (d) back-integration of the 25-deg branch vs track (age 300-350): E, Lz, r, bar-relative azimuth, distance to omega Cen vs age ===")
for key in ("25", "12.5"):
    p = MODELS[key]; h = HOSTS[key]
    tb = 360./AGAMA_T_MYR
    tc, oc = agama.orbit(potential=h, ic=today[key], timestart=T, time=-tb, trajsize=361, accuracy=1e-10)
    for lab in ("branch", "track"):
        xv = groups[key][lab]; ages = groups[key]["ages_b" if lab == "branch" else "ages_t"]
        k = rng.choice(len(xv), size=min(120, len(xv)), replace=False); xv, ages = xv[k], ages[k]
        orbs = agama.orbit(potential=h, ic=xv, timestart=T, time=-tb, trajsize=361, accuracy=1e-10)
        tt = orbs[0][0]; agegrid = (T-tt)*AGAMA_T_MYR
        X = np.array([o[1] for o in orbs])              # (N, 361, 6)
        E = 0.5*np.sum(X[..., 3:]**2, axis=-1) + np.array([h.potential(X[i, :, :3], t=tt) for i in range(len(X))])
        Lz = X[..., 0]*X[..., 4]-X[..., 1]*X[..., 3]; r = np.linalg.norm(X[..., :3], axis=-1)
        bar = np.deg2rad(p["angle"]) + p["omega"]*(tt-T); phi = np.arctan2(X[..., 1], X[..., 0])
        rel = (np.rad2deg(phi-bar)+90) % 180-90
        dcl = np.linalg.norm(X[..., :3]-oc[None, :, :3], axis=-1)
        print(" model %s %s N %d ages 16/50/84: %s" % (key, lab, len(xv), np.round(np.percentile(ages, [16, 50, 84]), 0)))
        for a in (0, 60, 120, 180, 240, 280, 300, 320, 340):
            j = np.argmin(np.abs(agegrid-a))
            print("   age %3.0f: E %s  Lz %s  r %s  bar-rel az(med) %6.1f  |x - x_oCen| 16/50/84 %s" % (
                a, np.round(np.percentile(E[:, j], [16, 50, 84]), 0), np.round(np.percentile(Lz[:, j], [16, 50, 84]), 0),
                np.round(np.percentile(r[:, j], [16, 50, 84]), 2), np.median(rel[:, j]), np.round(np.percentile(dcl[:, j], [16, 50, 84]), 2)))
        # pericentres of each particle: ages, r, bar-relative azimuth
        pa, pr, pz = [], [], []
        for i in range(len(xv)):
            ip = np.where((r[i, 1:-1] < r[i, :-2]) & (r[i, 1:-1] < r[i, 2:]))[0]+1
            pa += list(agegrid[ip]); pr += list(r[i, ip]); pz += list(rel[i, ip])
        pa, pr, pz = map(np.array, (pa, pr, pz))
        for a0, a1 in ((0, 120), (120, 210), (210, 300), (300, 360)):
            s = (pa >= a0) & (pa < a1)
            if s.sum():
                print("   pericentres age %3d-%3d: N %3d  age med %5.0f  r_peri 16/50/84 %s  bar-rel az 16/50/84 %s" % (
                    a0, a1, s.sum(), np.median(pa[s]), np.round(np.percentile(pr[s], [16, 50, 84]), 2), np.round(np.percentile(pz[s], [16, 50, 84]), 0)))
