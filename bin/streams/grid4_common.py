"""Shared set-up for the grid4 scripts: data (members, PM covariances, chord-frame coordinates), omega Cen profile, spray and
scoring helpers with the grid4 conventions (PMs fixed at the free-angle best, model A spray, release over the last 1000 Myr)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.restricted import AGAMA_T_MYR, host_potential
from ocen_dm.streams.frames import observables, to_model, OCEN_OBS
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score_conditional
from ocen_dm.streams.path import project_gc

PMRA, PMDEC = -3.2223, -6.7517
d = load_data(); nd = len(d["l"])
cov = np.zeros((nd, 2, 2)); cov[:, 0, 0] = d["e_pmra"]**2; cov[:, 1, 1] = d["e_pmdec"]**2; cov[:, 0, 1] = cov[:, 1, 0] = d["rho"]*d["e_pmra"]*d["e_pmdec"]
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
du, dx = project_gc(d["l"], d["b"], GC); dW = np.column_stack((dx, d["pmra"], d["pmdec"])); win = (du.min(), du.max())
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
T = 1955.58/AGAMA_T_MYR
_vr = json.loads((ROOT/"results/streams/spin/A_nodm_vrot.json").read_text())
SPIN = dict(r_kpc=np.array(_vr["r_pc"])/1e3, vrot_kms=np.array(_vr["vrot_kms"]), s=np.array(_vr["spin_vector_model"]))


def peri_release_times(host, today, nrel, window_myr=1000., sigma_myr=50.):
    """Release epochs [AGAMA units] concentrated at pericentres (Erkal+2019-style; width sigma_myr is our choice): deterministic
    quantiles of an equal-weight mixture of Gaussians (sigma_myr) centred on the centre orbit's pericentres within the last
    window_myr (+ 2 sigma margin), truncated to the window. Returns (t_release, pericentre ages [Myr])."""
    from ocen_dm.streams.restricted import agama_kpc
    agama = agama_kpc()
    _, back = agama.orbit(potential=host, ic=today, timestart=T, time=-T, trajsize=2, accuracy=1e-12)
    tc, orb = agama.orbit(potential=host, ic=back[-1], timestart=0., time=T, trajsize=int(T*AGAMA_T_MYR/0.5)+1, accuracy=1e-12)
    r = np.linalg.norm(orb[:, :3], axis=1); i = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0]+1
    age = (T-tc[i])*AGAMA_T_MYR; peri = age[age < window_myr+2*sigma_myr]
    x = np.linspace(0., window_myr, 20001); pdf = sum(np.exp(-0.5*((x-p)/sigma_myr)**2) for p in peri)
    cdf = np.cumsum(pdf); cdf /= cdf[-1]; ages = np.interp((np.arange(nrel)+0.5)/nrel, cdf, x)
    return np.sort(T-ages/AGAMA_T_MYR), peri


def make_spray(om, an, am, dist, nrel=8000, seed=1, spin=None, pm=None, host_kw=None, window_myr=1000., release="uniform", peri_sigma_myr=10., frame="baumgardt", proj_frame=None, mass_factor=1.0, max_age_myr=None, accuracy=1e-8, escape_recipe="fardal", arm_only=1, epoch_offset=None, host=None):
    """Trailing-arm spray (observables, chi, age). host_kw: extra host_potential options (bar_eta, bar_size, bar_model).
    release: 'uniform' (epochs uniform over the last window_myr) or 'peri' (concentrated at pericentres, peri_release_times,
    Gaussian width peri_sigma_myr; omega Cen's pericentres are ~88 Myr apart). frame: solar frame (frames.FRAMES) used both
    to place omega Cen in the model and to project the debris back to observables; proj_frame (default = frame) overrides the
    projection frame only. mass_factor scales omega Cen's enclosed-mass profile M(<r) at fixed radii (model A: 3.26e6 Msun).
    Speed: only the trailing arm is integrated (it is all that is returned; output identical to before). max_age_myr (opt-in,
    e.g. 700 = score()'s default age cut, ~2x faster again): integrate/return only debris released < max_age_myr ago; the kept
    particles are identical to the same particles of a full run, and the dict then carries age_cut_myr, which score() checks.
    accuracy: AGAMA orbit tolerance of the particles (default 1e-8 as before).
    escape_recipe: 'fardal' (default unchanged) or 'chen24' (actual satellite escape speed, radial release motion).
    arm_only=None returns both launch arms, allowing the score to select trailing chi regardless of launch label.
    epoch_offset (release='uniform' only; default None = the legacy grid, linspace(T - window, T, nrel)): u in [0, 1)
    gives a systematic grid with a random start, t_k = T - window + (k + u) window/nrel, k = 0..nrel-1, so independent
    replicates (u drawn per seed) re-randomise the release-epoch sampling as well as the escape offsets (2026-10-18).
    host (optional, 2026-10-22): an explicit AGAMA host (e.g. McMillan17); om, an, am, host_kw are then ignored."""
    pa, pd = (PMRA, PMDEC) if pm is None else pm
    host = host_potential("x", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am, **(host_kw or {})) if host is None else host
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], dist, pa, pd, OCEN_OBS["vlos"], frame)[0]
    if release == "uniform" and epoch_offset is not None:
        if not 0. <= epoch_offset < 1.:
            raise ValueError("epoch_offset must be in [0, 1)")
        trel = T-window_myr/AGAMA_T_MYR+(np.arange(nrel)+epoch_offset)*(window_myr/AGAMA_T_MYR)/nrel
    elif release == "uniform":
        trel = np.linspace(T-window_myr/AGAMA_T_MYR, T, nrel)
    else:
        if epoch_offset is not None:
            raise ValueError("epoch_offset applies to release='uniform' only")
        trel = peri_release_times(host, today, nrel, window_myr, peri_sigma_myr)[0]
    sp = spray_unwrapped(host, today, T, r_kpc, M*mass_factor, trel, seed=seed, spin=spin, arm_only=arm_only, max_age_myr=max_age_myr,
                         accuracy=accuracy, with_psi=False, escape_recipe=escape_recipe)
    tr = np.ones(len(sp["arm"]), bool) if arm_only is None else sp["arm"] == arm_only
    o = observables(sp["xv"][tr], proj_frame or frame)
    out = dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], d=o["dist"], chi=sp["chi"][tr], age=sp["t_release_myr_ago"][tr])
    if max_age_myr is not None:
        out["age_cut_myr"] = np.float64(max_age_myr)
    if escape_recipe != "fardal":
        out["escape_recipe"] = np.asarray(escape_recipe)
    if arm_only is None:
        out["arm_origin"] = sp["arm"][tr]
    return out


# Background support (2026-10-17, Codex review 2026-10-09): the uniform background is normalised over an explicit box that
# contains every member. Delta phi2: member range padded by 1 deg (-9.5..22.3 vs the old implied 12-deg width, which 119
# near-cluster members exceeded); PMs and v_los keep their old widths (30, 30 mas/yr; 400 km/s) as boxes containing all
# members, so only the Delta phi2 normalisation changes. score(m, legacy_background=True) replays earlier results.
BG_BOUNDS = ((float(np.floor(dx.min()))-1., float(np.ceil(dx.max()))+1.),
             (float(np.floor(d["pmra"].min()))-1., float(np.floor(d["pmra"].min()))-1.+30.),
             (float(np.floor(d["pmdec"].min()))-1., float(np.floor(d["pmdec"].min()))-1.+30.))
V_BOUNDS = (0., 400.)


# Bin weighting (2026-10-17, user): pooled=True is the default (one KDE of all debris; converges with particle number,
# particle_convergence.py / neff_study.py). legacy_bins=True restores the equal-weight chi bins of all results before
# 2026-10-17; legacy_background=True (exact replay of earlier results) implies it unless legacy_bins is given.
def score(m, legacy_background=False, legacy_bins=None, **kw):
    if legacy_bins is None:
        legacy_bins = legacy_background
    if not legacy_bins and "chi_edges" not in kw:
        kw.setdefault("pooled", True)
    if "age_cut_myr" in m and kw.get("age_max", 700.) > float(m["age_cut_myr"]):
        raise ValueError(f"spray integrated only to {float(m['age_cut_myr']):g} Myr (make_spray max_age_myr) < age_max")
    mu, mx = project_gc(m["l"], m["b"], GC)
    if not legacy_background:
        kw.setdefault("w_bounds", BG_BOUNDS); kw.setdefault("v_bounds", V_BOUNDS)
    res = score_conditional(du, dW, d["v"], d["e_v"], cov, mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"], m["chi"], m["age"], u_window=win, **kw)
    res["settings"] = score_settings(legacy_background, legacy_bins, kw)
    return res


def score_settings(legacy_background, legacy_bins, kw):
    """Provenance of a score (docs/kb/models/stream_likelihood_versions.md): version tag, every non-default option, code commit.
    Scripts that save scores should store this dict with their results."""
    if kw.get("pooled"):
        weighting = "pooled"
    elif kw.get("chi_edges") is not None:
        weighting = "chi_edges"
    else:
        weighting = "legacy_bins"
    tag = {("legacy_bins", True): "S2", ("legacy_bins", False): "S3"}.get((weighting, legacy_background),
                                                                               f"S4-{weighting}"+("-legacy_bg" if legacy_background else ""))
    opts = {k: (np.asarray(v).tolist() if isinstance(v, (np.ndarray, tuple, list)) else v) for k, v in kw.items()
            if k not in ("return_neff", "return_components", "chunk")}
    return dict(version=tag, weighting=weighting, legacy_background=bool(legacy_background), options=opts, commit=_commit())


@__import__("functools").lru_cache(maxsize=1)
def _commit():
    import subprocess
    try:
        c = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src/ocen_dm/streams", "bin/streams/grid4_common.py"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return c+("-dirty" if dirty else "")
    except Exception:
        return "unknown"
