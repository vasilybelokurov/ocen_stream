"""Regression for the spray speed-ups (2026-10-11): outputs must not change.
- make_spray default (only the trailing arm integrated) == stored pre-change full runs, every particle, every column;
- make_spray(max_age_myr=700) == the same stored runs restricted to age < 700 Myr, identical lnL;
- spray_unwrapped(arm_only, max_age_myr) == subset of a full run, incl. cluster spin and pericentre-concentrated release;
- score() refuses an age_max beyond a spray's explicit age cut.
Stored references in results/streams/ (skipped if absent). Run: OMP_NUM_THREADS=8 python -m pytest tests/test_spray_subset.py -q
(~1.5 min)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
G = pytest.importorskip("grid4_common")
KEYS = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")
# The stored runs predate the single cluster reference (commit 9601ba7, 2026-10-15), which moved the default state
# (centre, PMs at the 1e-4 level, v_los 232.78 -> 232.7 km/s). Byte-identity is tested at the state they were made with.
LEGACY_OBS = dict(ra=201.69683, dec=-47.47958, pmra=-3.2499, pmdec=-6.7461, vlos=232.78)


@pytest.fixture(autouse=True)
def _legacy_state(monkeypatch):
    monkeypatch.setattr(G, "OCEN_OBS", dict(LEGACY_OBS))

# (stored file, make_spray args, kwargs) -- all made by make_spray before the change (4x particles)
REF = [("results/streams/cluster_mass/mf1_s1.npz", (36., 28., 1.2, 5.6),
        dict(nrel=32000, seed=1, pm=(LEGACY_OBS["pmra"], LEGACY_OBS["pmdec"]), host_kw=dict(bar_size=1.15), mass_factor=1.0)),
       ("results/streams/sun_dist_bar/T1_R16_om34.5_an16_am1.4_sz1_ibata19_d5.43.npz", (34.5, 16., 1.4, 5.43),
        dict(nrel=32000, seed=1, pm=(G.PMRA, G.PMDEC), host_kw=dict(bar_size=1.0), frame="ibata19"))]


def _load(fname):
    f = ROOT/fname
    if not f.exists():
        pytest.skip(f"{fname} not present")
    return dict(np.load(f))


@pytest.mark.parametrize("fname,args,kw", REF[:1])
def test_default_output_unchanged(fname, args, kw):
    old = _load(fname); new = G.make_spray(*args, **kw)
    assert set(new) == set(old)
    for q in KEYS:
        np.testing.assert_array_equal(new[q], old[q], err_msg=q)


@pytest.mark.parametrize("fname,args,kw", REF)
def test_age_cut_matches_stored_full_run(fname, args, kw):
    old = _load(fname); new = G.make_spray(*args, max_age_myr=700., **kw)
    k = old["age"] < 700.
    assert len(new["age"]) == k.sum() and float(new["age_cut_myr"]) == 700.
    for q in KEYS:
        np.testing.assert_array_equal(new[q], old[q][k], err_msg=q)
    assert G.score(new)["total"] == G.score(old)["total"]


def _host_today():
    host = G.host_potential("x", bar_omega=36., bar_angle_deg=28., t_today=G.T, bar_amp=1.2)
    today = G.to_model(G.OCEN_OBS["ra"], G.OCEN_OBS["dec"], 5.6, G.PMRA, G.PMDEC, G.OCEN_OBS["vlos"])[0]
    return host, today


@pytest.mark.parametrize("case", ["uniform", "spin", "peri"])
def test_subset_options_bit_identical(case):
    from ocen_dm.streams.spray import spray_unwrapped
    host, today = _host_today()
    if case == "peri":
        trel = G.peri_release_times(host, today, 600, 1000., 10.)[0]
    else:
        trel = np.linspace(G.T-1000/G.AGAMA_T_MYR, G.T, 600)
    spin = G.SPIN if case == "spin" else None
    full = spray_unwrapped(host, today, G.T, G.r_kpc, G.M, trel, seed=3, spin=spin)
    sub = spray_unwrapped(host, today, G.T, G.r_kpc, G.M, trel, seed=3, spin=spin, arm_only=1, max_age_myr=700.)
    k = (full["arm"] == 1) & (full["t_release_myr_ago"] < 700.)
    for q in ("xv", "chi", "psi", "t_release_myr_ago", "arm"):
        np.testing.assert_array_equal(sub[q], full[q][k], err_msg=q)
    nopsi = spray_unwrapped(host, today, G.T, G.r_kpc, G.M, trel, seed=3, spin=spin, arm_only=1, with_psi=False)
    k1 = full["arm"] == 1
    for q in ("xv", "chi", "t_release_myr_ago"):
        np.testing.assert_array_equal(nopsi[q], full[q][k1], err_msg=q)


def test_chunked_integration_bit_identical():
    """spray_unwrapped(chunk=...) bounds memory; chunk sizes 1, 7 and > N must give identical outputs."""
    from ocen_dm.streams.spray import spray_unwrapped
    host, today = _host_today()
    trel = np.linspace(G.T-1000/G.AGAMA_T_MYR, G.T, 300)
    runs = [spray_unwrapped(host, today, G.T, G.r_kpc, G.M, trel, seed=3, chunk=c) for c in (10**9, 7, 1)]
    for r in runs[1:]:
        for q in ("xv", "chi", "psi", "t_release_myr_ago", "arm"):
            np.testing.assert_array_equal(r[q], runs[0][q], err_msg=q)


def test_score_refuses_age_beyond_cut():
    m = dict(l=np.zeros(3), b=np.zeros(3), pmra=np.zeros(3), pmdec=np.zeros(3), vlos=np.zeros(3), d=np.ones(3), chi=np.ones(3),
             age=np.array([100., 400., 490.]), age_cut_myr=np.float64(500.))
    with pytest.raises(ValueError):
        G.score(m, age_max=600.)
    with pytest.raises(ValueError):
        G.score(m)                       # default age_max 700 > cut 500


def test_score_default_is_pooled():
    """grid4_common.score defaults to pooled=True (2026-10-17); legacy_bins / legacy_background restore the chi bins."""
    m = _load(REF[0][0])
    s = G.score(m)
    assert s["K"] == 1 and s["total"] == G.score(m, pooled=True)["total"]
    leg = G.score(m, legacy_bins=True)
    assert leg["K"] > 1 and leg["total"] != s["total"]
    assert G.score(m, legacy_background=True)["K"] == leg["K"]
    assert G.score(m, chi_edges=np.arange(0., 205., 5.))["K"] == 40


def test_score_settings_tags():
    m = _load(REF[0][0])
    assert G.score(m)["settings"]["version"] == "S4-pooled"
    assert G.score(m, legacy_bins=True)["settings"]["version"] == "S3"
    assert G.score(m, legacy_background=True)["settings"]["version"] == "S2"
    assert G.score(m, chi_edges=np.arange(0., 205., 5.))["settings"]["version"] == "S4-chi_edges"


def test_epoch_offset_default_unchanged_and_grid_shifted():
    """make_spray(epoch_offset=None) is the legacy grid (bit-identical); an offset u gives T - W + (k + u) W / n."""
    a = G.make_spray(35., 25., 1.2, 5.6, nrel=200, seed=3, max_age_myr=700.)
    b = G.make_spray(35., 25., 1.2, 5.6, nrel=200, seed=3, max_age_myr=700., epoch_offset=None)
    for q in KEYS:
        np.testing.assert_array_equal(a[q], b[q], err_msg=q)
    c = G.make_spray(35., 25., 1.2, 5.6, nrel=200, seed=3, max_age_myr=700., epoch_offset=0.37)
    W = 1000.; ages = np.unique(np.round(c["age"], 6))
    expect = np.sort(W-(np.arange(200)+0.37)*W/200)
    expect = expect[expect < 700.]
    np.testing.assert_allclose(np.sort(ages), np.sort(np.round(expect, 6)), atol=1e-4)
    with pytest.raises(ValueError):
        G.make_spray(35., 25., 1.2, 5.6, nrel=20, seed=3, epoch_offset=1.0)
