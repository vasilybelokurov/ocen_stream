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

# (stored file, make_spray args, kwargs) -- all made by make_spray before the change (4x particles)
REF = [("results/streams/cluster_mass/mf1_s1.npz", (36., 28., 1.2, 5.6),
        dict(nrel=32000, seed=1, pm=(G.OCEN_OBS["pmra"], G.OCEN_OBS["pmdec"]), host_kw=dict(bar_size=1.15), mass_factor=1.0)),
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


def test_score_refuses_age_beyond_cut():
    m = dict(l=np.zeros(3), b=np.zeros(3), pmra=np.zeros(3), pmdec=np.zeros(3), vlos=np.zeros(3), d=np.ones(3), chi=np.ones(3),
             age=np.array([100., 400., 490.]), age_cut_myr=np.float64(500.))
    with pytest.raises(ValueError):
        G.score(m, age_max=600.)
    with pytest.raises(ValueError):
        G.score(m)                       # default age_max 700 > cut 500
