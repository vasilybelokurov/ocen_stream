#!/usr/bin/env python3
"""Inspect sky-conditioned PM residuals of saved bar-profile candidates.

No integrations. Model predictions use sky coordinates only to select
particles; observed proper motions never select the model particles.
PMs are decomposed along/across the measured sky ridge, and into a change
parallel/perpendicular to the model PM vector. A uniform distance rescaling
at fixed space velocity changes the PM-vector length, not its direction.
The angular diagnostic does not assume that a barred stream follows an orbit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin/streams"))

import astropy.coordinates as coord
import astropy.units as units
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grid4_common import GC, d, du, dx, project_gc
from ocen_dm.streams.path import gc_rotation, to_stream


def pm_basis(l, b):
    """ICRS PM basis vectors projected onto the measured ridge tangent/normal."""
    l, b = np.asarray(l), np.asarray(b)
    sky = coord.SkyCoord(l=l * units.deg, b=b * units.deg, frame="galactic").icrs
    rotation = gc_rotation(GC["pole"], GC["origin"])
    f1, f2 = to_stream(l, b, rotation)
    f1r, f2r = np.radians(f1), np.radians(f2)
    e1 = np.column_stack((-np.sin(f1r), np.cos(f1r), f1r * 0))
    e2 = np.column_stack((-np.sin(f2r) * np.cos(f1r), -np.sin(f2r) * np.sin(f1r), np.cos(f2r)))
    slope = np.polyval(np.polyder(GC["coeffs"]), f1)
    tangent = np.cos(f2r)[:, None] * e1 + slope[:, None] * e2
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    radial = np.column_stack((np.cos(f2r) * np.cos(f1r), np.cos(f2r) * np.sin(f1r), np.sin(f2r)))
    normal = np.cross(radial, tangent)
    lr, br = np.radians(l), np.radians(b)
    el = np.column_stack((-np.sin(lr), np.cos(lr), lr * 0))
    eb = np.column_stack((-np.sin(br) * np.cos(lr), -np.sin(br) * np.sin(lr), np.cos(br)))
    matrices = np.empty((len(l), 2, 2))
    for j, (pra, pdec) in enumerate(((1., 0.), (0., 1.))):
        c = coord.SkyCoord(ra=sky.ra, dec=sky.dec, pm_ra_cosdec=np.full(len(l), pra) * units.mas / units.yr,
                           pm_dec=np.full(len(l), pdec) * units.mas / units.yr).galactic
        vector = (np.asarray(c.pm_l_cosb.value)[:, None] * el + np.asarray(c.pm_b.value)[:, None] * eb) @ rotation.T
        matrices[:, 0, j] = np.sum(vector * tangent, axis=1)
        matrices[:, 1, j] = np.sum(vector * normal, axis=1)
    return matrices


def quantile(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0:
        return np.full(3, np.nan)
    cumulative = (cumulative - .5 * weights[order]) / cumulative[-1]
    return np.interp([.16, .5, .84], cumulative, values[order])


def particle_weights(m):
    keep = (m["chi"] > 0) & (m["age"] < 700)
    edges = np.arange(0., m["chi"][keep].max() + 5., 5.)
    bins = [np.where(keep & (m["chi"] >= edge) & (m["chi"] < edge + 5))[0] for edge in edges]
    bins = [indices for indices in bins if len(indices) >= 20]
    idx = np.concatenate(bins)
    weight = np.concatenate([np.full(len(indices), 1. / (len(bins) * len(indices))) for indices in bins])
    return idx, weight


def summarize(m, edges, basis):
    idx, weights = particle_weights(m)
    mu, mx = project_gc(m["l"][idx], m["b"][idx], GC)
    pm = np.column_stack((m["pmra"][idx], m["pmdec"][idx]))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Display ridge members in a declared 2-degree tube. This never changes
        # the all-member likelihood or fit; diagnostics report member counts.
        k = (du >= lo) & (du < hi) & (np.abs(dx) < 2.)
        if k.sum() < 10:
            continue
        matrix = basis[k].mean(axis=0)
        transform = lambda p: p @ matrix.T
        dp = np.column_stack((d["pmra"][k], d["pmdec"][k]))
        dp_stream, mp_stream = transform(dp), transform(pm)
        gu = np.exp(-.5 * ((du[k, None] - mu) / .5)**2)
        gx = np.exp(-.5 * ((dx[k, None] - mx) / .5)**2)
        kernel = gu * gx * weights
        normalizer = kernel.sum(axis=1)
        covered = normalizer > 1e-30
        if not covered.any():
            continue
        weight = (kernel[covered] / normalizer[covered, None]).mean(axis=0)
        obs = dict(pmra=np.percentile(dp[:, 0], [16, 50, 84]),
                   pmdec=np.percentile(dp[:, 1], [16, 50, 84]),
                   parallel=np.percentile(dp_stream[:, 0], [16, 50, 84]),
                   normal=np.percentile(dp_stream[:, 1], [16, 50, 84]))
        mod = dict(pmra=quantile(pm[:, 0], weight), pmdec=quantile(pm[:, 1], weight),
                   parallel=quantile(mp_stream[:, 0], weight), normal=quantile(mp_stream[:, 1], weight),
                   distance=quantile(m["d"][idx], weight))
        a = np.array([obs[q][1] for q in ("pmra", "pmdec")])
        z = np.array([mod[q][1] for q in ("pmra", "pmdec")])
        scale = float((a @ z) / (z @ z))
        perp = a - scale * z
        angle = float(np.degrees(np.arctan2(z[0] * a[1] - z[1] * a[0], z @ a)))
        rows.append(dict(lo=float(lo), hi=float(hi), n=int(k.sum()), covered=int(covered.sum()),
                         observed={key: value.tolist() for key, value in obs.items()},
                         model={key: value.tolist() for key, value in mod.items()},
                         pm_direction_offset_deg=angle, pm_length_ratio=scale,
                         residual_after_local_distance_scaling=perp.tolist(),
                         rms_distance_only_residual=float(np.linalg.norm(perp))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="bar_profile_refinement_20261013")
    args = ap.parse_args()
    outdir = ROOT / "results/plot_data" / args.label
    candidates = {}
    for an in (16, 28):
        payload = json.loads((outdir / f"fit_an{an}.json").read_text())
        candidates[f"{an} deg initial"] = max((r for r in payload["log"] if r["stage"] == "initial"), key=lambda r: r["lnpenalized"])
        candidates[f"{an} deg refined"] = payload["best"]
    basis = pm_basis(d["l"], d["b"])
    edges = np.arange(4., 28.1, 2.)
    out = dict(definition="sky-only particle weights; same chi-bin weights as likelihood; display |dphi2| < 2 deg", candidates={})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    keys = ("pmra", "pmdec", "parallel", "normal")
    colors = ("C3", "C1", "C0", "C2")
    for j, (name, candidate) in enumerate(candidates.items()):
        m = dict(np.load(ROOT / candidate["spray"]))
        rows = summarize(m, edges, basis)
        out["candidates"][name] = dict(parameters=candidate, bins=rows)
        u = np.array([.5 * (r["lo"] + r["hi"]) for r in rows])
        for ax, key in zip(axes.flat, keys):
            data = np.array([r["observed"][key] for r in rows])
            model = np.array([r["model"][key] for r in rows])
            if j == 0:
                ax.fill_between(u, data[:, 0], data[:, 2], color=".9")
                ax.plot(u, data[:, 1], "ks", ms=4, label="members: median / 16-84%")
            ax.plot(u, model[:, 1], color=colors[j], ls=":" if "initial" in name else "-", label=name)
            ax.set_ylabel(key + " [mas/yr]")
        axes[1, 1].plot(u, [r["pm_direction_offset_deg"] for r in rows], color=colors[j], label=name)
        axes[1, 2].plot(u, [r["rms_distance_only_residual"] for r in rows], color=colors[j], label=name)
    axes[1, 1].set_ylabel("PM direction error [deg]")
    axes[1, 2].set_ylabel("Residual after local distance scaling [mas/yr]")
    for ax in axes.flat:
        ax.set_xlabel("phi1 [deg]")
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Joint bar profiles: PMs selected by sky only; search seed (independent-seed validation separate)")
    fig.tight_layout()
    filename = ROOT / "plots" / f"{args.label}_pm_geometry.png"
    fig.savefig(filename, dpi=130)
    (outdir / "pm_geometry.json").write_text(json.dumps(out, indent=1))
    print(filename)
    for name, result in out["candidates"].items():
        knee = [r for r in result["bins"] if r["lo"] >= 18]
        print(name, "knee PMDEC residuals", [round(r["model"]["pmdec"][1]-r["observed"]["pmdec"][1], 2) for r in knee],
              "direction error", [round(r["pm_direction_offset_deg"], 1) for r in knee])


if __name__ == "__main__":
    main()
