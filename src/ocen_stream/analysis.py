"""Like-for-like diagnostics of live and restricted N-body snapshots.

The same centre finder (shrinking-sphere median of stars + remnants) and the same bound criterion
(energy in a spherical multipole potential of the bound particles, iterated) are applied to every
snapshot, whatever code produced it. Snapshot format: npz with t_myr, pos [pc], vel [km/s] in the
Galactocentric frame of bin/nbody/run_nbody.py; masses and species from the run's particles.npz
(restricted) or the ICs named in run.json (live).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from .restricted import FrozenCore, bound_set

SPECIES = ("stars", "remnants", "halo")


def run_particles(run):
    run = Path(run)
    if (run/"particles.npz").exists():
        d = np.load(run/"particles.npz")
    else:
        ics = json.loads((run/"run.json").read_text())["ics"]
        root = run
        while not (root/ics).exists() and root != root.parent:
            root = root.parent
        d = np.load(root/ics)
    return d["mass"].astype(float), d["species"].astype(np.int8)


def run_core(run):
    """The run's FrozenCore (restricted runs with --rfreeze), else None."""
    path = Path(run)/"frozen_core.npz"
    return FrozenCore.load(path) if path.exists() else None


def frozen_star_count(run, species_mass_star=None):
    """Number of star particles in the frozen core (equal-mass particles), 0 if none."""
    core = run_core(run)
    if core is None or 0 not in core.mass_by_species:
        return 0
    mass, species = run_particles(run)
    m_star = float(np.median(mass[species == 0])) if np.any(species == 0) else 1.
    return int(round(core.mass_by_species[0]/m_star))


def snapshot_times(run):
    files = sorted(glob.glob(str(Path(run)/"snap_[0-9]*.npz")))
    return files, np.array([float(np.load(f)["t_myr"]) for f in files])


def load(run, t_myr=None, name=None):
    """Snapshot nearest to t_myr (or by file name) as (t, xv [kpc, km/s])."""
    if name is not None:
        s = np.load(Path(run)/name)
    else:
        files, times = snapshot_times(run)
        s = np.load(files[int(np.argmin(np.abs(times-t_myr)))])
    xv = np.hstack((s["pos"].astype(float)/1e3, s["vel"].astype(float)))
    return float(s["t_myr"]), xv


def find_centre(xv, sel, guess=None, radii_pc=(200., 50., 20., 10., 5.)):
    """Shrinking-sphere median position and the median velocity inside the last sphere."""
    c = np.median(xv[sel, :3], axis=0) if guess is None else np.asarray(guess[:3], float)
    inside = sel
    for R in radii_pc:
        inside = sel & (np.sum((xv[:, :3]-c)**2, axis=1) < (R/1e3)**2)
        if inside.sum() < 50:
            break
        c = np.median(xv[inside, :3], axis=0)
    v = np.median(xv[inside, 3:], axis=0)
    return np.concatenate((c, v))


def state(xv, mass, species, guess=None, core=None):
    """Centre, bound mask and bound mass per species (frozen-core mass included if a FrozenCore is given)."""
    lum = species <= 1
    centre = find_centre(xv, lum, guess)
    near = np.sum((xv[:, :3]-centre[:3])**2, axis=1) < 0.5**2     # start the iteration from particles within 500 pc
    bound, _ = bound_set(xv, centre, mass, start=near, core=core)
    frozen = core.mass_by_species if core is not None else {}
    bm = {SPECIES[s]: float(mass[bound & (species == s)].sum()) + frozen.get(int(s), 0.)
          for s in sorted(set(np.unique(species).tolist()) | set(frozen))}
    return centre, bound, bm


def host_energy_lz(host, xv):
    E = host.potential(xv[:, :3]) + 0.5*np.sum(xv[:, 3:]**2, axis=1)
    Lz = xv[:, 0]*xv[:, 4] - xv[:, 1]*xv[:, 3]
    return E, Lz


def tail_metrics(host, xv, centre, bound, species, shell_kpc=(0.3, 2.0), n_frozen_stars=0):
    """Stellar debris statistics: counts, spreads of host energy and Lz relative to the cluster,
    and the local 3D structure of unbound stars in a shell around the cluster."""
    stars = species == 0
    unb = stars & ~bound
    Ec, Lzc = host_energy_lz(host, centre[None, :])
    E, Lz = host_energy_lz(host, xv[unb])
    rel = xv[unb] - centre
    d = np.linalg.norm(rel[:, :3], axis=1)
    sh = (d > shell_kpc[0]) & (d < shell_kpc[1])
    # local frame: along the cluster velocity, perpendicular in the orbital plane, normal
    v = centre[3:]/np.linalg.norm(centre[3:])
    n = np.cross(centre[:3], centre[3:]); n /= np.linalg.norm(n)
    w = np.cross(n, v)
    out = dict(n_unbound_stars=int(unb.sum()), unbound_star_fraction=float(unb.sum()/(stars.sum()+n_frozen_stars)),
               dE_rms=float(np.std(E-Ec[0])), dLz_rms=float(np.std(Lz-Lzc[0])),
               dE_median=float(np.median(E-Ec[0])), n_shell=int(sh.sum()))
    if sh.sum() > 20:
        out.update(width_normal_pc=float(np.std(rel[sh, :3] @ n)*1e3),
                   width_inplane_pc=float(np.std(rel[sh, :3] @ w)*1e3),
                   sigma_v_normal=float(np.std(rel[sh, 3:] @ n)),
                   sigma_v_inplane=float(np.std(rel[sh, 3:] @ w)),
                   sigma_v_along=float(np.std(rel[sh, 3:] @ v)),
                   leading_fraction=float(np.mean(rel[sh, :3] @ v > 0)))
    return out
