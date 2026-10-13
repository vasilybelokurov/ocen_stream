"""Chen et al. (2024) escape distribution (arXiv:2408.01496, Eqs. 2-4).

Positions and velocities are relative to the cluster in its instantaneous
nonrotating orbital basis: x away from the Galactic centre, z along orbital
angular momentum. Both arms are point inversions. The escape speed supplied
by the caller must use the same satellite potential used in integration.
The empirical recipe was tested on less massive clusters and less central
orbits than Omega Cen; this is an explicit alternative, not a calibration
for Omega Cen in a bar.
"""
from __future__ import annotations

import numpy as np

MEAN = np.array([1.6, -30., 0., 20., 0.])
SIGMA = np.array([.35, 23., 12., 20., 22.])
R_ALPHA_CORRELATION = -.7


def chen_parameters(rng, number):
    """Draw (r/rJ, phi, theta, alpha, beta); angular values are degrees."""
    z = rng.normal(size=(number, 5))
    z[:, 3] = R_ALPHA_CORRELATION * z[:, 0] + np.sqrt(1 - R_ALPHA_CORRELATION**2) * z[:, 3]
    values = MEAN + SIGMA * z
    bad = values[:, 0] <= 0
    while bad.any():
        values[bad] = chen_parameters(rng, int(bad.sum()))
        bad = values[:, 0] <= 0
    return values


def chen_release_ic(orbit, rj, rotation, rng, escape_speed):
    """Interleaved trailing/leading release states in model coordinates."""
    number = 2 * len(rj)
    parameters = chen_parameters(rng, number)
    radius = parameters[:, 0] * np.repeat(rj, 2)
    phi, theta, alpha, beta = np.radians(parameters[:, 1:]).T
    arm = np.tile([1., -1.], len(rj))
    pos = arm[:, None] * radius[:, None] * np.column_stack((np.cos(theta) * np.cos(phi),
                                                          np.cos(theta) * np.sin(phi), np.sin(theta)))
    speed = np.asarray(escape_speed(radius))
    if np.any(~np.isfinite(speed)) or np.any(speed <= 0):
        raise ValueError("Invalid satellite escape speed")
    vel = arm[:, None] * speed[:, None] * np.column_stack((np.cos(beta) * np.cos(alpha),
                                                         np.cos(beta) * np.sin(alpha), np.sin(beta)))
    basis = np.repeat(rotation, 2, axis=0)
    ic = np.repeat(orbit, 2, axis=0).copy()
    ic[:, :3] += np.einsum("ni,nij->nj", pos, basis)
    ic[:, 3:] += np.einsum("ni,nij->nj", vel, basis)
    return ic, arm.astype(np.int8)
