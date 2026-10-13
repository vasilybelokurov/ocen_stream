"""Hunter spiral perturbation in the streams' reflected Cartesian frame.

The supplied AGAMA model uses Sun at negative x and clockwise patterns.
Our streams use x_model=-x_source, so the spiral shape must be reflected,
not merely assigned a positive pattern speed. For the even Fourier modes,
reflection reverses every sine coefficient and preserves cosine coefficients.
This transformation is applied to Phi, dPhi/dR and dPhi/dz in the exported
CylSpline table. Original and reflected files are preserved with checksums.

The reference present-day phase is +25 degrees and pattern speed +22.5
km/s/kpc: the mirror of the reference AGAMA Hunter setup. Amplitude 1 is
the supplied perturbation (zero azimuthal monopole), not an extra disk.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np


@functools.lru_cache(maxsize=8)
def reflected_file(source):
    source = Path(source)
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    root = Path(__file__).resolve().parents[3]
    directory = root / "results/streams/spiral_potential"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"hunter_spiral_x_reflected_{digest[:16]}.ini"
    lines, mode, count = [], None, 0
    for line in data.decode().splitlines():
        if "#m" in line:
            mode = int(line.split()[0])
            lines.append(line)
        elif line.startswith("#"):
            lines.append(line)
        elif mode is not None and line.strip():
            # Negative mode indices are sine terms in AGAMA's real basis.
            factor = (-1) ** (abs(mode) + (1 if mode < 0 else 0))
            if factor == 1:
                lines.append(line)
            else:
                values = line.split()
                radius = values[0]
                lines.append(radius + "\t" + "\t".join(format(-float(v), ".17g") for v in values[1:]))
                count += 1
        else:
            lines.append(line)
    if count == 0:
        raise ValueError("No sine coefficients found in the spiral table")
    transformed = ("\n".join(lines) + "\n").encode()
    if not target.exists():
        temp = target.with_suffix(".ini.tmp")
        temp.write_bytes(transformed)
        os.replace(temp, target)
    if target.read_bytes() != transformed:
        raise ValueError("Stored reflected spiral disagrees with the source transformation")
    info = dict(source=str(source), source_sha256=digest, reflected=str(target),
                reflected_sha256=hashlib.sha256(transformed).hexdigest(),
                transformation="Phi_model(x,y,z) = Phi_source(-x,y,z); real Fourier parity", changed_rows=count)
    (directory / f"hunter_spiral_x_reflected_{digest[:16]}.json").write_text(json.dumps(info, indent=1))
    return str(target)


def with_spiral(host, t_today, amplitude=1., omega=22.5, phase_deg=25., source=None):
    """Add a rigidly rotating spiral to the host (kpc, Msun, km/s).

    Pattern phase and speed are independent of the bar. t_today is the model
    time of the present, in kpc/(km/s). amplitude 0 returns host unchanged.
    This exploratory extension does not model transient or evolving arms.
    """
    if amplitude == 0:
        return host
    if not np.isfinite([amplitude, omega, phase_deg, t_today]).all() or amplitude < 0:
        raise ValueError("Invalid spiral parameters")
    from .restricted import HUNTER24_DIR, agama_kpc
    ag = agama_kpc()
    source = source or str(Path(HUNTER24_DIR) / "MWPotentialHunter24_spiral.ini")
    potential = ag.Potential(file=reflected_file(str(source)))
    times = np.array([-1., t_today + 1.])
    angles = np.radians(phase_deg) + omega * (times - t_today)
    spiral = ag.Potential(potential=potential, scale=[amplitude, 1.], rotation=np.column_stack((times, angles)))
    return ag.Potential(host, spiral)
