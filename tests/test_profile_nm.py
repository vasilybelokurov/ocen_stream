"""Ask/tell Nelder-Mead of profile_nm.py converges on smooth 8-D functions inside the bounds and respects [0, 1]."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"bin/streams"))
from profile_nm import NM


def _run(f, x0, budget):
    ch = NM("t", 0.3, x0, 1, 0.5, budget)
    while not ch.done():
        for tag, z in ch.ask():
            ch.tell(tag, np.clip(z, 0, 1), f(np.clip(z, 0, 1)))
    i = int(np.argmin(ch.F)); return ch.S[i], ch.F[i], ch.n_eval


def test_quadratic_interior():
    c = np.linspace(0.3, 0.7, 8); w = np.linspace(1, 8, 8)
    x, fx, n = _run(lambda z: np.sum(w*(z-c)**2)*1000, np.full(8, 0.5), 400)
    assert np.max(np.abs(x-c)) < 0.06 and n <= 400


def test_optimum_on_bound_stays_inside():
    c = np.array([1.2, 0.5, 0.5, -0.1, 0.5, 0.5, 0.5, 0.5])
    x, fx, n = _run(lambda z: np.sum((z-c)**2), np.full(8, 0.5), 400)
    assert np.all((x >= 0) & (x <= 1)) and x[0] > 0.9 and x[3] < 0.1
