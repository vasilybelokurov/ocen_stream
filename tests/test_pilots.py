"""pilots.py on the analytic fake likelihood: Nelder-Mead converges, both pilots and their diagnostics run, a restart repeats no
evaluation and retraces the SMC bit for bit, a log with other provenance is refused."""
import json
import numpy as np
import pytest
from ocen_stream import pilots as PL


def run_nm(f, x0, budget=4000):
    ch = PL.NM(np.asarray(x0, float))
    while ch.n < budget and ch.size() > 1e-9:
        a = ch.ask(); ch.tell([f(u) for u in a])
    return ch.best()


def test_nm_quadratic_and_rosenbrock():
    c = np.linspace(0.2, 0.8, PL.D)
    x, fx = run_nm(lambda u: float(np.sum((u-c)**2*np.arange(1, PL.D+1))), np.full(PL.D, 0.5))
    assert np.max(np.abs(x-c)) < 1e-4 and fx < 1e-8
    r = lambda u: float(np.sum(100*(u[1:]-u[:-1]**2)**2+(1-u[:-1])**2))        # minimum at u = 1 (box edge)
    x, fx = run_nm(r, np.full(PL.D, 0.3), budget=20000)
    assert fx < 1e-3


def test_pilots_fake_restart_and_provenance(tmp_path, monkeypatch):
    out = tmp_path/"p"
    for job in ("jitter", "smc", "scan"):
        monkeypatch.setattr("sys.argv", ["pilots", "--fake", "--out", str(out), "--run", job]); PL.main()
    smc = np.load(out/"smc_final.npz"); d = np.load(out/"scan_diag.npz")
    assert float(smc["beta"]) > 0 and np.min(d["cmin"]) < 170                     # the 166 basin found by the scan
    assert np.any(np.abs(d["cmin"]-351) < 30)                                      # and the 351 basin
    (out/"smc_state.npz").unlink(); (out/"smc_final.npz").unlink()
    ev = PL.Evaluator(None, out, fake=True); x, c, b = PL.smc(ev, out); assert ev.nev == 0
    assert np.array_equal(x, smc["x"])
    (out/"manifest.json").write_text(json.dumps(dict(fake=False)))
    with pytest.raises(RuntimeError):
        PL.Evaluator(None, out, fake=True)
