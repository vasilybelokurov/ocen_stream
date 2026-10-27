"""runsync against a local bare repository: clone, push a run directory, a fresh clone sees it, pull on an existing clone."""
import subprocess
from ocen_stream import runsync


def test_clone_push_resume(tmp_path, monkeypatch):
    bare = tmp_path/"runs.git"; subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    monkeypatch.setenv("OCEN_RUNS_URL", str(bare))
    a = tmp_path/"a"; runsync.clone(a); (a/"run1"/"log").mkdir(parents=True); (a/"run1"/"log"/"000000.npz").write_bytes(b"x")
    assert runsync.push(a) and not runsync.push(a)                                     # second push: nothing new
    b = tmp_path/"b"; runsync.clone(b); assert (b/"run1"/"log"/"000000.npz").read_bytes() == b"x"
    (a/"run1"/"log"/"000001.npz").write_bytes(b"y"); assert runsync.push(a)
    runsync.clone(b); assert (b/"run1"/"log"/"000001.npz").exists()                  # existing clone is updated
