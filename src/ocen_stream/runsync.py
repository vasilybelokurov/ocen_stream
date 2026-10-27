"""Keep a run directory in the private runs repository (2026-10-27; replaces Google Drive for Colab logs: no consent pop-ups).

The runs repository (default vasilybelokurov/ocen_stream_runs, private) holds only run logs and sampler checkpoints. The token is a
fine-grained GitHub token for that repository alone (Contents read/write), stored in Colab's Secrets panel as GH_RUNS_TOKEN and passed
in the environment; it is never printed. A restart clones the repository again, so a job resumes from its last pushed log.
Usage (Colab):  python -m ocen_stream.runsync clone DEST          (clone or update the runs repository into DEST)
                python -m ocen_stream.runsync loop DEST [SECONDS] (commit and push every SECONDS, default 300; run in the background)
                python -m ocen_stream.runsync push DEST           (commit and push once, e.g. at the end of a job)
"""
import os, subprocess, sys, time
from pathlib import Path

REPO = os.environ.get("OCEN_RUNS_REPO", "vasilybelokurov/ocen_stream_runs")


def _git(dest, *args, check=True):
    return subprocess.run(["git", "-C", str(dest), *args], check=check, capture_output=True, text=True)


def clone(dest):
    dest = Path(dest); url = os.environ.get("OCEN_RUNS_URL")                      # OCEN_RUNS_URL: any git URL (tests)
    if not (dest/".git").exists():
        url = url or f"https://x-access-token:{os.environ['GH_RUNS_TOKEN']}@github.com/{REPO}.git"
        if subprocess.run(["git", "clone", "-q", url, str(dest)], capture_output=True).returncode:
            raise RuntimeError(f"runsync: could not clone {REPO} (token missing, expired or without access?)")
        _git(dest, "config", "user.name", "colab-runner"); _git(dest, "config", "user.email", "colab-runner@users.noreply.github.com")
    elif _git(dest, "rev-parse", "--verify", "HEAD", check=False).returncode == 0:
        _git(dest, "pull", "-q", "--rebase", "origin", "main", check=False)
    print(f"runs repository at {dest}", flush=True)


def push(dest, msg=None):
    """commit everything under dest and push; returns True when something was pushed (failures are reported, not raised)."""
    _git(dest, "add", "-A")
    if _git(dest, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    _git(dest, "commit", "-q", "-m", msg or time.strftime("sync %Y-%m-%d %H:%M:%S"))
    r = _git(dest, "push", "-q", "origin", "HEAD:main", check=False)
    if r.returncode:
        print(f"runsync: push failed ({r.returncode}); will retry", file=sys.stderr, flush=True); return False
    return True


def loop(dest, every=300):
    while True:
        time.sleep(every); push(dest)


if __name__ == "__main__":
    cmd, dest = sys.argv[1], sys.argv[2]
    {"clone": lambda: clone(dest), "push": lambda: print("pushed" if push(dest, "final sync") else "nothing to push"),
     "loop": lambda: loop(dest, float(sys.argv[3]) if len(sys.argv) > 3 else 300.)}[cmd]()
