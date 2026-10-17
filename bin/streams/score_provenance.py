#!/usr/bin/env python3
"""Index of stream-likelihood results: which score version and code commit produced each saved file (2026-10-17).

Every stream result file (results/streams/*, stream-related results/plot_data/*) is assigned
  - the score version in force when it was last written (boundaries = commit times in VERSIONS; definitions in
    docs/kb/models/stream_likelihood_versions.md), and
  - the last commit before that time (check it out to replay the code).
Assignment is by modification time, so a file rewritten later by a re-scoring script gets the later version; files whose own
JSON carries a "settings" dict (grid4_common.score_settings, from 2026-10-17) are tagged from that instead.
Output: docs/kb/models/stream_results_index.md (table) and results/plot_data/stream_results_index.json.

Usage: python bin/streams/score_provenance.py
"""
import fnmatch
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"bin/streams"))
from score_registry import ENTRIES  # noqa: E402
# (tag, commit, short description); a version holds from its commit time until the next one
VERSIONS = [("S0", "89b1473", "per-star sky-conditional score (score)"),
            ("S1", "7183e33", "chi-binned KDE with latent chi (score_chi_kde; bandwidths b5598ca)"),
            ("S2", "29d04f8", "conditional chi-KDE (score_conditional): equal-weight chi bins, implied background widths"),
            ("S3", "ff312ad", "S2 with the uniform background normalised over an explicit box"),
            ("S4", "SCORE_POOLED_COMMIT", "S3 with pooled=True as the default (one KDE of all debris)")]
STREAM = re.compile(r"spray|grid|stream|paper_|track|objective|particle_conv|score_|neff|bar_profile|escape|spiral|rescore|"
                    r"seed_test|bootstrap|agecut|sun_dist|frame_|^d[123]_|m5_|mass_sweep|cluster_mass|spin|rotB|phase2|"
                    r"profile_stage1|fit_bar|fit_lmc|lmc_spray|kernel_check|noise_test|scan_|unwrapping|ocen_state")
DATA_PRODUCT = re.compile(r"stream54_|stream_coordinate|stream5455_gaia_errors|stream_xp_metallicity|unwrapping_")
OUT_MD = ROOT/"docs/kb/models/stream_results_index.md"
OUT_JSON = ROOT/"results/plot_data/stream_results_index.json"


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def commit_time(c):
    return datetime.fromisoformat(git("show", "-s", "--format=%aI", c))


def boundaries():
    out = []
    for tag, c, desc in VERSIONS:
        if c == "SCORE_POOLED_COMMIT":      # the commit that made pooled the default
            c = git("log", "-1", "--format=%h", "-S", 'kw.setdefault("pooled", True)', "--", "bin/streams/grid4_common.py")
            if not c:
                continue                    # not committed yet
        out.append((tag, c, desc, commit_time(c)))
    return out


def files():
    fs = [p for p in (ROOT/"results/streams").iterdir() if not p.name.startswith(".")]
    fs += [p for p in (ROOT/"results/plot_data").iterdir() if STREAM.search(p.name) and p.name != OUT_JSON.name]
    return sorted(fs)


def newest_mtime(p):
    if p.is_file():
        return p.stat().st_mtime
    ts = [q.stat().st_mtime for q in p.rglob("*") if q.is_file()]
    return max(ts) if ts else p.stat().st_mtime


def own_settings(p):
    if p.suffix != ".json" or p.stat().st_size > 50_000_000:
        return None
    try:
        r = json.loads(p.read_text())
    except Exception:
        return None
    found = set()

    def walk(x):
        if isinstance(x, dict):
            if "settings" in x and isinstance(x["settings"], dict) and "version" in x["settings"]:
                found.add(x["settings"]["version"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x[:2000]:
                walk(v)
    walk(r)
    return ",".join(sorted(found)) or None


def registry_entry(rel):
    for e in ENTRIES:
        for pat in e["outputs"]:
            if (pat.endswith("/") and (rel+"/").startswith(pat)) or fnmatch.fnmatch(rel, pat):
                return e
    return None


def main():
    B = boundaries()
    rows = []
    for p in files():
        t = datetime.fromtimestamp(newest_mtime(p), tz=timezone.utc)
        tag = "pre-S0"
        for tg, c, desc, tc in B:
            if t >= tc:
                tag = tg
        rel = str(p.relative_to(ROOT))
        e = registry_entry(rel)
        own = own_settings(p)
        if own:
            version, by = own, "settings"
        elif DATA_PRODUCT.search(p.name):
            version, by = "n/a (data product)", "name"
        elif e is not None:
            version = e["score"].replace("wrapper", f"{tag} (wrapper default at write time)") if "wrapper" in e["score"] else e["score"]
            by = "registry"
        else:
            version, by = f"{tag}?", "mtime only"
        code = git("rev-list", "-1", f"--before={t.isoformat()}", "HEAD")[:7]
        rows.append(dict(path=rel+("/" if p.is_dir() else ""), written_utc=t.strftime("%Y-%m-%d %H:%M"), version=version, tagged_by=by,
                         code_commit=code, script=e["script"] if e else "", options=e["options"] if e else "",
                         spray=e["spray"] if e else "", recorded=e["recorded"] if e else ""))
    OUT_JSON.write_text(json.dumps(dict(versions=[dict(tag=t, commit=c, description=d, from_utc=tc.isoformat()) for t, c, d, tc in B],
                                        files=rows), indent=1))
    esc = lambda x: str(x).replace("|", "\\|")
    lines = ["# Stream results index", "",
             "Generated by `bin/streams/score_provenance.py` from the code audit in `bin/streams/score_registry.py`; do not edit by "
             "hand. Version definitions: [stream likelihood versions](stream_likelihood_versions.md).", "",
             "Columns: **path** (repository-relative); **written** (UTC, newest file in a directory); **score** (version and "
             "function; 'wrapper default at write time' = grid4_common.score with the version in force then); **by** (`registry` = "
             "script audit, `settings` = the file's own record, `mtime only` = no registry entry, version from the date only); "
             "**commit** (last commit before the write); **script**, **options** (non-default score options), **spray** (particle "
             "settings), **file records** (what the output itself stores).", "",
             "| path | written | score | by | commit | script | options | spray | file records |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| `{r['path']}` | {r['written_utc']} | {esc(r['version'])} | {r['tagged_by']} | `{r['code_commit']}` | {esc(r['script'])} | "
              f"{esc(r['options'])} | {esc(r['spray'])} | {esc(r['recorded'])} |" for r in rows]
    OUT_MD.write_text("\n".join(lines)+"\n")
    from collections import Counter
    print(Counter(r["tagged_by"] for r in rows), len(rows), "files ->", OUT_MD.relative_to(ROOT))
    for r in rows:
        if r["tagged_by"] == "mtime only":
            print("  no registry entry:", r["path"])


if __name__ == "__main__":
    main()
