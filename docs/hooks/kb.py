"""MkDocs hook for the oCen_stream knowledge base (adapted from oCen_dm, 2026-10-25).

* Serves files that live outside ``docs/``: committed ``plots/*.png`` at
  ``plots/``. The project journal
  is kept locally and is not published; links to it become plain text.
* Generates ``kb/literature.md`` from ``docs/kb/literature.json`` (the stream subset of oCen_dm's list).
* Rewrites links in every page against the file's real location in the repo:
  targets inside ``docs/`` or ``plots/`` stay site links; other tracked repo
  files (code, configs, tests) point to GitHub; untracked targets
  (``results/``, ``data/``) become plain text, since collaborators do not have
  them. Absolute local links into the repository (Codex reviews) are mapped
  the same way.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/"docs"
GITHUB = "https://github.com/vasilybelokurov/ocen_stream/blob/main/"
SCIX_ID = "RSXq3-NtTTCApvfPUbdwrg"
LINK = re.compile(r"(!?)\[((?:[^\[\]]|\[[^\]]*\])*)\]\((<[^>]+>|[^)\s]+)(\s+\"[^\"]*\")?\)")
SPACED = re.compile(r"(!?)\[([^\[\]]*)\]\(([^()<>\"]*\s[^()<>\"]*)\)")
_REAL = {}          # virtual src_uri -> real path of a generated page (for link resolution)
_TRACKED = None


def _tracked():
    global _TRACKED
    if _TRACKED is None:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        _TRACKED = set(out.split("\n"))
    return _TRACKED


# ---------------------------------------------------------------- generated pages


def _literature_page():
    d = json.loads((DOCS/"kb/literature.json").read_text())
    papers = d["papers"]
    topics = ["data-source", "omega-cen-mass-imbh", "cluster-dynamics-methods", "tides-and-dm-survival",
              "tails-and-streams", "mw-potential-bar-lmc", "stream-fitting-methods", "stellar-populations-mf", "other"]
    names = {"data-source": "Data sources", "omega-cen-mass-imbh": "Omega Cen mass, remnants and central mass",
             "cluster-dynamics-methods": "Cluster dynamics methods (DF, Jeans, multimass, N-body)",
             "tides-and-dm-survival": "Tides and dark-matter survival", "tails-and-streams": "Tails and streams",
             "mw-potential-bar-lmc": "Milky Way potential, bar and LMC", "stream-fitting-methods": "Stream-fitting methods",
             "stellar-populations-mf": "Stellar populations and mass function", "other": "Other"}
    n_lib = sum(1 for p in papers if p.get("bibcode") and p["relevance"] != "tangential")
    out = ["# Literature", "",
           f"!!! abstract \"SciX library\"\n    All {n_lib} papers below with a bibcode and relevance *core* or *context* are in the public library "
           f"[omega Cen dark matter (oCen_dm)], shared with the oCen_dm project,(https://scixplorer.org/public-libraries/{SCIX_ID}) "
           f"([ADS view](https://ui.adsabs.harvard.edu/public-libraries/{SCIX_ID})).", "",
           "Generated from `docs/kb/literature.json`, compiled from every citation in the repository "
           f"({d.get('generated', '')}); identifiers were checked in ADS. *Core*: the project uses or depends on it; "
           "*context*: cited for background; *tangential*: mentioned once (not in the library).", ""]
    for t in topics:
        sel = sorted((p for p in papers if p["topic"] == t), key=lambda p: ({"core": 0, "context": 1}.get(p["relevance"], 2), p["cite"]))
        if not sel:
            continue
        out += [f"## {names[t]}", "", "| Paper | Title | Relevance | Links |", "|---|---|---|---|"]
        for p in sel:
            links = []
            if p.get("bibcode"):
                links.append(f"[ADS](https://ui.adsabs.harvard.edu/abs/{p['bibcode'].replace('&', '%26')})")
            if p.get("arxiv"):
                links.append(f"[arXiv](https://arxiv.org/abs/{p['arxiv']})")
            if p.get("doi"):
                links.append(f"[DOI](https://doi.org/{p['doi']})")
            if not links:
                links.append("unresolved")
            title = (p.get("title") or "").replace("|", "/")
            out.append(f"| {p['cite']} | {title} | {p['relevance']} | {' '.join(links)} |")
        out.append("")
    bad = d.get("erroneous_identifiers_exclude") or []
    if bad:
        out += ["## Identifiers found to be wrong", "",
                "These identifiers appear in repository text (mostly Codex research reports) but point to other papers; "
                "the correct identifiers are used above.", "", "| As written | Note |", "|---|---|"]
        for b in bad:
            ident = b.get("identifier") or b.get("arxiv") or b.get("doi") or b.get("bibcode") or ""
            out.append(f"| `{ident}` | {' '.join(str(b.get('note') or b.get('reason') or '').split())} |")
    return "\n".join(out)+"\n"


def on_files(files, config):
    for rel in sorted(t for t in _tracked() if t.startswith("plots/") and t.endswith(".png")):   # committed plots only
        files.append(File(rel, str(ROOT), config["site_dir"], config["use_directory_urls"]))
    _REAL["kb/literature.md"] = DOCS/"kb/literature.md"
    files.append(File.generated(config, "kb/literature.md", content=_literature_page()))
    return files


# ---------------------------------------------------------------- link rewriting
def _rewrite(page_uri, target):
    """Return a new link target, or None to turn the link into plain text."""
    raw = target[1:-1] if target.startswith("<") else target
    if re.match(r"^[a-z][a-z0-9+.-]*:", raw, re.I) or raw.startswith("#"):
        return target
    path, frag = (raw.split("#", 1)+[""])[:2]
    line = re.search(r":(\d+)(?:-\d+)?$", path)
    if line and not Path(path).suffix.startswith(f".{line.group(1)}"):
        path, frag = path[:line.start()], frag or f"L{line.group(1)}"
    real_page = _REAL.get(page_uri, DOCS/page_uri)
    if path.startswith("/"):
        p = Path(path)
        try:
            real = ROOT/p.relative_to(ROOT)
        except ValueError:
            return None
    elif not path:
        return target
    else:
        real = Path(os.path.normpath(real_page.parent/path))
        if not real.exists() and (ROOT/path).exists():      # repo-root-relative links in reviews
            real = ROOT/path
    try:
        rel_repo = real.relative_to(ROOT).as_posix()
    except ValueError:
        return None
    if rel_repo.startswith("docs/") and real.exists() and not rel_repo.startswith("docs/hooks/"):
        virtual = rel_repo[len("docs/"):]
    elif rel_repo.startswith("plots/") and rel_repo in _tracked():
        virtual = rel_repo
    elif rel_repo in _tracked() or (real.is_dir() and any(t.startswith(rel_repo+"/") for t in _tracked())):
        kind = "tree/main/" if real.is_dir() else ""
        url = (GITHUB.replace("blob/main/", kind) if kind else GITHUB)+rel_repo
        return url+(f"#{frag}" if frag else "")
    else:
        return None
    new = posixpath.relpath(virtual, posixpath.dirname(page_uri) or ".")
    if re.fullmatch(r"L\d+(-L?\d+)?", frag):      # source-line anchors only exist on GitHub
        frag = ""
    return new+(f"#{frag}" if frag else "")


def on_page_markdown(markdown, page, config, files):
    uri = page.file.src_uri

    def sub(m):
        bang, text, target, title = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        new = _rewrite(uri, target)
        if new is None:
            shown = target.strip("<>")
            return f"*[figure not in the repository: `{shown}`]*" if bang else (text if text.strip() else f"`{shown}`")
        return f"{bang}[{text}]({new}{title})"

    def spaced(m):          # e.g. "[packet:2021](question packet)": not a path, keep the text
        return m.group(2)

    # leave fenced code blocks untouched
    parts = re.split(r"(?ms)(^```.*?^```)", markdown)
    return "".join(p if p.startswith("```") else LINK.sub(sub, SPACED.sub(spaced, p)) for p in parts)
