"""Dataset provenance: registry parsing, checksums, and the file manifest.

The registry lives in ``provenance/datasets.yaml``. Observed checksums of
downloaded files are appended to ``provenance/checksums.txt`` and recorded in
``provenance/manifest.json``; neither is edited by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from .paths import provenance_dir, raw_dir

__all__ = [
    "FileRecord",
    "DatasetRecord",
    "Registry",
    "load_registry",
    "file_digest",
    "verify_file",
    "write_manifest",
    "environment_record",
]

_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class FileRecord:
    """One external file belonging to a dataset.

    Attributes
    ----------
    dataset : str
        Key of the owning dataset in the registry.
    name : str
        Local file name under ``data/raw/<dataset>/``.
    url : str or None
        Retrieval URL. ``None`` for files with no direct link.
    published_md5 : str or None
        Provider-published checksum, or ``None`` when the provider publishes
        none. The algorithm is given by ``published_algorithm``.
    checksum_source : str or None
        Where ``published_md5`` came from. ``None`` iff ``published_md5`` is.
    expected_bytes : int or None
        Approximate expected size, used only as a sanity check.
    expected_bytes_tolerance : float
        Fractional tolerance on ``expected_bytes``.
    required : bool
        Whether the pipeline cannot proceed without this file.
    published_algorithm : str
        Hash algorithm of ``published_md5`` (Zenodo publishes ``md5`` today but
        the API returns ``'<algorithm>:<hexdigest>'``, so it is not assumed).
    """

    dataset: str
    name: str
    url: str | None
    published_md5: str | None = None
    checksum_source: str | None = None
    expected_bytes: int | None = None
    expected_bytes_tolerance: float = 0.1
    required: bool = True
    published_algorithm: str = "md5"

    @property
    def path(self) -> Path:
        """Local destination path of this file."""
        return raw_dir() / self.dataset / self.name

    @property
    def is_glob(self) -> bool:
        """Whether ``name`` is a pattern (manual downloads with unknown format)."""
        return "*" in self.name or "?" in self.name


@dataclass(frozen=True)
class DatasetRecord:
    """One external dataset (a paper plus its data products)."""

    key: str
    title: str
    role: str
    kind: str
    optional: bool
    paper: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    files: tuple[FileRecord, ...] = ()

    @property
    def raw_dir(self) -> Path:
        """Directory holding this dataset's raw files."""
        return raw_dir() / self.key


@dataclass(frozen=True)
class Registry:
    """The parsed dataset registry."""

    path: Path
    schema_version: int
    datasets: dict[str, DatasetRecord]

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.datasets.values())

    def __getitem__(self, key: str) -> DatasetRecord:
        if key not in self.datasets:
            raise KeyError(
                f"unknown dataset {key!r}; registry has {sorted(self.datasets)}"
            )
        return self.datasets[key]

    def of_kind(self, kind: str) -> list[DatasetRecord]:
        """Return datasets whose retrieval kind is ``kind`` (e.g. ``'zenodo'``)."""
        return [d for d in self if d.kind == kind]


def load_registry(path: Path | str | None = None) -> Registry:
    """Load and validate ``provenance/datasets.yaml``.

    Parameters
    ----------
    path : path-like, optional
        Registry file. Defaults to ``<root>/provenance/datasets.yaml``.

    Returns
    -------
    Registry

    Raises
    ------
    ValueError
        If a required key is missing, or a file declares an MD5 without saying
        where that MD5 came from (guards against invented metadata).
    """
    path = Path(path) if path is not None else provenance_dir() / "datasets.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    if not isinstance(doc, dict) or "datasets" not in doc:
        raise ValueError(f"{path}: expected a mapping with a 'datasets' key")

    datasets: dict[str, DatasetRecord] = {}
    for key, entry in doc["datasets"].items():
        for required_key in ("title", "role", "data"):
            if required_key not in entry:
                raise ValueError(f"{path}: dataset {key!r} is missing {required_key!r}")

        files: list[FileRecord] = []
        for spec in entry.get("files") or []:
            md5 = spec.get("published_md5")
            source = spec.get("checksum_source")
            if md5 is not None and not source:
                raise ValueError(
                    f"{path}: {key}/{spec.get('name')} declares published_md5 with no "
                    "checksum_source; every checksum must state its provenance"
                )
            algorithm = str(spec.get("published_algorithm", "md5")).lower()
            if md5 is not None and algorithm not in hashlib.algorithms_available:
                raise ValueError(
                    f"{path}: {key}/{spec.get('name')} declares unknown hash "
                    f"algorithm {algorithm!r}"
                )
            files.append(
                FileRecord(
                    dataset=key,
                    name=spec["name"],
                    url=spec.get("url"),
                    published_md5=md5,
                    published_algorithm=algorithm,
                    checksum_source=source,
                    expected_bytes=spec.get("expected_bytes"),
                    expected_bytes_tolerance=float(
                        spec.get("expected_bytes_tolerance", 0.1)
                    ),
                    required=bool(spec.get("required", True)),
                )
            )

        datasets[key] = DatasetRecord(
            key=key,
            title=entry["title"],
            role=entry["role"],
            kind=entry["data"].get("kind", "none"),
            optional=bool(entry.get("optional", False)),
            paper=entry.get("paper", {}) or {},
            data=entry["data"] or {},
            files=tuple(files),
        )

    return Registry(
        path=path,
        schema_version=int(doc.get("schema_version", 1)),
        datasets=datasets,
    )


def file_digest(path: Path | str, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file, read in 1 MiB chunks.

    Parameters
    ----------
    path : path-like
        File to hash.
    algorithm : str, optional
        Any name accepted by :func:`hashlib.new`. Default ``'sha256'``.

    Returns
    -------
    str
        Lower-case hex digest.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(record: FileRecord, path: Path | None = None) -> dict[str, Any]:
    """Check a downloaded file against its provenance record.

    Parameters
    ----------
    record : FileRecord
        Registry entry describing the expected file.
    path : pathlib.Path, optional
        File to check. Defaults to ``record.path``.

    Returns
    -------
    dict
        Keys include ``exists``, ``bytes``, ``md5``, ``sha256``,
        ``checksum_ok`` (``None`` when the provider publishes none),
        ``size_ok`` (``None`` when no expected size is given),
        ``verification`` (``'checksum'``, ``'size'`` or ``'none'``),
        ``verified`` (a published checksum was matched) and ``ok``.

    Notes
    -----
    A zero-byte file is never ``ok``: an outage or a truncated transfer
    otherwise passes as a successful download when no checksum is published.
    """
    path = Path(path) if path is not None else record.path
    out: dict[str, Any] = {
        "dataset": record.dataset,
        "name": record.name,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": None,
        "md5": None,
        "sha256": None,
        "published_checksum": record.published_md5,
        "published_algorithm": record.published_algorithm,
        "checksum_source": record.checksum_source,
        "checksum_ok": None,
        "size_ok": None,
        "verification": "none",
        "verified": False,
        "ok": False,
    }
    if not out["exists"]:
        return out

    out["bytes"] = path.stat().st_size
    out["md5"] = file_digest(path, "md5")
    out["sha256"] = file_digest(path, "sha256")

    if record.published_md5:
        algorithm = (record.published_algorithm or "md5").lower()
        observed = out["md5"] if algorithm == "md5" else file_digest(path, algorithm)
        out["observed_published_algorithm_digest"] = observed
        out["checksum_ok"] = observed == record.published_md5.lower()
        out["verification"] = "checksum"
    elif record.expected_bytes:
        out["verification"] = "size"

    if record.expected_bytes:
        tol = record.expected_bytes_tolerance
        out["size_ok"] = (
            abs(out["bytes"] - record.expected_bytes) <= tol * record.expected_bytes
        )

    checks = [v for v in (out["checksum_ok"], out["size_ok"]) if v is not None]
    out["verified"] = out["checksum_ok"] is True
    out["ok"] = bool(out["bytes"]) and all(checks)
    return out


def environment_record() -> dict[str, Any]:
    """Return interpreter/platform and key package versions for the manifest."""
    versions: dict[str, str] = {}
    for module in ("numpy", "scipy", "astropy", "pandas", "pyarrow", "yaml", "requests"):
        try:
            mod = __import__(module)
            versions[module] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[module] = "missing"
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def write_manifest(
    results: list[dict[str, Any]],
    path: Path | None = None,
    *,
    merge: bool = True,
) -> Path:
    """Update the download manifest and the plain-text checksum list.

    Parameters
    ----------
    results : list of dict
        Output of :func:`verify_file`, one entry per file touched by this run.
        Failed and unverified attempts are recorded too, not dropped.
    path : pathlib.Path, optional
        Manifest destination. Defaults to ``provenance/manifest.json``.
    merge : bool, optional
        Merge into the existing manifest, keyed by ``(dataset, name)``, so that
        fetching one dataset does not erase the provenance of the others.

    Returns
    -------
    pathlib.Path
        The manifest path written.
    """
    path = Path(path) if path is not None else provenance_dir() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[tuple[str, str], dict[str, Any]] = {}
    if merge and path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        for item in previous.get("files", []):
            entries[(item.get("dataset", ""), item.get("name", ""))] = item

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for item in results:
        item = dict(item)
        item.setdefault("recorded_utc", stamp)
        key = (item.get("dataset", ""), item.get("name", ""))
        previous = entries.get(key)
        # A failed attempt records itself, but must not erase the identity of a
        # copy that was verified earlier and is still on disk: preprocessing
        # compares against that anchor.
        if previous and previous.get("ok") and not item.get("ok"):
            item["last_verified"] = {
                k: previous.get(k)
                for k in ("sha256", "md5", "bytes", "recorded_utc", "path")
            }
            if item.get("sha256") is None:
                item["sha256"] = previous.get("sha256")
                item["bytes"] = previous.get("bytes")
                item["sha256_from"] = "last_verified"
        entries[key] = item

    ordered = [entries[key] for key in sorted(entries)]
    payload = {
        "written_utc": stamp,
        "environment": environment_record(),
        "files": ordered,
    }
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# sha256  bytes  dataset/name   (written by `ocen fetch-data`; do not edit)",
    ]
    for item in ordered:
        if item.get("sha256"):
            lines.append(
                f"{item['sha256']}  {item['bytes']}  {item['dataset']}/{item['name']}"
            )
    _atomic_write(path.parent / "checksums.txt", "\n".join(lines) + "\n")
    return path


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary file in the same directory."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
