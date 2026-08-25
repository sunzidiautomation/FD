"""Persist every generation with the provenance needed to reproduce it.

An image on its own is not a result. Six months from now, writing the
paper, the questions are: which BASM produced this, what was alpha_0, which
blocks did each attribute route to, which commit was the code on. This
module writes that alongside every image.

Layout::

    outputs/
      manifest.jsonl          one line per run, append-only
      <run_id>.png            the image
      <run_id>.json           the full RunRecord

``manifest.jsonl`` is append-only on purpose: a crashed or interrupted
Kaggle session leaves every completed run intact and readable.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.jsonl"

_VERSION_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "spacy",
    "scikit-fuzzy",
    "numpy",
)


def git_commit() -> str:
    """Short hash of the running code, or 'unknown' outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown" if result.returncode == 0 else "unknown"


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that can change an image."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {"python": platform.python_version()}
    for name in _VERSION_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return versions


@dataclass
class RunRecord:
    """Everything needed to reproduce one generation."""

    run_id: str
    prompt: str
    seed: int
    steps: int
    guidance_scale: float
    routing: bool
    fuzzy: bool
    basm_source: str
    config: dict[str, Any]
    routed: list[dict[str, Any]] = field(default_factory=list)
    guard_events: list[dict[str, Any]] = field(default_factory=list)
    alpha_scale: float = 1.0
    tag: str = ""
    notes: str = ""
    timestamp: str = ""
    git_commit: str = ""
    versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.git_commit:
            self.git_commit = git_commit()
        if not self.versions:
            self.versions = package_versions()


def describe_plan(plan, guard=None) -> dict[str, Any]:
    """Flatten a RoutingPlan (and optional guard) into JSON-safe fields."""
    if plan is None:
        return {"routed": [], "guard_events": [], "alpha_scale": 1.0}

    routed = [
        {
            "id": rc.component.id,
            "attribute": rc.component.attr.value,
            "text": rc.component.text,
            "hedge": rc.component.hedge,
            "blocks": [[int(b), float(s)] for b, s in rc.blocks],
            "intensity": float(rc.intensity),
        }
        for rc in plan.routed
    ]
    events = (
        [
            {"step": e.step, "reason": e.reason, "value": float(e.value)}
            for e in guard.events
        ]
        if guard is not None
        else []
    )
    return {
        "routed": routed,
        "guard_events": events,
        "alpha_scale": float(plan.alpha_scale),
    }


def save_run(out_dir: str | Path, record: RunRecord, image=None) -> Path:
    """Write the image and its record, and append to the manifest.

    Returns the path of the JSON record.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if image is not None:
        image.save(out / f"{record.run_id}.png")

    payload = asdict(record)
    payload["image"] = f"{record.run_id}.png" if image is not None else None

    json_path = out / f"{record.run_id}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(out / MANIFEST_NAME, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    return json_path


def load_manifest(out_dir: str | Path) -> list[dict[str, Any]]:
    """Read every run recorded in ``out_dir``. Skips corrupt lines."""
    path = Path(out_dir) / MANIFEST_NAME
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written line from an interrupted session
    return records


def summarise(out_dir: str | Path) -> str:
    """One line per run, for a quick look at what a session produced."""
    records = load_manifest(out_dir)
    if not records:
        return f"no runs recorded in {out_dir}"

    lines = [f"{len(records)} run(s) in {out_dir}", ""]
    for r in records:
        blocks = sorted({b for c in r.get("routed", []) for b, _ in c["blocks"]})
        lines.append(
            f"  {r['run_id']:<26} seed={r['seed']:<3} "
            f"routing={str(r['routing']):<5} blocks={blocks} "
            f"guard={len(r.get('guard_events', []))}"
        )
    return "\n".join(lines)
