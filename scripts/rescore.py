"""Recompute a demo bundle's HASM from the images it already saved.

    python scripts/rescore.py --bundle flair_lighting/

A sweep's expensive part is the generations, and ``run_demo_sweep`` keeps
every one of them. The scoring on top is cheap, runs on CPU, and can be
redone as often as the metric changes -- no GPU, no regeneration.

This exists because the first head-level sweep scored a destroyed frame
highest: a collapsed generation maximises "how much did this attribute
change" without controlling the attribute at all, and min-max
normalisation then anchored every honest cell to that corrupt ceiling.
``IntegrityGate`` rejects those frames; this rescoring applies it to
already-generated bundles.

Only attributes whose delta metric needs neither a CLIP scorer nor a real
object mask can be rescored offline -- see RESCORABLE. Everything else has
to go back through the harness with ClipSeg and CLIP loaded, because
substituting a full-frame mask would silently produce a different number
rather than the same number computed better.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.demo.report import write_report
from flair_t2i.demo.sweep import DemoPaths
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit
from flair_t2i.metrics.integrity import IntegrityGate
from flair_t2i.metrics.registry import delta_for

#: Attributes whose metric ignores the mask and needs no scorer, so the
#: offline number is identical to the campaign's.
RESCORABLE = (AttributeClass.LIGHTING,)

_UNIT = re.compile(r"_b(\d+)_h(\d+)\.png$")


def _units(paths: DemoPaths, attr: AttributeClass):
    found = {}
    for path in sorted(paths.heads.glob(f"{attr.value}_b*_h*.png")):
        match = _UNIT.search(path.name)
        if match:
            block, head = map(int, match.groups())
            found[HeadUnit(block=block, head=head)] = path
    return found


def rescore(paths: DemoPaths, attr: AttributeClass, gate: IntegrityGate):
    """Return (raw scores by unit, rejected units, rejection reasons)."""
    baseline = Image.open(paths.baseline_image(attr)).convert("RGB")
    metric = delta_for(attr, scorer=None, phrase=None)

    raw: dict[HeadUnit, float] = {}
    rejected: set[HeadUnit] = set()
    reasons: dict[str, str] = {}

    for unit, path in _units(paths, attr).items():
        candidate = Image.open(path).convert("RGB")
        verdict = gate.check(baseline, candidate)
        if not verdict.ok:
            rejected.add(unit)
            raw[unit] = 0.0
            reasons[f"b{unit.block}_h{unit.head}"] = verdict.reason
            continue
        raw[unit] = float(metric(baseline, candidate, None))

    return raw, rejected, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--min-colour-ratio", type=float, default=0.75)
    parser.add_argument("--max-structural-change", type=float, default=0.60)
    parser.add_argument(
        "--title", default="FLAIR — head-level routing", help="report heading"
    )
    args = parser.parse_args()

    paths = DemoPaths(args.bundle)
    gate = IntegrityGate(
        min_colour_ratio=args.min_colour_ratio,
        max_structural_change=args.max_structural_change,
    )

    attrs = [a for a in RESCORABLE if paths.baseline_image(a).exists()]
    if not attrs:
        parser.error(
            f"no rescorable attribute found in {args.bundle}. "
            f"Offline rescoring supports {[a.value for a in RESCORABLE]}; "
            "other attributes need ClipSeg/CLIP via scripts/calibrate.py."
        )

    all_raw, all_rejected, all_reasons = {}, set(), {}
    for attr in attrs:
        raw, rejected, reasons = rescore(paths, attr, gate)
        all_raw[attr] = raw
        all_rejected |= rejected
        all_reasons[attr.value] = reasons
        kept = len(raw) - len(rejected)
        print(f"{attr.value}: kept {kept}/{len(raw)}, rejected {len(rejected)}")

    blocks = sorted({u.block for r in all_raw.values() for u in r})
    heads = sorted({u.head for r in all_raw.values() for u in r})
    tensor = np.zeros((len(blocks), len(heads), len(attrs)))
    for plane, attr in enumerate(attrs):
        for i, b in enumerate(blocks):
            for j, h in enumerate(heads):
                tensor[i, j, plane] = all_raw[attr].get(HeadUnit(b, h), 0.0)
        column = tensor[:, :, plane]
        low, high = column.min(), column.max()
        tensor[:, :, plane] = (
            (column - low) / (high - low) if high - low > 1e-12 else 0.0
        )

    hasm = HASM(tensor, tuple(blocks), tuple(heads), tuple(attrs))
    hasm.save(paths.root / "hasm.npz")
    (paths.root / "rejected.json").write_text(
        json.dumps(all_reasons, indent=2), encoding="utf-8"
    )
    report = write_report(hasm, paths, title=args.title, rejected=all_rejected)

    print(f"\nwrote {paths.root / 'hasm.npz'}")
    print(f"wrote {paths.root / 'rejected.json'}")
    print(f"wrote {report}\n")
    for attr in hasm.attributes:
        top = hasm.top_k(attr, 5)
        print(f"{attr.value} — top 5 after gating:")
        for unit, score in top:
            print(f"    B{unit.block:<3} H{unit.head:<3} {score:.3f}")


if __name__ == "__main__":
    main()
