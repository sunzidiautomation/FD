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
from flair_t2i.calibration.corpus import DEFAULT_CORPUS_PATH, load_corpus
from flair_t2i.demo.report import write_report
from flair_t2i.demo.sweep import DemoPaths
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit
from flair_t2i.metrics.embedding import crop_to_mask
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


def rescore(
    paths: DemoPaths,
    attr: AttributeClass,
    gate: IntegrityGate,
    scorer=None,
    masker=None,
    phrase: str | None = None,
):
    """Return (raw scores by unit, rejected units, rejection reasons)."""
    baseline = Image.open(paths.baseline_image(attr)).convert("RGB")

    # identity crops to this; the scene-level metrics ignore it. The label
    # has to be the corpus's own -- segmenting for "object" instead of
    # "sedan" would crop to whatever ClipSeg guessed, silently.
    mask = None
    if masker is not None:
        pairs = load_corpus(DEFAULT_CORPUS_PATH).get(attr, [])
        if not pairs:
            raise SystemExit(f"no corpus pairs for {attr.value}")
        mask = masker(baseline, pairs[0].object_label)
        # The swap injected pairs[0].changed, so that is what a head
        # controlling this attribute should have moved the object toward.
        if attr is AttributeClass.IDENTITY and phrase is None:
            phrase = pairs[0].changed

    metric = delta_for(attr, scorer=scorer, phrase=phrase)

    raw: dict[HeadUnit, float] = {}
    rejected: set[HeadUnit] = set()
    reasons: dict[str, str] = {}

    for unit, path in _units(paths, attr).items():
        candidate = Image.open(path).convert("RGB")
        # Gate the same region the metric measures. A frame whose object is
        # destroyed but whose background survives passes a whole-frame gate
        # and then maxes out an object-cropped metric -- which is how a car
        # buried in confetti outranked every genuine identity change.
        verdict = gate.check(
            crop_to_mask(baseline, mask), crop_to_mask(candidate, mask)
        )
        if not verdict.ok:
            rejected.add(unit)
            raw[unit] = 0.0
            reasons[f"b{unit.block}_h{unit.head}"] = verdict.reason
            continue
        raw[unit] = float(metric(baseline, candidate, mask))

    return raw, rejected, reasons


def _repair_from_matrix(paths: DemoPaths, gate: IntegrityGate, args) -> None:
    """Gate the saved images and re-normalise the existing scores.

    Keeps whatever metric produced the bundle -- it only removes cells whose
    generation collapsed and rescales the survivors, which is the part that
    needs no model.
    """
    hasm = HASM.load(paths.root / "hasm.npz")
    reasons: dict[str, dict[str, str]] = {}
    rejected: set[HeadUnit] = set()

    for attr in hasm.attributes:
        baseline_path = paths.baseline_image(attr)
        if not baseline_path.exists():
            raise SystemExit(f"missing baseline for {attr.value}: {baseline_path}")
        baseline = Image.open(baseline_path).convert("RGB")

        per_attr: dict[str, str] = {}
        for unit, path in _units(paths, attr).items():
            verdict = gate.check(baseline, Image.open(path).convert("RGB"))
            if not verdict.ok:
                rejected.add(unit)
                per_attr[f"b{unit.block}_h{unit.head}"] = verdict.reason
        reasons[attr.value] = per_attr
        total = len(_units(paths, attr))
        print(
            f"{attr.value}: kept {total - len(per_attr)}/{total}, "
            f"rejected {len(per_attr)}"
        )

    repaired = hasm.excluding(rejected)
    repaired.save(paths.root / "hasm.npz")
    (paths.root / "rejected.json").write_text(
        json.dumps(reasons, indent=2), encoding="utf-8"
    )
    report = write_report(repaired, paths, title=args.title, rejected=rejected)

    print(f"\nwrote {paths.root / 'hasm.npz'}")
    print(f"wrote {paths.root / 'rejected.json'}")
    print(f"wrote {report}\n")
    for attr in repaired.attributes:
        print(f"{attr.value} — top 5 after gating:")
        for unit, score in repaired.top_k(attr, 5):
            print(f"    B{unit.block:<3} H{unit.head:<3} {score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--from-matrix",
        action="store_true",
        help="reuse the scores in the bundle's hasm.npz instead of recomputing "
        "the metric, dropping collapsed cells and re-normalising over the rest. "
        "Min-max is affine, so this is exact -- and it works for attributes "
        "whose metric needs CLIP or ClipSeg, which are not installed locally.",
    )
    parser.add_argument(
        "--with-clip",
        action="store_true",
        help="load ClipSeg and CLIP so attributes needing them (identity, "
        "style, action) can be recomputed from the saved images. Needs "
        "transformers -- run this on Kaggle, not in the local image. Still "
        "no generation: minutes, not hours.",
    )
    parser.add_argument(
        "--attribute",
        default=None,
        help="attribute name, required with --from-matrix when the bundle's "
        "baseline filename does not name it",
    )
    parser.add_argument(
        "--metric-device", default="cuda", help="where ClipSeg/CLIP run"
    )
    parser.add_argument(
        "--phrase", default=None, help="action phrase, only needed for --attribute action"
    )
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

    if args.from_matrix:
        _repair_from_matrix(paths, gate, args)
        return

    scorer = masker = None
    rescorable = RESCORABLE
    if args.with_clip:
        from flair_t2i.metrics.embedding import ClipScorer
        from flair_t2i.metrics.masking import ClipSegMasker

        scorer = ClipScorer(device=args.metric_device)
        masker = ClipSegMasker(device=args.metric_device)
        rescorable = tuple(AttributeClass)

    attrs = [a for a in rescorable if paths.baseline_image(a).exists()]
    if not attrs:
        available = sorted(p.stem for p in paths.baselines.glob("*.png"))
        parser.error(
            f"no rescorable attribute found in {args.bundle} (baselines: "
            f"{available}). Recomputing offline supports "
            f"{[a.value for a in RESCORABLE]}; for anything else use "
            "--from-matrix, which gates the saved images and re-normalises "
            "the existing scores without needing CLIP or ClipSeg."
        )

    all_raw, all_rejected, all_reasons = {}, set(), {}
    for attr in attrs:
        raw, rejected, reasons = rescore(
            paths, attr, gate, scorer=scorer, masker=masker, phrase=args.phrase
        )
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
