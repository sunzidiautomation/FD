"""Contrastive prompt pairs for BASM calibration (spec section 3.4).

Each pair differs from its partner in exactly one word, so the measured
change is attributable to one attribute. A pair that changes two things
measures two things, and the resulting BASM column would be meaningless --
so ``validate_corpus`` enforces that invariant at load time rather than
leaving it to reviewers of the JSON.

``object_label`` is a segmentation query for the BASELINE image, which is
where the harness computes the mask. Identity pairs legitimately name a
different object in ``changed``; the label describes ``base``.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

from ..attributes import AttributeClass

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs.json"
)

#: Seed count shipped with the repo. Spec section 3.4 targets ~10; the
#: campaign plan raises this once per-generation wall time is measured.
MIN_PAIRS_PER_ATTRIBUTE = 5


def _words(text: str) -> list[str]:
    return [word.strip(string.punctuation) for word in text.lower().split()]


@dataclass(frozen=True)
class ContrastivePair:
    base: str
    changed: str
    object_label: str
    phrase: str | None = None

    @property
    def changed_word(self) -> str | None:
        """The single word ``changed`` introduces, or None if that is unclear.

        ``validate_corpus`` already guarantees exactly one word of
        difference, and that word names what the swap injected -- "blue"
        out of "a blue sports car on a road". A target-directed metric
        needs it to know what to measure toward.

        None rather than a guess when two words differ or the lengths do:
        attributing an image change to the wrong word is the failure the
        one-word invariant exists to prevent.
        """
        base, changed = _words(self.base), _words(self.changed)
        if len(base) != len(changed):
            return None
        differing = [c for b, c in zip(base, changed) if b != c]
        return differing[0] if len(differing) == 1 else None


def load_corpus(path: str | Path) -> dict[AttributeClass, list[ContrastivePair]]:
    with open(Path(path), encoding="utf-8") as handle:
        raw = json.load(handle)

    corpus = {
        AttributeClass(name): [ContrastivePair(**entry) for entry in entries]
        for name, entries in raw.items()
    }
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: dict[AttributeClass, list[ContrastivePair]]) -> None:
    """Raise unless every structural invariant the harness relies on holds."""
    missing = set(AttributeClass) - set(corpus)
    if missing:
        raise ValueError(
            f"corpus is missing attributes: {sorted(a.value for a in missing)}"
        )

    for attr, pairs in corpus.items():
        if len(pairs) < MIN_PAIRS_PER_ATTRIBUTE:
            raise ValueError(
                f"{attr.value} needs at least {MIN_PAIRS_PER_ATTRIBUTE} pairs, "
                f"got {len(pairs)}"
            )

        for pair in pairs:
            if attr is AttributeClass.ACTION and not pair.phrase:
                raise ValueError(f"action pair {pair.base!r} needs a phrase")

            base_words = pair.base.split()
            changed_words = pair.changed.split()
            differing = (
                sum(a != b for a, b in zip(base_words, changed_words))
                if len(base_words) == len(changed_words)
                else -1
            )
            if differing != 1:
                raise ValueError(
                    f"{attr.value} pair must differ in exactly one word: "
                    f"{pair.base!r} vs {pair.changed!r}"
                )
