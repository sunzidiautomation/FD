import json

import pytest

from flair_t2i.attributes import AttributeClass
from flair_t2i.calibration.corpus import (
    DEFAULT_CORPUS_PATH,
    MIN_PAIRS_PER_ATTRIBUTE,
    ContrastivePair,
    load_corpus,
    validate_corpus,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DEFAULT_CORPUS_PATH)


def test_corpus_covers_all_seven_attributes(corpus):
    assert set(corpus) == set(AttributeClass)


def test_every_attribute_meets_the_minimum_pair_count(corpus):
    for attr, pairs in corpus.items():
        assert len(pairs) >= MIN_PAIRS_PER_ATTRIBUTE, attr


def test_pairs_differ_in_exactly_one_word(corpus):
    """The invariant that makes a measured delta attributable."""
    for attr, pairs in corpus.items():
        for pair in pairs:
            base_words = pair.base.split()
            changed_words = pair.changed.split()
            assert len(base_words) == len(changed_words), (attr, pair.base)
            differing = sum(a != b for a, b in zip(base_words, changed_words))
            assert differing == 1, (attr, pair.base, pair.changed)


def test_object_label_appears_in_the_base_prompt(corpus):
    """The mask is segmented from the baseline image, so the label must fit it."""
    for attr, pairs in corpus.items():
        for pair in pairs:
            assert pair.object_label in pair.base, (attr, pair.base)


def test_action_pairs_carry_a_phrase(corpus):
    for pair in corpus[AttributeClass.ACTION]:
        assert pair.phrase


def test_validate_rejects_a_missing_attribute():
    only_color = {
        AttributeClass.COLOR: [ContrastivePair("a red car", "a blue car", "car")]
        * MIN_PAIRS_PER_ATTRIBUTE
    }
    with pytest.raises(ValueError, match="missing"):
        validate_corpus(only_color)


def test_validate_rejects_too_few_pairs(corpus):
    thin = {attr: pairs[:1] for attr, pairs in corpus.items()}
    with pytest.raises(ValueError, match="at least"):
        validate_corpus(thin)


def test_validate_rejects_an_action_pair_without_a_phrase(corpus):
    broken = {attr: list(pairs) for attr, pairs in corpus.items()}
    broken[AttributeClass.ACTION] = [
        ContrastivePair(p.base, p.changed, p.object_label, phrase=None)
        for p in broken[AttributeClass.ACTION]
    ]
    with pytest.raises(ValueError, match="phrase"):
        validate_corpus(broken)


def test_validate_rejects_a_pair_that_changes_two_words(corpus):
    broken = {attr: list(pairs) for attr, pairs in corpus.items()}
    broken[AttributeClass.COLOR] = [
        ContrastivePair("a red sports car", "a blue vintage car", "sports car")
    ] * MIN_PAIRS_PER_ATTRIBUTE
    with pytest.raises(ValueError, match="exactly one word"):
        validate_corpus(broken)


def test_validate_rejects_a_pair_of_different_lengths(corpus):
    broken = {attr: list(pairs) for attr, pairs in corpus.items()}
    broken[AttributeClass.COLOR] = [
        ContrastivePair("a red car", "a blue sports car", "car")
    ] * MIN_PAIRS_PER_ATTRIBUTE
    with pytest.raises(ValueError, match="exactly one word"):
        validate_corpus(broken)


def test_corpus_file_is_valid_json():
    with open(DEFAULT_CORPUS_PATH, encoding="utf-8") as handle:
        json.load(handle)


# --------------------------------------------------- the one word that differs
#
# validate_corpus already enforces exactly one word of difference. That
# word names what the swap injected, which is what a target-directed
# metric needs -- "blue" out of "a blue sports car on a road".


def test_changed_word_is_the_one_that_differs():
    pair = ContrastivePair(
        base="a red sports car on a road",
        changed="a blue sports car on a road",
        object_label="sports car",
    )
    assert pair.changed_word == "blue"


def test_changed_word_ignores_case_and_punctuation():
    pair = ContrastivePair(
        base="A Red sports car.", changed="A Blue sports car.", object_label="car"
    )
    assert pair.changed_word == "blue"


def test_changed_word_is_none_when_the_lengths_differ():
    pair = ContrastivePair(
        base="a red car", changed="a red sports car", object_label="car"
    )
    assert pair.changed_word is None


def test_changed_word_is_none_when_two_words_differ():
    """Deliberately silent rather than guessing which change to attribute."""
    pair = ContrastivePair(
        base="a red small car", changed="a blue large car", object_label="car"
    )
    assert pair.changed_word is None


def test_every_shipped_colour_pair_names_a_colour():
    """The metric can only be directed if the corpus actually names a hue."""
    from PIL import ImageColor

    from flair_t2i.attributes import AttributeClass

    pairs = load_corpus(DEFAULT_CORPUS_PATH)[AttributeClass.COLOR]
    for pair in pairs:
        assert pair.changed_word in ImageColor.colormap, pair.changed
