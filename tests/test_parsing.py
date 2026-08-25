import pytest

spacy = pytest.importorskip("spacy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.parsing import parse_prompt


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def _by_attr(components):
    return {c.attr: c for c in components}


def test_parses_the_running_example(nlp):
    got = _by_attr(parse_prompt("A small red sports car under warm evening light", nlp))

    assert set(got) == {
        AttributeClass.IDENTITY,
        AttributeClass.SIZE,
        AttributeClass.COLOR,
        AttributeClass.LIGHTING,
    }
    assert got[AttributeClass.IDENTITY].text == "sports car"
    assert got[AttributeClass.SIZE].text == "small sports car"
    assert got[AttributeClass.COLOR].text == "red sports car"
    assert "warm evening light" in got[AttributeClass.LIGHTING].text


def test_component_ids_are_stable_and_attribute_derived(nlp):
    components = parse_prompt("a red car", nlp)
    ids = {c.id for c in components}
    assert "c_color" in ids
    assert "c_identity" in ids


def test_detects_hedge_words(nlp):
    got = _by_attr(parse_prompt("a very red car", nlp))
    assert got[AttributeClass.COLOR].hedge == "very"


def test_no_hedge_is_none(nlp):
    got = _by_attr(parse_prompt("a red car", nlp))
    assert got[AttributeClass.COLOR].hedge is None


def test_prompt_with_no_attributes_yields_identity_only(nlp):
    components = parse_prompt("a car", nlp)
    assert [c.attr for c in components] == [AttributeClass.IDENTITY]


def test_texture_and_action_are_recognised(nlp):
    got = _by_attr(parse_prompt("a rusty car driving", nlp))
    assert got[AttributeClass.TEXTURE].text == "rusty car"
    assert AttributeClass.ACTION in got
