import pytest

from flair_t2i.attributes import AttributeClass, CORE_ATTRIBUTES
from flair_t2i.components import Component, TextBatchLayout


def test_seven_attribute_classes_exist():
    assert len(list(AttributeClass)) == 7
    assert AttributeClass.COLOR.value == "color"
    assert AttributeClass.ACTION.value == "action"


def test_core_attributes_are_the_documented_four():
    assert CORE_ATTRIBUTES == (
        AttributeClass.IDENTITY,
        AttributeClass.COLOR,
        AttributeClass.SIZE,
        AttributeClass.LIGHTING,
    )


def test_layout_row_zero_is_always_base():
    layout = TextBatchLayout(component_ids=("c_color", "c_size"))
    assert layout.BASE_ROW == 0
    assert layout.n_rows() == 3
    assert layout.row_for("c_color") == 1
    assert layout.row_for("c_size") == 2


def test_layout_rejects_wrong_batch_size():
    layout = TextBatchLayout(component_ids=("c_color",))
    layout.validate(2)  # base + 1 component: OK
    with pytest.raises(ValueError, match="expected 2 rows"):
        layout.validate(3)


def test_layout_rejects_unknown_component():
    layout = TextBatchLayout(component_ids=("c_color",))
    with pytest.raises(KeyError, match="c_bogus"):
        layout.row_for("c_bogus")


def test_layout_rejects_duplicate_component_ids():
    with pytest.raises(ValueError, match="duplicate"):
        TextBatchLayout(component_ids=("c_color", "c_color"))


def test_component_is_frozen():
    c = Component(id="c_color", text="a red sports car", attr=AttributeClass.COLOR)
    assert c.hedge is None
    with pytest.raises(Exception):
        c.text = "changed"
