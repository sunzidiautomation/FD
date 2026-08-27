import numpy as np

from flair_t2i.attributes import AttributeClass
from flair_t2i.demo.report import heat_color, render_report, write_report
from flair_t2i.demo.sweep import DemoPaths
from flair_t2i.hasm import HASM

ATTRS = (AttributeClass.COLOR, AttributeClass.SIZE)


def _hasm():
    tensor = np.array(
        [
            [[0.10, 0.90], [0.20, 0.30]],
            [[1.00, 0.05], [0.40, 0.60]],
        ]
    )
    return HASM(tensor, (0, 1), (0, 1), ATTRS)


def test_heat_color_is_a_css_colour():
    assert heat_color(0.0).startswith("rgb(")
    assert heat_color(1.0).startswith("rgb(")


def test_heat_color_is_monotonic_in_score():
    assert heat_color(0.0) != heat_color(1.0)


def test_report_names_every_attribute(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    assert "color" in html
    assert "size" in html


def test_report_marks_the_peak_unit_per_attribute(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    # COLOR peaks at block 1, head 0
    assert "block 1, head 0" in html


def test_report_links_every_head_image(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    for block in (0, 1):
        for head in (0, 1):
            assert f"heads/color_b{block}_h{head}.png" in html


def test_report_is_self_contained_html(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    assert html.lstrip().startswith("<!doctype html>")
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html


def test_rejected_units_are_marked_and_not_clickable(tmp_path):
    """A rejected cell is a destroyed frame, not a low score.

    Rendering it like a zero would tell the reader 'this head does not
    affect the attribute', when what happened is 'this head is load-bearing
    and disturbing it broke the image'. Opposite meanings.
    """
    from flair_t2i.heads import HeadUnit

    html = render_report(
        _hasm(), DemoPaths(tmp_path), title="FLAIR", rejected={HeadUnit(1, 0)}
    )

    assert "rejected" in html
    # the surviving cells still link to their images
    assert "heads/color_b0_h0.png" in html


def test_write_report_creates_index_html(tmp_path):
    path = write_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    assert path == tmp_path / "index.html"
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
