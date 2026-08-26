import json

import pytest

torch = pytest.importorskip("torch")

from flair_t2i.artifacts import (
    MANIFEST_NAME,
    RunRecord,
    describe_plan,
    load_manifest,
    package_versions,
    save_run,
    summarise,
)
from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.guard import CoherenceGuard, GuardEvent
from flair_t2i.heads import HeadUnit
from flair_t2i.routing import RoutedComponent, RoutingPlan

CFG = FlairConfig(device="cpu")


def _record(run_id="r1", **kw):
    defaults = dict(
        run_id=run_id,
        prompt="a red car",
        seed=0,
        steps=20,
        guidance_scale=4.5,
        routing=True,
        fuzzy=True,
        basm_source="uniform",
        config={"alpha_0": 0.75},
    )
    defaults.update(kw)
    return RunRecord(**defaults)


def _plan():
    return RoutingPlan(
        routed=(
            RoutedComponent(
                component=Component(
                    id="c_color", text="a red car", attr=AttributeClass.COLOR,
                    hedge="very",
                ),
                embedding=torch.ones((2, 2)),
                units=((HeadUnit(7, 0), 0.93), (HeadUnit(3, 0), 0.22)),
                intensity=1.138,
            ),
        ),
        cfg=CFG,
    )


def test_record_fills_provenance_automatically():
    record = _record()
    assert record.timestamp
    assert record.git_commit
    assert "python" in record.versions


def test_package_versions_marks_missing_packages_absent():
    versions = package_versions()
    assert versions["diffusers"] == "absent"  # not in the CPU test image
    assert versions["torch"] != "absent"


def test_describe_plan_flattens_to_json_safe_fields():
    described = describe_plan(_plan())
    entry = described["routed"][0]

    assert entry["attribute"] == "color"
    assert entry["hedge"] == "very"
    assert entry["units"] == [[7, 0, 0.93], [3, 0, 0.22]]
    assert entry["intensity"] == pytest.approx(1.138)
    json.dumps(described)  # must not raise


def test_describe_plan_handles_no_plan():
    assert describe_plan(None) == {
        "routed": [],
        "guard_events": [],
        "alpha_scale": 1.0,
    }


def test_describe_plan_includes_guard_events():
    guard = CoherenceGuard(CFG)
    guard.events.append(GuardEvent(step=2, reason="cross_stream_similarity", value=0.1))

    described = describe_plan(_plan(), guard)

    assert described["guard_events"] == [
        {"step": 2, "reason": "cross_stream_similarity", "value": 0.1}
    ]


def test_save_run_writes_json_and_manifest(tmp_path):
    save_run(tmp_path, _record())

    assert (tmp_path / "r1.json").exists()
    assert (tmp_path / MANIFEST_NAME).exists()
    assert json.loads((tmp_path / "r1.json").read_text())["prompt"] == "a red car"


def test_save_run_without_an_image_records_none(tmp_path):
    save_run(tmp_path, _record())
    assert json.loads((tmp_path / "r1.json").read_text())["image"] is None
    assert not (tmp_path / "r1.png").exists()


def test_save_run_writes_the_image_when_given_one(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (8, 8), (255, 0, 0))

    save_run(tmp_path, _record(), image=image)

    assert (tmp_path / "r1.png").exists()
    assert json.loads((tmp_path / "r1.json").read_text())["image"] == "r1.png"


def test_manifest_appends_rather_than_overwrites(tmp_path):
    save_run(tmp_path, _record("r1"))
    save_run(tmp_path, _record("r2"))

    records = load_manifest(tmp_path)
    assert [r["run_id"] for r in records] == ["r1", "r2"]


def test_load_manifest_of_a_fresh_directory_is_empty(tmp_path):
    assert load_manifest(tmp_path) == []


def test_load_manifest_survives_a_half_written_line(tmp_path):
    save_run(tmp_path, _record("r1"))
    with open(tmp_path / MANIFEST_NAME, "a", encoding="utf-8") as handle:
        handle.write('{"run_id": "r2", "seed":\n')  # interrupted session

    records = load_manifest(tmp_path)
    assert [r["run_id"] for r in records] == ["r1"]


def test_summarise_reports_each_run(tmp_path):
    described = describe_plan(_plan())
    save_run(tmp_path, _record("r1", **described))

    text = summarise(tmp_path)
    assert "1 run(s)" in text
    assert "r1" in text
    assert "[3, 7]" in text


def test_summarise_of_an_empty_directory_says_so(tmp_path):
    assert "no runs recorded" in summarise(tmp_path)
