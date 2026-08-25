import pytest
from PIL import Image

from flair_t2i.calibration.prefilter import VitalityReport, run_prefilter

PROMPTS = ["a red car", "a blue vase"]
SEEDS = [0, 1]

#: Block 2 is the one that matters in this fake model.
VITAL = 2


def fake_generate(prompt: str, seed: int, bypass: int | None) -> Image.Image:
    shade = 200 if bypass == VITAL else 100
    return Image.new("RGB", (8, 8), (shade, shade, shade))


def fake_distance(a: Image.Image, b: Image.Image) -> float:
    return abs(a.getpixel((0, 0))[0] - b.getpixel((0, 0))[0]) / 255.0


def test_prefilter_ranks_the_vital_block_first():
    report = run_prefilter(
        fake_generate,
        n_blocks=4,
        prompts=PROMPTS,
        seeds=SEEDS,
        distance_fn=fake_distance,
        top_n=1,
    )
    assert report.vital_blocks == (VITAL,)


def test_prefilter_scores_every_block():
    report = run_prefilter(
        fake_generate,
        n_blocks=4,
        prompts=PROMPTS,
        seeds=SEEDS,
        distance_fn=fake_distance,
        top_n=2,
    )
    assert set(report.scores) == {0, 1, 2, 3}


def test_non_vital_blocks_score_zero():
    report = run_prefilter(
        fake_generate,
        n_blocks=4,
        prompts=PROMPTS,
        seeds=SEEDS,
        distance_fn=fake_distance,
        top_n=4,
    )
    assert report.scores[0] == pytest.approx(0.0)
    assert report.scores[VITAL] > 0.0


def test_vital_blocks_are_returned_in_ascending_order():
    report = run_prefilter(
        fake_generate,
        n_blocks=4,
        prompts=PROMPTS,
        seeds=SEEDS,
        distance_fn=fake_distance,
        top_n=3,
    )
    assert list(report.vital_blocks) == sorted(report.vital_blocks)


def test_top_n_clamps_to_the_block_count():
    report = run_prefilter(
        fake_generate,
        n_blocks=3,
        prompts=PROMPTS,
        seeds=SEEDS,
        distance_fn=fake_distance,
        top_n=99,
    )
    assert len(report.vital_blocks) == 3


def test_baselines_are_generated_once_per_prompt_seed_not_per_block():
    """The whole campaign budget depends on this reuse."""
    calls = {"baseline": 0, "bypass": 0}

    def counting_generate(prompt, seed, bypass):
        calls["baseline" if bypass is None else "bypass"] += 1
        return fake_generate(prompt, seed, bypass)

    run_prefilter(
        counting_generate,
        n_blocks=4,
        prompts=PROMPTS,
        seeds=SEEDS,
        distance_fn=fake_distance,
        top_n=1,
    )

    assert calls["baseline"] == len(PROMPTS) * len(SEEDS)
    assert calls["bypass"] == len(PROMPTS) * len(SEEDS) * 4


def test_report_is_serialisable(tmp_path):
    report = run_prefilter(
        fake_generate,
        n_blocks=3,
        prompts=PROMPTS,
        seeds=SEEDS,
        distance_fn=fake_distance,
        top_n=2,
    )
    path = tmp_path / "vitality.json"
    report.save(path)

    restored = VitalityReport.load(path)
    assert restored.vital_blocks == report.vital_blocks
    assert restored.scores == pytest.approx(report.scores)


def _report(values: list[float]) -> VitalityReport:
    scores = {i: v for i, v in enumerate(values)}
    return VitalityReport(scores=scores, vital_blocks=())


def test_elbow_keeps_blocks_scoring_at_least_half_the_top():
    # seven blocks clear the 0.5 cutoff, which is inside [6, 12]
    report = _report([1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.51, 0.1, 0.05, 0.0])
    assert report.elbow() == (0, 1, 2, 3, 4, 5, 6)


def test_elbow_clamps_up_when_too_few_blocks_qualify():
    report = _report([1.0, 0.9] + [0.05] * 8)
    assert len(report.elbow(low=6, high=12)) == 6


def test_elbow_clamps_down_when_too_many_qualify():
    report = _report([1.0] * 20)
    assert len(report.elbow(low=6, high=12)) == 12


def test_elbow_returns_blocks_in_ascending_order():
    report = _report([0.2, 1.0, 0.9, 0.1, 0.8, 0.7, 0.6, 0.55])
    assert list(report.elbow()) == sorted(report.elbow())
