import pytest
from PIL import Image

from flair_t2i.latents import LatentRecorder


def _decode(latents):
    return Image.new("RGB", (8, 8), (int(latents), 0, 0))


def test_target_steps_map_fractions_onto_step_indices():
    recorder = LatentRecorder(_decode, at=(0.0, 0.5))
    assert recorder.target_steps(20) == {0, 10}


def test_target_steps_never_exceed_the_last_index():
    recorder = LatentRecorder(_decode, at=(1.0,))
    assert recorder.target_steps(10) == {9}


def test_records_only_at_target_steps():
    recorder = LatentRecorder(_decode, at=(0.0, 0.5))
    for step in range(20):
        recorder(step, 20, step)

    assert [frac for frac, _ in recorder.frames] == [0.0, 0.5]


def test_frames_carry_the_decoded_image():
    recorder = LatentRecorder(_decode, at=(0.0,))
    recorder(0, 4, 42)

    assert recorder.frames[0][1].getpixel((0, 0)) == (42, 0, 0)


def test_zero_total_steps_records_nothing():
    recorder = LatentRecorder(_decode, at=(0.0,))
    recorder(0, 0, 1)
    assert recorder.frames == []


def test_reset_clears_frames_between_generations():
    recorder = LatentRecorder(_decode, at=(0.0,))
    recorder(0, 4, 1)
    recorder.reset()
    assert recorder.frames == []
