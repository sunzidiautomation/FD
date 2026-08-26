"""Run the HASM calibration campaign on Kaggle GPU.

Two phases, separately resumable:

    python scripts/calibrate.py prefilter --top-n 10 --out calibration_runs/
    python scripts/calibrate.py hasm --seeds 0 --out calibration_runs/

The HASM phase sweeps every (attribute, block, head) cell and checkpoints
each one as it completes. Re-run the same command after a session times
out and it resumes. The prefilter phase is retained for the FLUX port; the
HASM phase does not read its output.

Choose --top-n from the vitality elbow the prefilter prints; see the master
roadmap section 2.3. Choose --seeds from the budget arithmetic in section
2.2, using the T_GEN the smoke test measured.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy
import torch
from diffusers import StableDiffusion3Pipeline

from flair_t2i.attributes import CORE_ATTRIBUTES
from flair_t2i.calibration.corpus import DEFAULT_CORPUS_PATH, load_corpus
from flair_t2i.calibration.harness import calibrate, make_swap_generate_fn
from flair_t2i.calibration.prefilter import (
    VitalityReport,
    lpips_distance,
    make_bypass_generate_fn,
    run_prefilter,
)
from flair_t2i.config import FlairConfig
from flair_t2i.hasm import HASM
from flair_t2i.metrics.embedding import ClipScorer
from flair_t2i.metrics.masking import ClipSegMasker
from flair_t2i.pipeline import FlairPipeline

PREFILTER_PROMPTS = [
    "a small red sports car under warm evening light",
    "a large ceramic vase on a wooden table",
    "a rusty metal sign beside a brick wall",
]


def _pipeline(cfg: FlairConfig) -> FlairPipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    # Calibration never reads this matrix; it swaps prompts directly.
    placeholder = HASM.uniform((0,), (0,), CORE_ATTRIBUTES)
    return FlairPipeline(pipe, cfg, placeholder, nlp=spacy.load("en_core_web_sm"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prefilter", "hasm"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--vitality", type=Path)
    parser.add_argument("--out", type=Path, default=Path("calibration_runs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = FlairConfig(device="cuda")
    fp = _pipeline(cfg)

    if args.phase == "prefilter":
        n_blocks = len(fp.pipe.transformer.transformer_blocks)
        total = len(PREFILTER_PROMPTS) * len(args.seeds) * (1 + n_blocks)
        print(f"scoring {n_blocks} blocks -- {total} generations")

        report = run_prefilter(
            make_bypass_generate_fn(fp, steps=args.steps),
            n_blocks=n_blocks,
            prompts=PREFILTER_PROMPTS,
            seeds=args.seeds,
            distance_fn=lpips_distance(cfg.device),
            top_n=args.top_n,
        )
        report.save(args.out / "vitality.json")

        ranked = sorted(report.scores.items(), key=lambda p: -p[1])
        print("\nvitality, most vital first:")
        for block, score in ranked[:15]:
            print(f"  B{block:<3} {score:.4f}")
        print(f"\n--top-n {args.top_n} kept: {report.vital_blocks}")
        print(f"elbow rule suggests:  {report.elbow()}")
        return

    corpus = load_corpus(DEFAULT_CORPUS_PATH)
    n_blocks = len(fp.pipe.transformer.transformer_blocks)
    n_heads = fp.pipe.transformer.transformer_blocks[0].attn.heads
    block_ids = tuple(range(n_blocks))
    head_ids = tuple(range(n_heads))

    pairs = sum(len(v) for v in corpus.values())
    units = n_blocks * n_heads
    total = pairs * len(args.seeds) * (1 + units)
    print(f"calibrating {len(corpus)} attributes over {units} head units")
    print(f"  {n_blocks} blocks x {n_heads} heads")
    print(f"  {pairs} pairs x {len(args.seeds)} seed(s) -- up to {total} generations")
    print(f"checkpointing to {args.out / 'cells'} -- safe to re-run after a timeout\n")

    hasm = calibrate(
        make_swap_generate_fn(fp, steps=args.steps),
        corpus=corpus,
        block_ids=block_ids,
        head_ids=head_ids,
        masker=ClipSegMasker(device=cfg.device),
        seeds=args.seeds,
        scorer=ClipScorer(device=cfg.device),
        checkpoint_dir=args.out,
        progress=lambda attr, unit, value: print(
            f"  {attr.value:<9} B{unit.block:<3}H{unit.head:<3} raw={value:.4f}"
        ),
    )
    hasm.save(args.out / "hasm.npz")
    hasm.to_basm().save(args.out / "basm.npz")

    print("\ncalibrated HASM:")
    for attr in hasm.attributes:
        print(f"  {attr.value:<9} top units: {hasm.top_k(attr, 3)}")

    basm = hasm.to_basm()
    peaks = {attr: basm.top_k(attr, 1)[0][0] for attr in basm.attributes}
    if len(set(peaks.values())) == 1:
        print("\n  WARNING: every attribute peaks on the same block.")
        print("  There is no disentanglement to exploit -- see roadmap 2.5.")

    print(f"\nwrote {args.out / 'hasm.npz'} and {args.out / 'basm.npz'}")


if __name__ == "__main__":
    main()
