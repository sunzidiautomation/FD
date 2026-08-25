"""Week 1 exit criterion: routed generation runs end-to-end on SD3.5-M.

Run on Kaggle GPU:
    python scripts/smoke_test.py --steps 20 --out outputs/

This is a manual script, not a pytest test -- it needs a GPU and the
SD3.5-M weights. It uses an UNCALIBRATED uniform BASM, so the images prove
the plumbing works, not that routing helps. Real BASM values arrive in
Week 3.

Record the two numbers this prints -- N_BLOCKS and the baseline generation
time -- in calibration_runs/measurements.txt. The calibration campaign's
parameters are derived from them (see the master roadmap, section 2).
"""

import argparse
import time
from pathlib import Path

import spacy
import torch
from diffusers import StableDiffusion3Pipeline

from flair_t2i.attributes import CORE_ATTRIBUTES
from flair_t2i.basm import BASM
from flair_t2i.config import FlairConfig
from flair_t2i.pipeline import FlairPipeline

PROMPT = "A small red sports car under warm evening light"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = FlairConfig(device="cuda")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    n_blocks = len(pipe.transformer.transformer_blocks)
    print(f"N_BLOCKS = {n_blocks}")

    basm = BASM.uniform(tuple(range(n_blocks)), CORE_ATTRIBUTES)
    fp = FlairPipeline(pipe, cfg, basm, nlp=spacy.load("en_core_web_sm"))

    started = time.perf_counter()
    baseline = fp.generate(PROMPT, seed=args.seed, steps=args.steps, routing=False)
    t_gen = time.perf_counter() - started
    baseline.save(args.out / "smoke_baseline.png")
    print(f"T_GEN = {t_gen:.1f}s  (one {args.steps}-step generation)")

    routed = fp.generate(PROMPT, seed=args.seed, steps=args.steps, routing=True)
    routed.save(args.out / "smoke_routed.png")

    assert fp.last_plan is not None, "routing produced no plan"
    print(f"routed components: {[rc.component.id for rc in fp.last_plan.routed]}")
    print(f"blocks touched:    {sorted(fp.last_plan.blocks_touched())}")
    print(f"guard events:      {len(fp.last_guard.events)}")

    for hedge in ["slightly", "", "very"]:
        text = f"A {hedge} small red sports car under warm evening light".replace(
            "  ", " "
        )
        image = fp.generate(text, seed=args.seed, steps=args.steps)
        name = hedge or "plain"
        image.save(args.out / f"smoke_hedge_{name}.png")
        intensity = fp.last_plan.routed[0].intensity if fp.last_plan.routed else None
        print(f"  {name:<9} intensity={intensity}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
