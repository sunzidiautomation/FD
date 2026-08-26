"""Week 1 exit criterion: routed generation runs end-to-end on SD3.5-M.

Run on Kaggle GPU:
    python scripts/smoke_test.py --steps 20 --out outputs/

This is a manual script, not a pytest test -- it needs a GPU and the
SD3.5-M weights. It uses an UNCALIBRATED uniform HASM, so the images prove
the plumbing works, not that routing helps. Real HASM values arrive in
Week 3.

Every generation is saved with its full provenance (prompt, seed, config,
routing plan, guard events, git commit, package versions) so the images
stay interpretable months later -- see flair_t2i/artifacts.py.

Record the numbers this prints -- N_BLOCKS, N_HEADS, and T_GEN -- in
calibration_runs/measurements.txt. The calibration campaign's parameters
are derived from them (master roadmap, section 2).
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy
import torch
from diffusers import StableDiffusion3Pipeline

from flair_t2i.artifacts import RunRecord, describe_plan, save_run, summarise
from flair_t2i.attributes import CORE_ATTRIBUTES
from flair_t2i.config import FlairConfig
from flair_t2i.hasm import HASM
from flair_t2i.pipeline import FlairPipeline

PROMPT = "A small red sports car under warm evening light"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--tag", default="smoke", help="prefix for run ids")
    parser.add_argument("--hasm", type=Path, help="a calibrated hasm.npz")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = FlairConfig(device="cuda")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    n_blocks = len(pipe.transformer.transformer_blocks)
    n_heads = pipe.transformer.transformer_blocks[0].attn.heads
    print(f"N_BLOCKS = {n_blocks}")
    print(f"N_HEADS  = {n_heads}")

    if args.hasm:
        hasm = HASM.load(args.hasm)
        hasm_source = f"calibrated:{args.hasm}"
    else:
        hasm = HASM.uniform(
            tuple(range(n_blocks)), tuple(range(n_heads)), CORE_ATTRIBUTES
        )
        hasm_source = "uniform (UNCALIBRATED placeholder)"
    print(f"HASM     {hasm_source}")

    fp = FlairPipeline(pipe, cfg, hasm, nlp=spacy.load("en_core_web_sm"))

    def record_run(run_id: str, prompt: str, image, routing: bool, fuzzy: bool):
        described = describe_plan(fp.last_plan, fp.last_guard)
        save_run(
            args.out,
            RunRecord(
                run_id=run_id,
                prompt=prompt,
                seed=args.seed,
                steps=args.steps,
                guidance_scale=4.5,
                routing=routing,
                fuzzy=fuzzy,
                basm_source=hasm_source,
                config=asdict(cfg),
                tag=args.tag,
                **described,
            ),
            image=image,
        )

    # --- baseline (routing off) -----------------------------------------
    started = time.perf_counter()
    baseline = fp.generate(PROMPT, seed=args.seed, steps=args.steps, routing=False)
    t_gen = time.perf_counter() - started
    record_run(f"{args.tag}_baseline", PROMPT, baseline, routing=False, fuzzy=False)
    print(f"T_GEN = {t_gen:.1f}s  (one {args.steps}-step generation)")

    # --- routed ----------------------------------------------------------
    routed = fp.generate(PROMPT, seed=args.seed, steps=args.steps, routing=True)
    record_run(f"{args.tag}_routed", PROMPT, routed, routing=True, fuzzy=True)

    assert fp.last_plan is not None, "routing produced no plan"
    print(f"routed components: {[rc.component.id for rc in fp.last_plan.routed]}")
    print(f"units touched:     {sorted(fp.last_plan.routed[0].units)}")
    print(f"blocks touched:    {sorted(fp.last_plan.blocks_touched())}")
    print(f"guard events:      {len(fp.last_guard.events)}")

    # --- hedge ladder ----------------------------------------------------
    print("\nhedge ladder:")
    for hedge in ["slightly", "", "very"]:
        text = f"A {hedge} small red sports car under warm evening light".replace(
            "  ", " "
        )
        image = fp.generate(text, seed=args.seed, steps=args.steps)
        name = hedge or "plain"
        record_run(f"{args.tag}_hedge_{name}", text, image, routing=True, fuzzy=True)
        intensity = fp.last_plan.routed[0].intensity if fp.last_plan.routed else None
        print(f"  {name:<9} intensity={intensity}")

    # --- the measurements the campaign budget needs ----------------------
    runs = Path("calibration_runs")
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "measurements.txt").write_text(
        f"N_BLOCKS={n_blocks}\nN_HEADS={n_heads}\nT_GEN={t_gen:.2f}\nSTEPS={args.steps}\n"
    )
    print(f"\nwrote {runs / 'measurements.txt'}")
    print(f"\n{summarise(args.out)}")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
