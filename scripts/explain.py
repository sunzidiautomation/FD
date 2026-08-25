"""Explain FLAIR's routing decisions for a prompt -- CPU only, no GPU, no model.

Everything up to text encoding is pure computation: parsing, fuzzy hedge
resolution, BASM lookup, block selection, and the alpha schedule. This
script runs all of it and prints what FLAIR *would* do, so you can inspect
and tune routing before spending Kaggle GPU time.

    python scripts/explain.py "A small red sports car under warm evening light"
    python scripts/explain.py "a very rusty car driving" --steps 8
    python scripts/explain.py "a slightly red car" --basm calibration_runs/basm.npz

Embeddings are deterministic fakes -- real ones need the T5/CLIP encoders.
That affects only the coherence-guard cosine check; every routing decision
shown here is exactly what the real pipeline computes.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import torch

from flair_t2i.attributes import CORE_ATTRIBUTES, AttributeClass
from flair_t2i.basm import BASM
from flair_t2i.config import FlairConfig
from flair_t2i.fuzzy.resolve import resolve_components
from flair_t2i.guard import CoherenceGuard
from flair_t2i.parsing import parse_prompt
from flair_t2i.routing import build_routing_plan

SEQ, DIM = 8, 16


def synthetic_basm(n_blocks: int, attributes: tuple[AttributeClass, ...]) -> BASM:
    """A stand-in matrix where each attribute peaks on its own block.

    Clearly labelled as synthetic wherever it is used. The real matrix
    arrives from the Week 3-4 calibration campaign.
    """
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0.05, 0.35, size=(n_blocks, len(attributes)))
    stride = max(1, n_blocks // max(len(attributes), 1))
    for col in range(len(attributes)):
        matrix[(3 + col * stride) % n_blocks, col] = rng.uniform(0.80, 0.95)
    return BASM(
        matrix=matrix, block_ids=tuple(range(n_blocks)), attributes=attributes
    )


def fake_embedding(text: str) -> torch.Tensor:
    """Deterministic stand-in for a T5/CLIP encoding of ``text``."""
    generator = torch.Generator().manual_seed(zlib.crc32(text.encode()) % (2**31))
    return torch.randn((SEQ, DIM), generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompt")
    parser.add_argument("--blocks", type=int, default=24, help="N_BLOCKS to assume")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--basm", type=Path, help="a real calibrated basm.npz")
    parser.add_argument("--no-fuzzy", action="store_true")
    parser.add_argument("--alpha-0", type=float, default=None)
    args = parser.parse_args()

    cfg = FlairConfig(device="cpu")
    if args.alpha_0 is not None:
        cfg.alpha_0 = args.alpha_0

    if args.basm:
        basm = BASM.load(args.basm)
        source = f"calibrated -- {args.basm}"
    else:
        basm = synthetic_basm(args.blocks, CORE_ATTRIBUTES)
        source = "SYNTHETIC placeholder (real one arrives in Week 3-4)"

    print(f'\nPrompt   "{args.prompt}"')
    print(f"BASM     {source}")
    print(f"Config   alpha_0={cfg.alpha_0}  t_window={cfg.t_window}  "
          f"top_k={cfg.top_k_default}\n")

    # ---- 1. parse -------------------------------------------------------
    components = parse_prompt(args.prompt, nlp=None)
    print("1. PARSED COMPONENTS                          (parsing.py)")
    print(f"   {'attribute':<10} {'hedge':<10} text")
    for c in components:
        print(f"   {c.attr.value:<10} {str(c.hedge or '-'):<10} {c.text}")

    routable = [c for c in components if c.attr in basm.attributes]
    dropped = [c.attr.value for c in components if c.attr not in basm.attributes]
    if dropped:
        print(f"\n   dropped (no calibrated column): {', '.join(dropped)}")
    if not routable:
        print("\n   Nothing routable. Stopping.")
        return

    # ---- 2. fuzzy -------------------------------------------------------
    if args.no_fuzzy:
        intensities, k_overrides = {}, {}
        print("\n2. FUZZY                                      (disabled)")
    else:
        intensities, k_overrides, results = resolve_components(routable)
        print("\n2. FUZZY HEDGE RESOLUTION                     (fuzzy/hedges.py)")
        print(f"   {'attribute':<10} {'operator':<13} {'intensity':>9}  k")
        for c in routable:
            r = results[c.id]
            print(f"   {c.attr.value:<10} {r.kind.value:<13} "
                  f"{r.intensity:>9.3f}  {r.k}")

    # ---- 3. routing plan ------------------------------------------------
    embeddings = {c.id: fake_embedding(c.text) for c in routable}
    plan = build_routing_plan(
        routable, embeddings, basm, cfg, intensities, k_overrides
    )

    print("\n3. ROUTING PLAN                               (routing.py + basm.py)")
    print(f"   {'attribute':<10} {'blocks (score)':<34} intensity")
    for rc in plan.routed:
        blocks = "  ".join(f"B{b}({s:.2f})" for b, s in rc.blocks)
        print(f"   {rc.component.attr.value:<10} {blocks:<34} {rc.intensity:.3f}")
    print(f"\n   blocks touched: {sorted(plan.blocks_touched())}")

    # ---- 4. guard -------------------------------------------------------
    guard = CoherenceGuard(cfg)
    event = guard.check_streams(plan, step=0)
    guard.apply(plan, event)
    print("\n4. COHERENCE GUARD                            (guard.py)")
    if event is None:
        print(f"   no violation (threshold {cfg.guard_cos_threshold}) -- "
              f"alpha_scale stays {plan.alpha_scale:.2f}")
    else:
        print(f"   {event.reason} = {event.value:.3f} < "
              f"{cfg.guard_cos_threshold} -> alpha_scale {plan.alpha_scale:.2f}")
    print("   (uses fake embeddings -- indicative only)")

    # ---- 5. alpha schedule ----------------------------------------------
    print("\n5. INJECTION STRENGTH PER STEP                (schedule.py + routing.py)")
    print("   alpha = alpha_0 * S[l,a] * intensity * sched(t) * alpha_scale\n")
    header = "   step  frac  " + "".join(
        f"{rc.component.attr.value[:8]:>9}" for rc in plan.routed
    )
    print(header)
    for step in range(args.steps):
        frac = step / args.steps
        row = f"   {step:>4}  {frac:>4.2f}  "
        for rc in plan.routed:
            top_block = rc.blocks[0][0]
            row += f"{plan.alpha(rc, top_block, frac):>9.3f}"
        print(row)

    active = sum(
        1
        for step in range(args.steps)
        if any(
            plan.alpha(rc, rc.blocks[0][0], step / args.steps) > 0
            for rc in plan.routed
        )
    )
    print(f"\n   injection active for {active}/{args.steps} steps, "
          f"then the base prompt runs alone.\n")


if __name__ == "__main__":
    main()
