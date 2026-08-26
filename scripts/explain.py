"""Explain FLAIR's routing decisions for a prompt -- CPU only, no GPU, no model.

Everything up to text encoding is pure computation: parsing, fuzzy hedge
resolution, HASM lookup, head selection, and the alpha schedule. This
script runs all of it and prints what FLAIR *would* do, so you can inspect
and tune routing before spending Kaggle GPU time.

    python scripts/explain.py "A small red sports car under warm evening light"
    python scripts/explain.py "a very rusty car driving" --steps 8
    python scripts/explain.py "a slightly red car" --hasm calibration_runs/hasm.npz

Add ``--save`` to persist the routing record (no image -- that needs a GPU)::

    python scripts/explain.py "a very red car" --save outputs/

In Docker the container filesystem is thrown away, so mount a volume or the
saved files never reach the host::

    docker run --rm -v "$(pwd)/outputs:/app/outputs" flair-test \\
        python scripts/explain.py "a very red car" --save outputs/

Embeddings are deterministic fakes -- real ones need the T5/CLIP encoders.
That affects only the coherence-guard cosine check; every routing decision
shown here is exactly what the real pipeline computes.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from flair_t2i.artifacts import RunRecord, describe_plan, save_run
from flair_t2i.attributes import CORE_ATTRIBUTES, AttributeClass
from flair_t2i.config import FlairConfig
from flair_t2i.fuzzy.resolve import resolve_components
from flair_t2i.guard import CoherenceGuard
from flair_t2i.hasm import HASM
from flair_t2i.parsing import parse_prompt
from flair_t2i.routing import build_routing_plan

SEQ, DIM = 8, 16


def synthetic_hasm(
    n_blocks: int, n_heads: int, attributes: tuple[AttributeClass, ...]
) -> HASM:
    """A stand-in HASM: each attribute peaks on a different (block, head)."""
    rng = np.random.default_rng(0)
    tensor = rng.uniform(0.05, 0.35, size=(n_blocks, n_heads, len(attributes)))
    stride = max(1, n_blocks // max(1, len(attributes)))
    for col in range(len(attributes)):
        block = (3 + col * stride) % n_blocks
        head = col % n_heads
        tensor[block, head, col] = rng.uniform(0.80, 0.95)
    return HASM(
        tensor=tensor,
        block_ids=tuple(range(n_blocks)),
        head_ids=tuple(range(n_heads)),
        attributes=attributes,
    )


def fake_embedding(text: str) -> torch.Tensor:
    """Deterministic stand-in for a T5/CLIP encoding of ``text``."""
    generator = torch.Generator().manual_seed(zlib.crc32(text.encode()) % (2**31))
    return torch.randn((SEQ, DIM), generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompt")
    parser.add_argument("--blocks", type=int, default=24, help="N_BLOCKS to assume")
    parser.add_argument("--heads", type=int, default=24, help="N_HEADS to assume")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hasm", type=Path, help="a real calibrated hasm.npz")
    parser.add_argument("--no-fuzzy", action="store_true")
    parser.add_argument("--alpha-0", type=float, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=Path("outputs"),
        help="save the routing record (no image) to this directory",
    )
    parser.add_argument("--tag", default="explain", help="prefix for the run id")
    args = parser.parse_args()

    cfg = FlairConfig(device="cpu")
    if args.alpha_0 is not None:
        cfg.alpha_0 = args.alpha_0

    if args.hasm:
        hasm = HASM.load(args.hasm)
        source = f"calibrated -- {args.hasm}"
    else:
        hasm = synthetic_hasm(args.blocks, args.heads, CORE_ATTRIBUTES)
        source = "SYNTHETIC placeholder (real one arrives in Week 3-4)"

    print(f'\nPrompt   "{args.prompt}"')
    print(f"HASM     {source}")
    print(f"Config   alpha_0={cfg.alpha_0}  t_window={cfg.t_window}  "
          f"top_k={cfg.top_k_default}\n")

    # ---- 1. parse -------------------------------------------------------
    components = parse_prompt(args.prompt, nlp=None)
    print("1. PARSED COMPONENTS                          (parsing.py)")
    print(f"   {'attribute':<10} {'hedge':<10} text")
    for c in components:
        print(f"   {c.attr.value:<10} {str(c.hedge or '-'):<10} {c.text}")

    routable = [c for c in components if c.attr in hasm.attributes]
    dropped = [c.attr.value for c in components if c.attr not in hasm.attributes]
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
        routable, embeddings, hasm, cfg, intensities, k_overrides
    )

    print("\n3. ROUTING PLAN                               (routing.py + hasm.py)")
    print(f"   {'attribute':<10} {'units (score)':<34} intensity")
    for rc in plan.routed:
        units = "  ".join(f"B{u.block}H{u.head}({s:.2f})" for u, s in rc.units)
        print(f"   {rc.component.attr.value:<10} {units:<34} {rc.intensity:.3f}")
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
    print("   alpha = alpha_0 * S[l,h,a] * intensity * sched(t) * alpha_scale\n")
    header = "   step  frac  " + "".join(
        f"{rc.component.attr.value[:8]:>9}" for rc in plan.routed
    )
    print(header)
    for step in range(args.steps):
        frac = step / args.steps
        row = f"   {step:>4}  {frac:>4.2f}  "
        for rc in plan.routed:
            top_unit = rc.units[0][0]
            row += f"{plan.alpha(rc, top_unit, frac):>9.3f}"
        print(row)

    active = sum(
        1
        for step in range(args.steps)
        if any(
            plan.alpha(rc, rc.units[0][0], step / args.steps) > 0
            for rc in plan.routed
        )
    )
    print(f"\n   injection active for {active}/{args.steps} steps, "
          f"then the base prompt runs alone.")

    # ---- 6. save --------------------------------------------------------
    if args.save:
        slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40]
        digest = zlib.crc32(
            f"{args.prompt}{cfg.alpha_0}{args.no_fuzzy}{source}".encode()
        )
        record = RunRecord(
            run_id=f"{args.tag}_{slug}_{digest:08x}",
            prompt=args.prompt,
            seed=-1,  # no sampling happened; routing is seed-independent
            steps=args.steps,
            guidance_scale=0.0,
            routing=True,
            fuzzy=not args.no_fuzzy,
            basm_source=source,
            config=asdict(cfg),
            tag=args.tag,
            notes="routing decisions only -- no image, CPU dry run",
            **describe_plan(plan, guard),
        )
        path = save_run(args.save, record)
        print(f"\n   saved {path}")
        print("   (routing record only -- generating the image needs a GPU)")

    print()


if __name__ == "__main__":
    main()
