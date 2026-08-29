"""Produce the supervisor demonstration bundle.

    python scripts/demo.py --out flair_head_demo --steps 12

Sweeps all 7 attributes over every (block, head) unit at one contrastive
pair each -- 7 x (1 + blocks x heads) generations, roughly one fifth the
full campaign. Writes every image plus an index.html that needs no Python
to read. Zip the output directory and hand it over.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy
import torch
from diffusers import StableDiffusion3Pipeline

from flair_t2i.attributes import CORE_ATTRIBUTES
from flair_t2i.calibration.corpus import DEFAULT_CORPUS_PATH, load_corpus
from flair_t2i.calibration.harness import make_swap_generate_fn
from flair_t2i.config import FlairConfig
from flair_t2i.demo.report import write_report
from flair_t2i.demo.sweep import DemoPaths, run_demo_sweep
from flair_t2i.hasm import HASM
from flair_t2i.metrics.embedding import ClipScorer
from flair_t2i.metrics.integrity import IntegrityGate
from flair_t2i.metrics.masking import ClipSegMasker
from flair_t2i.pipeline import FlairPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("flair_head_demo"))
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metric-device",
        default="cuda",
        help="where ClipSeg/CLIP run. ClipSeg on CPU costs 1-3s per cell and "
        "there are thousands of cells; fp16 on GPU it is ~150MB. Use 'cpu' "
        "only if VRAM genuinely will not stretch.",
    )
    parser.add_argument(
        "--pairs",
        "--pair-indices",
        dest="pairs",
        type=str,
        default="1",
        help="Contrastive pairs: count (e.g. '1' for first pair, '5' for all 5) or "
        "comma/range indices (e.g. '1,2' or '1-2' for 2nd and 3rd pairs).",
    )
    parser.add_argument(
        "--attribute",
        "--attributes",
        dest="attributes",
        type=str,
        default=None,
        help="Comma-separated attribute names to run (e.g. 'color' or "
        "'color,size'). Default runs all attributes.",
    )
    parser.add_argument(
        "--blocks",
        "--block-ids",
        dest="blocks",
        type=str,
        default=None,
        help="Range or comma-separated blocks to sweep (e.g. '0-11', '12-23', or '0,1,2'). Default runs all blocks.",
    )
    args = parser.parse_args()

    # Set CUDA memory allocator configuration to prevent fragmentation OOM
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Load .env if present
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    cfg = FlairConfig(device="cuda")
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("hf_token")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        token=token,
    )
    pipe.enable_model_cpu_offload()

    n_blocks = len(pipe.transformer.transformer_blocks)
    n_heads = pipe.transformer.transformer_blocks[0].attn.heads

    if args.blocks:
        if "-" in args.blocks:
            start, end = map(int, args.blocks.split("-", 1))
            block_ids = tuple(range(start, end + 1))
        else:
            block_ids = tuple(int(b.strip()) for b in args.blocks.split(",") if b.strip())
    else:
        block_ids = tuple(range(n_blocks))

    head_ids = tuple(range(n_heads))

    fp = FlairPipeline(
        pipe,
        cfg,
        HASM.uniform((0,), (0,), CORE_ATTRIBUTES),
        nlp=spacy.load("en_core_web_sm"),
    )

    from flair_t2i.attributes import AttributeClass

    raw_corpus = load_corpus(DEFAULT_CORPUS_PATH)
    if args.attributes:
        requested = [AttributeClass(a.strip()) for a in args.attributes.split(",") if a.strip()]
        raw_corpus = {attr: pairs for attr, pairs in raw_corpus.items() if attr in requested}
        if not raw_corpus:
            raise ValueError(f"no matching attributes found for {args.attributes!r}")

    def _select_pairs(pair_list, spec: str):
        spec = spec.strip()
        if "," in spec:
            indices = [int(x.strip()) for x in spec.split(",") if x.strip()]
        elif "-" in spec:
            start, end = map(int, spec.split("-", 1))
            indices = list(range(start, end + 1))
        else:
            n = int(spec)
            indices = list(range(min(n, len(pair_list))))
        return [pair_list[i] for i in indices if 0 <= i < len(pair_list)]

    corpus = {
        attr: _select_pairs(pairs, args.pairs)
        for attr, pairs in raw_corpus.items()
    }
    total_pairs = sum(len(p) for p in corpus.values())
    total = sum(len(p) * (1 + len(block_ids) * len(head_ids)) for p in corpus.values())
    print(f"{len(block_ids)} blocks {block_ids} x {len(head_ids)} heads = {len(block_ids) * len(head_ids)} units")
    print(f"{len(corpus)} attribute(s) {[a.value for a in corpus]} -- {total_pairs} pair(s) total ({total} generations):")
    for attr, p_list in corpus.items():
        for idx, p in enumerate(p_list):
            print(f"  [{attr.value} #{idx}] '{p.base}' -> '{p.changed}'")
    print()

    hasm = run_demo_sweep(
        make_swap_generate_fn(fp, steps=args.steps),
        corpus=corpus,
        block_ids=block_ids,
        head_ids=head_ids,
        masker=ClipSegMasker(device=args.metric_device),
        paths=DemoPaths(args.out),
        seeds=[args.seed],
        gate=IntegrityGate(),
        scorer=ClipScorer(device=args.metric_device),
        progress=lambda attr, unit, value: print(
            f"  {attr.value:<9} B{unit.block:<3}H{unit.head:<3} raw={value:.4f}"
        ),
    )

    report = write_report(hasm, DemoPaths(args.out), title="FLAIR — head-level routing")
    print(f"\nwrote {report}")
    for attr in hasm.attributes:
        unit, score = hasm.top_k(attr, 1)[0]
        print(f"  {attr.value:<9} peaks at B{unit.block} H{unit.head}  ({score:.3f})")


if __name__ == "__main__":
    main()
