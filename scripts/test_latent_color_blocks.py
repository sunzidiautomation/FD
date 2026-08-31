"""Direct Latent Space Color Sensitivity Test on Block 0 (B0) vs Block 1 (B1).

Measures color change directly in Latent Space (no VAE needed during scoring).
Computes:
  1. Latent L2 Frobenius Distance
  2. Latent Cosine Distance
  3. Latent Gram Matrix Shift (Channel Covariance)
  4. Combined Latent Color Score

Runs on Kaggle GPU:
    python scripts/test_latent_color_blocks.py --steps 12 --out outputs/latent_b0_b1_test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from PIL import Image, ImageDraw, ImageFont

from flair_t2i.attributes import AttributeClass, CORE_ATTRIBUTES
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit
from flair_t2i.patching import install_head_routing, uninstall_head_routing
from flair_t2i.pipeline import FlairPipeline
from flair_t2i.processor import PlanRef
from flair_t2i.metrics.latent_color import pure_latent_color_score, latent_spatial_gradient


def compute_latent_metrics(z_base: torch.Tensor, z_swapped: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    """Compute 100% pure Latent Space color metrics without ANY VAE decoding."""
    # 1. Point-to-point Frobenius L2 distance
    diff = z_swapped - z_base
    l2_dist = torch.linalg.norm(diff).item()
    rel_l2 = (torch.linalg.norm(diff) / (torch.linalg.norm(z_base) + 1e-8)).item()

    # 2. Latent Cosine Distance
    v_base = z_base.flatten(start_dim=1)
    v_swap = z_swapped.flatten(start_dim=1)
    cos_sim = F.cosine_similarity(v_base, v_swap).item()
    cos_dist = 1.0 - cos_sim

    # 3. Disentangled Pure Color Score (Gram Shift x Structural Preservation)
    disentangled = pure_latent_color_score(z_base, z_swapped, mask=mask)

    return {
        "l2_distance": l2_dist,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_sim,
        "cosine_distance": cos_dist,
        "gram_matrix_shift": disentangled["gram_shift"],
        "color_shift": disentangled["color_shift"],
        "structure_preservation": disentangled["structure_preservation"],
        "combined_color_score": disentangled["pure_color_score"],
    }


def latent_to_pca_rgb(latent: torch.Tensor) -> Image.Image:
    """Project a 16-channel latent onto 3 RGB channels using SVD/PCA."""
    feat = latent.squeeze(0).float().cpu()
    c, h, w = feat.shape
    flat = feat.view(c, -1).T
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(flat_centered, full_matrices=False)
    proj = flat_centered @ vh[:3, :].T
    p_min = proj.min(dim=0, keepdim=True)[0]
    p_max = proj.max(dim=0, keepdim=True)[0]
    scaled = (proj - p_min) / (p_max - p_min + 1e-8) * 255.0
    rgb_arr = scaled.view(h, w, 3).numpy().astype(np.uint8)
    return Image.fromarray(rgb_arr).resize((256, 256), resample=Image.Resampling.NEAREST)


def latent_diff_heatmap(z_base: torch.Tensor, z_swapped: torch.Tensor) -> Image.Image:
    """Render spatial difference heatmap between base and swapped latents."""
    diff = (z_swapped - z_base).squeeze(0).float().cpu()  # [16, H, W]
    diff_norm = torch.linalg.norm(diff, dim=0)             # [H, W]
    d_min, d_max = diff_norm.min(), diff_norm.max()
    scaled = ((diff_norm - d_min) / (d_max - d_min + 1e-8) * 255.0).numpy().astype(np.uint8)
    
    # Yellow/Red Fire colormap
    rgb = np.zeros((scaled.shape[0], scaled.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = scaled                          # Red
    rgb[..., 1] = (scaled * 0.6).astype(np.uint8) # Green
    rgb[..., 2] = (scaled * 0.1).astype(np.uint8) # Blue
    return Image.fromarray(rgb).resize((256, 256), resample=Image.Resampling.NEAREST)


def decode_latent_to_pil(pipe: StableDiffusion3Pipeline, latent: torch.Tensor) -> Image.Image:
    """Decode a latent tensor into RGB PIL image (for visual confirmation)."""
    with torch.no_grad():
        shift = getattr(pipe.vae.config, "shift_factor", 0.0609)
        scale = getattr(pipe.vae.config, "scaling_factor", 1.5035)
        unscaled = (latent.to(pipe.vae.dtype) / scale) + shift
        decoded = pipe.vae.decode(unscaled, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        return image.resize((256, 256), resample=Image.Resampling.LANCZOS)


def run_single_swap_latent(
    pipe: StableDiffusion3Pipeline,
    fp: FlairPipeline,
    base_prompt: str,
    swap_prompt: str,
    block_id: int,
    head_id: int,
    steps: int,
    seed: int,
    cfg: FlairConfig,
) -> torch.Tensor:
    """Run generation with swap and return ONLY the final latent tensor (no VAE decode)."""
    comp = Component(id="swap", text=swap_prompt, attr=AttributeClass.COLOR)
    embs = fp.encode_components([comp])
    unit = HeadUnit(block=block_id, head=head_id)
    plan = RoutingPlan(
        routed=(RoutedComponent(component=comp, embedding=embs["swap"], units=((unit, 1.0),)),),
        cfg=cfg,
    )
    ref = PlanRef(plan=plan, total_steps=steps, do_cfg=True)
    handles = install_head_routing(pipe.transformer, ref)

    final_latent = None
    try:
        def on_step(p, step_index, timestep, callback_kwargs):
            nonlocal final_latent
            ref.step = step_index
            if step_index == steps - 1:
                final_latent = callback_kwargs["latents"].detach().clone()
            return callback_kwargs

        pipe(
            prompt=base_prompt,
            num_inference_steps=steps,
            guidance_scale=4.5,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            callback_on_step_end=on_step,
        )
    finally:
        uninstall_head_routing(handles)

    return final_latent


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Direct Latent Color Sensitivity across Blocks on a chosen Head")
    parser.add_argument("--base-prompt", default="a red sports car on a road")
    parser.add_argument("--swap-prompt", default="a blue sports car on a road")
    parser.add_argument("--head", type=int, default=0, help="Attention Head index to test (default: 0)")
    parser.add_argument(
        "--blocks",
        default="0,1",
        help="Blocks to test: '0,1', '0-23', 'all', or comma-separated list like '0,5,10,18'. (Default: '0,1')",
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/latent_blocks_test"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Load .env token
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    cfg = FlairConfig(device="cuda")
    token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

    print("[1/3] Loading Stable Diffusion 3.5...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        token=token,
    )
    pipe.enable_model_cpu_offload()
    fp = FlairPipeline(pipe, cfg, HASM.uniform((0,), (0,), CORE_ATTRIBUTES), nlp=spacy.load("en_core_web_sm"))

    n_total_blocks = len(pipe.transformer.transformer_blocks)

    # Parse blocks argument
    blocks_str = str(args.blocks).strip().lower()
    if blocks_str in ("all", "0-23", "24"):
        block_ids = list(range(n_total_blocks))
    elif "-" in blocks_str:
        start, end = map(int, blocks_str.split("-", 1))
        block_ids = list(range(start, end + 1))
    else:
        block_ids = [int(b.strip()) for b in blocks_str.split(",") if b.strip()]

    print("=" * 75)
    print("DIRECT LATENT SPACE COLOR SENSITIVITY SWEEP")
    print(f"Base Prompt:   '{args.base_prompt}'")
    print(f"Swap Prompt:   '{args.swap_prompt}'")
    print(f"Testing Blocks: {block_ids} (Total: {len(block_ids)}) on Head {args.head}")
    print(f"Steps:         {args.steps} | Seed: {args.seed}")
    print("=" * 75)

    # --- Run 1: Base Latent Generation ---
    print("\n[2/3] Generating Base Latent (no swap)...")
    base_latent = None
    def on_step_base(p, step_index, timestep, callback_kwargs):
        nonlocal base_latent
        if step_index == args.steps - 1:
            base_latent = callback_kwargs["latents"].detach().clone()
        return callback_kwargs

    pipe(
        prompt=args.base_prompt,
        num_inference_steps=args.steps,
        guidance_scale=4.5,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        callback_on_step_end=on_step_base,
    )

    # --- Run 2: Sweeping Selected Blocks on the specified Head ---
    print(f"\n[3/3] Sweeping {len(block_ids)} Blocks on Head {args.head} (Pure Latents, No VAE)...")
    block_results: dict[int, dict] = {}
    swapped_latents: dict[int, torch.Tensor] = {}

    for idx, b_id in enumerate(block_ids):
        t0 = time.perf_counter()
        z_swap = run_single_swap_latent(
            pipe, fp, args.base_prompt, args.swap_prompt,
            block_id=b_id, head_id=args.head, steps=args.steps, seed=args.seed, cfg=cfg,
        )
        elapsed = time.perf_counter() - t0
        swapped_latents[b_id] = z_swap

        metrics = compute_latent_metrics(base_latent, z_swap)
        metrics["elapsed_time"] = elapsed
        block_results[b_id] = metrics

        print(f"  [{idx+1:02d}/{len(block_ids):02d}] Block {b_id:02d} Head {args.head} -> "
              f"L2 Diff: {metrics['l2_distance']:.4f} | Cos Dist: {metrics['cosine_distance']:.5f} | "
              f"Gram: {metrics['gram_matrix_shift']:.4f} | Score: {metrics['combined_color_score']:.4f} ({elapsed:.1f}s)")

    # --- RANKED LEADERBOARD ---
    print("\n" + "=" * 90)
    print(f"RANKED PURE LATENT COLOR SENSITIVITY LEADERBOARD (Head {args.head}) - NO VAE DECODER")
    print("=" * 90)
    print(f"{'Rank':<5} | {'Block':<8} | {'Pure Color Score':<18} | {'Color Shift':<13} | {'Shape Intact %':<16} | {'Gram Shift':<12}")
    print("-" * 90)

    ranked_blocks = sorted(block_results.items(), key=lambda x: x[1]["combined_color_score"], reverse=True)
    for rank, (b_id, m) in enumerate(ranked_blocks, 1):
        star = " ⭐ (Top Disentangled Peak)" if rank == 1 else ""
        preserv_pct = m["structure_preservation"] * 100.0
        print(f"{rank:<5} | B{b_id:<7} | {m['combined_color_score']:<18.5f} | {m['color_shift']:<13.4f} | {preserv_pct:<15.1f}% | {m['gram_matrix_shift']:<12.4f}{star}")

    top_block = ranked_blocks[0][0]
    top_score = ranked_blocks[0][1]["combined_color_score"]
    print("=" * 90)
    print(f"🏆 BEST COLOR BLOCK ON HEAD {args.head}: Block {top_block} (Pure Color Score: {top_score:.4f})")
    print("=" * 90)

    # --- BUILD VISUAL COMPARISON IMAGE ---
    print("\nBuilding comparative visualization montage...")
    # Select top blocks (or all if <= 6) for montage display
    display_blocks = [b for b, _ in ranked_blocks[:6]]
    
    pca_base = latent_to_pca_rgb(base_latent)
    dec_base = decode_latent_to_pil(pipe, base_latent)

    cell_w, cell_h = 256, 256
    pad = 10
    header_h = 60
    label_h = 28

    n_cols = 1 + len(display_blocks)
    canvas_w = pad + n_cols * (cell_w + pad)
    canvas_h = header_h + 3 * (cell_h + label_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 24))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, 12), f"FLAIR Latent Color Sensitivity Sweep: Top Blocks on Head {args.head}", fill=(240, 240, 250))
    draw.text((pad, 34), f"Base: '{args.base_prompt}'  ->  Swap: '{args.swap_prompt}'", fill=(170, 180, 200))

    # Base column
    draw.text((pad + 4, header_h + 4), "Row 1: Raw Latent PCA", fill=(180, 190, 210))
    draw.text((pad + 4, header_h + cell_h + label_h + pad + 4), "Row 2: Difference Heatmap |Δz|", fill=(180, 190, 210))
    draw.text((pad + 4, header_h + 2 * (cell_h + label_h + pad) + 4), "Row 3: Decoded Image", fill=(180, 190, 210))

    cols = [("Base (No Swap)", pca_base, Image.new("RGB", (cell_w, cell_h), color=(0, 0, 0)), dec_base)]
    for b_id in display_blocks:
        z_s = swapped_latents[b_id]
        score = block_results[b_id]["combined_color_score"]
        pca_s = latent_to_pca_rgb(z_s)
        heat_s = latent_diff_heatmap(base_latent, z_s)
        dec_s = decode_latent_to_pil(pipe, z_s)
        cols.append((f"Block {b_id} (Score: {score:.3f})", pca_s, heat_s, dec_s))

    for c_idx, (col_name, pca_img, heat_img, dec_img) in enumerate(cols):
        x_left = pad + c_idx * (cell_w + pad)
        for r_idx, img in enumerate([pca_img, heat_img, dec_img]):
            y_top = header_h + r_idx * (cell_h + label_h + pad)
            y_img = y_top + label_h
            canvas.paste(img, (x_left, y_img))
            draw.text((x_left + 4, y_img + 4), col_name, fill=(255, 255, 255))

    # Save results as structured JSON
    import json
    json_path = args.out / "latent_results.json"
    clean_results = {
        "base_prompt": args.base_prompt,
        "swap_prompt": args.swap_prompt,
        "head_tested": args.head,
        "steps": args.steps,
        "seed": args.seed,
        "ranked_leaderboard": [
            {
                "rank": rank,
                "block": b_id,
                "pure_color_score": m["combined_color_score"],
                "color_shift_delta_e": m["color_shift"],
                "shape_intact_pct": round(m["structure_preservation"] * 100.0, 2),
                "gram_matrix_shift": m["gram_matrix_shift"],
                "l2_distance": m["l2_distance"],
                "cosine_distance": m["cosine_distance"],
                "time_sec": round(m.get("elapsed_time", 0.0), 2),
            }
            for rank, (b_id, m) in enumerate(ranked_blocks, 1)
        ],
        "top_block": top_block,
        "top_score": top_score,
    }
    json_path.write_text(json.dumps(clean_results, indent=2), encoding="utf-8")
    print(f"Saved structured JSON results to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
