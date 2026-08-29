"""Visualize the step-by-step evolution of Latent Space (z_t -> z_0) and VAE decoding.

Runs on Kaggle GPU:
    python scripts/visualize_latents.py --prompt "a red sports car on a road" --steps 20 --out outputs/latent_viz

With Head Swap comparison:
    python scripts/visualize_latents.py \\
        --prompt "a red sports car on a road" \\
        --swap-prompt "a blue sports car on a road" \\
        --block 8 --head 2 \\
        --steps 20 --out outputs/latent_viz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import spacy
import torch
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
from flair_t2i.routing import RoutedComponent, RoutingPlan


def latent_to_pca_rgb(latent: torch.Tensor) -> Image.Image:
    """Project a 16-channel latent [1, 16, H, W] onto 3 RGB channels using SVD/PCA."""
    # [16, H, W] -> [16, H*W] -> [H*W, 16]
    feat = latent.squeeze(0).float().cpu()
    c, h, w = feat.shape
    flat = feat.view(c, -1).T  # [H*W, 16]

    # Center the features
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    # SVD for top 3 principal components
    _, _, vh = torch.linalg.svd(flat_centered, full_matrices=False)
    proj = flat_centered @ vh[:3, :].T  # [H*W, 3]

    # Min-max scale per channel to [0, 255]
    p_min = proj.min(dim=0, keepdim=True)[0]
    p_max = proj.max(dim=0, keepdim=True)[0]
    scaled = (proj - p_min) / (p_max - p_min + 1e-8) * 255.0
    rgb_arr = scaled.view(h, w, 3).numpy().astype(np.uint8)

    img = Image.fromarray(rgb_arr)
    # Resize up with nearest neighbor or bilinear to match display
    return img.resize((256, 256), resample=Image.Resampling.NEAREST)


def decode_latent_to_pil(pipe: StableDiffusion3Pipeline, latent: torch.Tensor) -> Image.Image:
    """Decode a latent tensor [1, 16, H, W] into a full RGB PIL Image."""
    with torch.no_grad():
        # Stable Diffusion 3 scaling & shift factors
        shift = getattr(pipe.vae.config, "shift_factor", 0.0609)
        scale = getattr(pipe.vae.config, "scaling_factor", 1.5035)
        unscaled = (latent.to(pipe.vae.dtype) / scale) + shift
        decoded = pipe.vae.decode(unscaled, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        return image.resize((256, 256), resample=Image.Resampling.LANCZOS)


def make_grid_image(
    steps_records: list[dict],
    title: str,
    has_swap: bool = False,
) -> Image.Image:
    """Build a side-by-side visualization montage across denoising steps."""
    cell_w, cell_h = 256, 256
    pad = 12
    header_h = 50
    label_h = 28

    n_steps = len(steps_records)
    rows = 3 if has_swap else 2

    canvas_w = pad + n_steps * (cell_w + pad)
    canvas_h = header_h + rows * (cell_h + label_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(canvas)

    # Title
    draw.text((pad, 14), title, fill=(240, 240, 245))

    row_labels = [
        "Raw Latent Space (PCA False-Color RGB Projection)",
        "Intermediate VAE Decoded Pixel Image",
    ]
    if has_swap:
        row_labels.append("Head Swap Latent Perturbation |Δz_t| (Difference Heatmap)")

    for r_idx, r_label in enumerate(row_labels):
        y_top = header_h + r_idx * (cell_h + label_h + pad)
        draw.text((pad, y_top + 4), r_label, fill=(180, 185, 200))

        for c_idx, rec in enumerate(steps_records):
            x_left = pad + c_idx * (cell_w + pad)
            y_img = y_top + label_h

            if r_idx == 0:
                img = rec["pca"]
            elif r_idx == 1:
                img = rec["decoded"]
            else:
                img = rec.get("diff_map", Image.new("RGB", (cell_w, cell_h), color=(0, 0, 0)))

            canvas.paste(img, (x_left, y_img))
            step_text = f"Step {rec['step']}/{rec['total_steps']} (t={rec['step_frac']*100:.0f}%)"
            draw.text((x_left + 4, y_img + 4), step_text, fill=(255, 255, 255))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step Latent Space Visualizer")
    parser.add_argument("--prompt", type=str, default="a red sports car on a road")
    parser.add_argument("--swap-prompt", type=str, default=None, help="Optional changed prompt for head swap")
    parser.add_argument("--block", type=int, default=8, help="Transformer block for swap")
    parser.add_argument("--head", type=int, default=0, help="Attention head for swap")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture-every", type=int, default=4, help="Capture every N steps")
    parser.add_argument("--out", type=Path, default=Path("outputs/latent_viz"))
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

    print(f"Loading Stable Diffusion 3.5 ({cfg.model_id})...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        token=token,
    )
    pipe.enable_model_cpu_offload()

    # --- Run 1: Base Generation with Latent Recording ------------------
    base_latents: dict[int, torch.Tensor] = {}
    base_records: list[dict] = []

    print(f"\n[1/2] Denoising Base Prompt: '{args.prompt}' (Seed {args.seed})")

    def on_step_base(pipe, step_index, timestep, callback_kwargs):
        lat = callback_kwargs["latents"].detach().clone()
        if step_index % args.capture_every == 0 or step_index == args.steps - 1:
            base_latents[step_index] = lat
            print(f"  -> Captured Latent at Step {step_index:2d}/{args.steps} | Shape: {list(lat.shape)} | Norm: {lat.norm().item():.2f}")
        return callback_kwargs

    base_result = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=4.5,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        callback_on_step_end=on_step_base,
    )
    base_image = base_result.images[0]
    base_image.save(args.out / "final_base_image.png")

    for step_idx, lat in sorted(base_latents.items()):
        pca_img = latent_to_pca_rgb(lat)
        dec_img = decode_latent_to_pil(pipe, lat)
        base_records.append({
            "step": step_idx,
            "total_steps": args.steps,
            "step_frac": step_idx / args.steps,
            "pca": pca_img,
            "decoded": dec_img,
            "latent": lat,
        })
        pca_img.save(args.out / f"base_latent_pca_step_{step_idx:02d}.png")
        dec_img.save(args.out / f"base_decoded_step_{step_idx:02d}.png")

    montage_base = make_grid_image(
        base_records,
        title=f"FLAIR Latent Space Evolution — Base Prompt: '{args.prompt}'",
        has_swap=False,
    )
    montage_base.save(args.out / "latent_evolution_base.png")
    print(f"Wrote base montage: {args.out / 'latent_evolution_base.png'}")

    # --- Run 2: (Optional) Head Swapped Generation & Difference --------
    if args.swap_prompt:
        print(f"\n[2/2] Denoising with Swap at Block {args.block}, Head {args.head}: '{args.swap_prompt}'")
        fp = FlairPipeline(pipe, cfg, HASM.uniform((0,), (0,), CORE_ATTRIBUTES), nlp=spacy.load("en_core_web_sm"))
        comp = Component(id="swap", text=args.swap_prompt, attr=AttributeClass.IDENTITY)
        embs = fp.encode_components([comp])
        unit = HeadUnit(block=args.block, head=args.head)
        plan = RoutingPlan(
            routed=(RoutedComponent(component=comp, embedding=embs["swap"], units=((unit, 1.0),)),),
            cfg=cfg,
        )
        ref = PlanRef(plan=plan, total_steps=args.steps, do_cfg=True)
        handles = install_head_routing(pipe.transformer, ref)

        swapped_latents: dict[int, torch.Tensor] = {}
        swap_records: list[dict] = []

        try:
            def on_step_swap(p, step_index, timestep, callback_kwargs):
                ref.step = step_index
                lat = callback_kwargs["latents"].detach().clone()
                if step_index % args.capture_every == 0 or step_index == args.steps - 1:
                    swapped_latents[step_index] = lat
                return callback_kwargs

            swap_result = pipe(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=4.5,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                callback_on_step_end=on_step_swap,
            )
            swap_image = swap_result.images[0]
            swap_image.save(args.out / "final_swapped_image.png")
        finally:
            uninstall_head_routing(handles)

        for step_idx, lat_s in sorted(swapped_latents.items()):
            lat_b = base_latents[step_idx]
            pca_img = latent_to_pca_rgb(lat_s)
            dec_img = decode_latent_to_pil(pipe, lat_s)

            # Compute spatial L2 difference norm between latents
            diff = (lat_s - lat_b).squeeze(0).float().cpu()  # [16, H, W]
            diff_norm = torch.linalg.norm(diff, dim=0)  # [H, W]
            d_min, d_max = diff_norm.min(), diff_norm.max()
            diff_scaled = ((diff_norm - d_min) / (d_max - d_min + 1e-8) * 255.0).numpy().astype(np.uint8)
            # Create a yellow/fire heatmap
            diff_rgb = np.zeros((diff_scaled.shape[0], diff_scaled.shape[1], 3), dtype=np.uint8)
            diff_rgb[..., 0] = diff_scaled  # Red
            diff_rgb[..., 1] = (diff_scaled * 0.6).astype(np.uint8)  # Green
            diff_rgb[..., 2] = (diff_scaled * 0.1).astype(np.uint8)  # Blue
            diff_map = Image.fromarray(diff_rgb).resize((256, 256), resample=Image.Resampling.NEAREST)

            swap_records.append({
                "step": step_idx,
                "total_steps": args.steps,
                "step_frac": step_idx / args.steps,
                "pca": pca_img,
                "decoded": dec_img,
                "diff_map": diff_map,
            })

        montage_swap = make_grid_image(
            swap_records,
            title=f"FLAIR Latent Trajectory Swap: B{args.block}H{args.head} '{args.swap_prompt}'",
            has_swap=True,
        )
        montage_swap.save(args.out / "latent_evolution_with_swap.png")
        print(f"Wrote swap comparison montage: {args.out / 'latent_evolution_with_swap.png'}")

    print(f"\nAll visualizations successfully saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
