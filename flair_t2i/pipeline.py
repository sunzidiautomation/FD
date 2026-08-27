"""Top-level FLAIR pipeline (spec sections 3.2-3.6).

Component streams are text-encoded once here, before denoising, and handed
to the RoutingPlan. The denoising batch keeps its usual
``[negative, positive]`` shape.
"""

from __future__ import annotations

import torch

from .components import Component
from .config import FlairConfig
from .fuzzy.resolve import resolve_components
from .guard import CoherenceGuard
from .hasm import HASM
from .latents import LatentRecorder
from .parsing import parse_prompt
from .patching import install_head_routing, uninstall_head_routing
from .processor import PlanRef
from .routing import RoutingPlan, build_routing_plan


class FlairPipeline:
    def __init__(
        self,
        pipe,
        cfg: FlairConfig,
        hasm: HASM,
        nlp=None,
        granularity: str = "head",
    ) -> None:
        self.pipe = pipe
        self.cfg = cfg
        self.hasm = hasm
        self.nlp = nlp
        self.granularity = granularity
        self.last_plan: RoutingPlan | None = None
        self.last_guard: CoherenceGuard | None = None
        #: text -> embedding, already projected into transformer space. The
        #: encoders are frozen, so a phrase always encodes the same thing.
        #: A sweep runs thousands of generations over a handful of distinct
        #: phrases, and under cpu_offload every encode drags T5-XXL (~9.5GB)
        #: onto the GPU and back -- so this is the difference between
        #: encoding 7 times and encoding 4,000 times.
        self._embeddings: dict[str, torch.Tensor] = {}

    def clear_embedding_cache(self) -> None:
        """Drop cached text embeddings. Only needed if the encoders change."""
        self._embeddings.clear()

    @torch.inference_mode()
    def encode_components(self, components: list[Component]) -> dict[str, torch.Tensor]:
        """Encode every component's text once, batched, and cache by text."""
        if not components:
            return {}

        # dict.fromkeys keeps first-seen order while dropping repeats, so a
        # phrase shared by two components is encoded once, not twice.
        wanted = dict.fromkeys(c.text for c in components)
        missing = [t for t in wanted if t not in self._embeddings]

        if missing:
            prompt_embeds, _, _, _ = self.pipe.encode_prompt(
                prompt=missing,
                prompt_2=missing,
                prompt_3=missing,
                do_classifier_free_guidance=False,
                max_sequence_length=self.cfg.max_sequence_length,
            )
            if hasattr(self.pipe, "transformer") and hasattr(
                self.pipe.transformer, "context_embedder"
            ):
                # encoder_hidden_states inside the blocks is post-embedder,
                # so components must land in that space to be comparable.
                embedder = self.pipe.transformer.context_embedder
                prompt_embeds = embedder(
                    prompt_embeds.to(
                        device=embedder.weight.device, dtype=embedder.weight.dtype
                    )
                )

            for i, text in enumerate(missing):
                self._embeddings[text] = prompt_embeds[i]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {c.id: self._embeddings[c.text] for c in components}

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int = 0,
        steps: int = 20,
        guidance_scale: float = 4.5,
        routing: bool = True,
        fuzzy: bool = True,
        recorder: LatentRecorder | None = None,
    ):
        self.last_plan = None
        self.last_guard = None

        ref = PlanRef(total_steps=steps, do_cfg=guidance_scale > 1.0)

        if routing:
            components = parse_prompt(prompt, self.nlp)
            routable = [c for c in components if c.attr in self.hasm.attributes]
            embeddings = self.encode_components(routable)

            if fuzzy:
                intensities, k_overrides, _ = resolve_components(routable)
            else:
                intensities, k_overrides = {}, {}

            plan = build_routing_plan(
                routable,
                embeddings,
                self.hasm,
                self.cfg,
                intensities,
                k_overrides,
                granularity=self.granularity,
            )
            if plan.routed:
                ref.plan = plan
                self.last_plan = plan
                self.last_guard = CoherenceGuard(self.cfg)
                self.last_guard.apply(plan, self.last_guard.check_streams(plan, step=0))

        handles = install_head_routing(self.pipe.transformer, ref)
        try:

            def on_step(pipe, step_index, timestep, callback_kwargs):
                ref.step = step_index
                if recorder is not None and "latents" in callback_kwargs:
                    recorder(step_index, steps, callback_kwargs["latents"])
                return callback_kwargs

            result = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=self.cfg.height,
                width=self.cfg.width,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
        finally:
            uninstall_head_routing(handles)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result.images[0]
