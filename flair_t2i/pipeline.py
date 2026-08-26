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

    def encode_components(self, components: list[Component]) -> dict[str, torch.Tensor]:
        """Encode every component's text once, batched."""
        if not components:
            return {}

        texts = [c.text for c in components]
        prompt_embeds, _, _, _ = self.pipe.encode_prompt(
            prompt=texts,
            prompt_2=texts,
            prompt_3=texts,
            do_classifier_free_guidance=False,
            max_sequence_length=self.cfg.max_sequence_length,
        )
        return {c.id: prompt_embeds[i] for i, c in enumerate(components)}

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
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
        finally:
            uninstall_head_routing(handles)

        return result.images[0]
