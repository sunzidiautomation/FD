"""Top-level FLAIR pipeline (spec sections 3.2-3.6).

Component streams are text-encoded once here, before denoising, and handed
to the RoutingPlan. The denoising batch keeps its usual
``[negative, positive]`` shape.
"""

from __future__ import annotations

import torch

from .basm import BASM
from .components import Component
from .config import FlairConfig
from .guard import CoherenceGuard
from .parsing import parse_prompt
from .patching import install_flair, uninstall_flair
from .processor import PlanRef
from .routing import RoutingPlan, build_routing_plan


class FlairPipeline:
    def __init__(self, pipe, cfg: FlairConfig, basm: BASM, nlp=None) -> None:
        self.pipe = pipe
        self.cfg = cfg
        self.basm = basm
        self.nlp = nlp
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
    ):
        self.last_plan = None
        self.last_guard = None

        ref = PlanRef(total_steps=steps, do_cfg=guidance_scale > 1.0)

        if routing:
            components = parse_prompt(prompt, self.nlp)
            routable = [c for c in components if c.attr in self.basm.attributes]
            embeddings = self.encode_components(routable)
            plan = build_routing_plan(routable, embeddings, self.basm, self.cfg)
            if plan.routed:
                ref.plan = plan
                self.last_plan = plan
                self.last_guard = CoherenceGuard(self.cfg)
                self.last_guard.apply(plan, self.last_guard.check_streams(plan, step=0))

        handles = install_flair(self.pipe.transformer, ref)
        try:

            def on_step(pipe, step_index, timestep, callback_kwargs):
                ref.step = step_index
                return callback_kwargs

            result = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
        finally:
            uninstall_flair(handles)

        return result.images[0]
