"""Runtime configuration. Defaults follow spec sections 3.3-3.6."""

from dataclasses import dataclass


@dataclass
class FlairConfig:
    device: str = "cuda"

    # Injection (spec section 3.5)
    alpha_0: float = 0.75
    t_window: tuple[float, float] = (0.0, 0.6)
    top_k_default: int = 1

    # Coherence guard (spec section 3.6)
    guard_cos_threshold: float = 0.55
    guard_membership_threshold: float = 0.5
    guard_backoff: float = 0.5

    # Backbone
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    max_sequence_length: int = 256


DEFAULT_CONFIG = FlairConfig()
