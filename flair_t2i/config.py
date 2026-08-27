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

    # Generation resolution. SD3.5 defaults to 1024x1024, which is roughly
    # 4-6x the cost of 512: the latent grid goes from 32x32 to 64x64 tokens,
    # and attention scales with the square of that. Every budget in the
    # campaign plan assumes 512. Raise it for final paper figures, not for
    # calibration sweeps.
    height: int = 512
    width: int = 512


DEFAULT_CONFIG = FlairConfig()
