"""Timestep weighting for injection (spec section 3.5).

Text influence is strongest early in denoising and fades to nothing by the
end of the window, following the schedule shape HeadRouter reports for
MM-DiT editing.
"""


def timestep_scale(step_frac: float, t_window: tuple[float, float]) -> float:
    """Return the injection scale in [0, 1] at ``step_frac`` of denoising."""
    start, end = t_window
    if step_frac < start or step_frac > end:
        return 0.0
    if end <= start:
        return 1.0
    return 1.0 - (step_frac - start) / (end - start)
