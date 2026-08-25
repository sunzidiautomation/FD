"""Turn parsed components into per-component injection intensity and breadth."""

from __future__ import annotations

from ..components import Component
from .hedges import HedgeResult, resolve_hedge
from .membership import default_label


def resolve_components(
    components: list[Component],
) -> tuple[dict[str, float], dict[str, int], dict[str, HedgeResult]]:
    """Return (intensities, k_overrides, results) keyed by component id."""
    intensities: dict[str, float] = {}
    k_overrides: dict[str, int] = {}
    results: dict[str, HedgeResult] = {}

    for component in components:
        label = default_label(component.attr)
        result = resolve_hedge(component.attr, label, component.hedge)
        intensities[component.id] = result.intensity
        k_overrides[component.id] = result.k
        results[component.id] = result

    return intensities, k_overrides, results
