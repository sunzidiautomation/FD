"""Parsed prompt components and the explicit text-encoding batch layout.

Spec section 3.5 flags an implicit batch-layout assumption
(``base_i = B - n_rows``) as the single most dangerous implementation
detail in FLAIR. This module removes that arithmetic entirely: row 0 is
always the base prompt, component rows are named, and every access is
validated.

Note that this layout describes the TEXT-ENCODING batch only. Component
streams are never stacked into the denoising batch -- see
``flair_t2i.routing``.
"""

from dataclasses import dataclass

from .attributes import AttributeClass


@dataclass(frozen=True)
class Component:
    """One attribute component extracted from a prompt."""

    id: str
    text: str
    attr: AttributeClass
    hedge: str | None = None


@dataclass(frozen=True)
class TextBatchLayout:
    """Maps text-encoder batch rows to streams. Row 0 is always the base."""

    component_ids: tuple[str, ...]

    BASE_ROW: int = 0

    def __post_init__(self) -> None:
        if len(set(self.component_ids)) != len(self.component_ids):
            raise ValueError(f"duplicate component ids: {self.component_ids}")

    def n_rows(self) -> int:
        return 1 + len(self.component_ids)

    def row_for(self, component_id: str) -> int:
        try:
            return 1 + self.component_ids.index(component_id)
        except ValueError:
            raise KeyError(
                f"{component_id} is not in this layout: {self.component_ids}"
            ) from None

    def validate(self, batch_size: int) -> None:
        if batch_size != self.n_rows():
            raise ValueError(
                f"expected {self.n_rows()} rows "
                f"(base + {len(self.component_ids)} components), got {batch_size}"
            )
