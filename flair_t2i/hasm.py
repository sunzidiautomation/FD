"""The Head-Attribute Sensitivity Matrix.

A ``[blocks x heads x attributes]`` tensor, calibrated by the same causal
contrastive-swap procedure that produced the BASM. Reducing over the head
axis yields an ordinary BASM at no additional measurement cost, which is
what keeps block-level routing available as a derived special case rather
than a second calibration campaign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .attributes import AttributeClass
from .basm import BASM
from .heads import HeadUnit


class HASM:
    def __init__(
        self,
        tensor: np.ndarray,
        block_ids: tuple[int, ...],
        head_ids: tuple[int, ...],
        attributes: tuple[AttributeClass, ...],
    ) -> None:
        tensor = np.asarray(tensor, dtype=np.float64)
        expected = (len(block_ids), len(head_ids), len(attributes))
        if tensor.shape != expected:
            raise ValueError(f"tensor shape {tensor.shape} does not match {expected}")
        if tensor.size and (tensor.min() < 0.0 or tensor.max() > 1.0):
            raise ValueError("sensitivity scores must be within [0, 1]")

        self.tensor = tensor
        self.block_ids = tuple(block_ids)
        self.head_ids = tuple(head_ids)
        self.attributes = tuple(attributes)
        self._block_index = {b: i for i, b in enumerate(self.block_ids)}
        self._head_index = {h: i for i, h in enumerate(self.head_ids)}
        self._attr_index = {a: i for i, a in enumerate(self.attributes)}

    @classmethod
    def uniform(
        cls,
        block_ids: tuple[int, ...],
        head_ids: tuple[int, ...],
        attributes: tuple[AttributeClass, ...],
    ) -> "HASM":
        """An uncalibrated tensor, for tests and pre-calibration smoke runs."""
        shape = (len(block_ids), len(head_ids), len(attributes))
        return cls(np.full(shape, 0.5), block_ids, head_ids, attributes)

    def _plane(self, attr: AttributeClass) -> int:
        if attr not in self._attr_index:
            raise KeyError(f"{attr.value} is not calibrated in this HASM")
        return self._attr_index[attr]

    def score(self, unit: HeadUnit, attr: AttributeClass) -> float:
        if unit.block not in self._block_index:
            raise KeyError(f"block {unit.block} is not in this HASM")
        if unit.head not in self._head_index:
            raise KeyError(f"head {unit.head} is not in this HASM")
        return float(
            self.tensor[
                self._block_index[unit.block],
                self._head_index[unit.head],
                self._plane(attr),
            ]
        )

    def top_k(self, attr: AttributeClass, k: int) -> list[tuple[HeadUnit, float]]:
        plane = self.tensor[:, :, self._plane(attr)]
        ranked = sorted(
            (
                (HeadUnit(block=b, head=h), float(plane[i, j]))
                for i, b in enumerate(self.block_ids)
                for j, h in enumerate(self.head_ids)
            ),
            key=lambda pair: (-pair[1], pair[0].block, pair[0].head),
        )
        return ranked[: max(0, k)]

    def to_basm(self, reduce: str = "max") -> BASM:
        """Collapse the head axis into an ordinary block-level BASM."""
        if reduce == "max":
            matrix = self.tensor.max(axis=1)
        elif reduce == "mean":
            matrix = self.tensor.mean(axis=1)
        else:
            raise ValueError(f"unknown reduction {reduce!r}; use 'max' or 'mean'")
        return BASM(
            matrix=matrix, block_ids=self.block_ids, attributes=self.attributes
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            tensor=self.tensor,
            block_ids=np.array(self.block_ids),
            head_ids=np.array(self.head_ids),
            attributes=np.array([a.value for a in self.attributes]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HASM":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            tensor=data["tensor"],
            block_ids=tuple(int(b) for b in data["block_ids"]),
            head_ids=tuple(int(h) for h in data["head_ids"]),
            attributes=tuple(AttributeClass(a) for a in data["attributes"]),
        )

    @classmethod
    def merge(cls, hasms: list["HASM"]) -> "HASM":
        """Merge multiple single-attribute or partial HASMs into one combined HASM."""
        if not hasms:
            raise ValueError("cannot merge empty HASM list")
        block_ids = hasms[0].block_ids
        head_ids = hasms[0].head_ids

        # Collect unique attributes across all input HASMs
        attributes: list[AttributeClass] = []
        for h in hasms:
            if h.block_ids != block_ids or h.head_ids != head_ids:
                raise ValueError(
                    "all HASMs to merge must have identical block_ids and head_ids"
                )
            for a in h.attributes:
                if a not in attributes:
                    attributes.append(a)

        tensor = np.zeros(
            (len(block_ids), len(head_ids), len(attributes)), dtype=np.float64
        )
        for a_idx, attr in enumerate(attributes):
            for h in hasms:
                if attr in h.attributes:
                    plane = h.tensor[:, :, h._plane(attr)]
                    tensor[:, :, a_idx] = plane
                    break

        return cls(tensor, block_ids, head_ids, tuple(attributes))

