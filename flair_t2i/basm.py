"""The Block-Attribute Sensitivity Matrix (spec section 3.4).

Calibration is offline and runs once per backbone; this module is only the
container and query API. The calibration campaign that fills it in has its
own plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .attributes import AttributeClass


class BASM:
    def __init__(
        self,
        matrix: np.ndarray,
        block_ids: tuple[int, ...],
        attributes: tuple[AttributeClass, ...],
    ) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        expected = (len(block_ids), len(attributes))
        if matrix.shape != expected:
            raise ValueError(f"matrix shape {matrix.shape} does not match {expected}")
        if matrix.size and (matrix.min() < 0.0 or matrix.max() > 1.0):
            raise ValueError("sensitivity scores must be within [0, 1]")

        self.matrix = matrix
        self.block_ids = tuple(block_ids)
        self.attributes = tuple(attributes)
        self._block_index = {b: i for i, b in enumerate(self.block_ids)}
        self._attr_index = {a: i for i, a in enumerate(self.attributes)}

    @classmethod
    def uniform(
        cls, block_ids: tuple[int, ...], attributes: tuple[AttributeClass, ...]
    ) -> "BASM":
        """An uncalibrated matrix, for tests and pre-calibration smoke runs."""
        return cls(
            np.full((len(block_ids), len(attributes)), 0.5), block_ids, attributes
        )

    def _col(self, attr: AttributeClass) -> int:
        if attr not in self._attr_index:
            raise KeyError(f"{attr.value} is not calibrated in this BASM")
        return self._attr_index[attr]

    def score(self, block_id: int, attr: AttributeClass) -> float:
        if block_id not in self._block_index:
            raise KeyError(f"block {block_id} is not in this BASM")
        return float(self.matrix[self._block_index[block_id], self._col(attr)])

    def top_k(self, attr: AttributeClass, k: int) -> list[tuple[int, float]]:
        col = self.matrix[:, self._col(attr)]
        ranked = sorted(zip(self.block_ids, col), key=lambda pair: (-pair[1], pair[0]))
        return [(int(b), float(s)) for b, s in ranked[: max(0, k)]]

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            matrix=self.matrix,
            block_ids=np.array(self.block_ids),
            attributes=np.array([a.value for a in self.attributes]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BASM":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            matrix=data["matrix"],
            block_ids=tuple(int(b) for b in data["block_ids"]),
            attributes=tuple(AttributeClass(a) for a in data["attributes"]),
        )
