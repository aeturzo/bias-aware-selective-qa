from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class UQState:
    """A completed table state for imputation-based UQ.

    `observed_mask` is True for entries that came from the observed data and
    must remain fixed under proposals.
    """

    X_complete: np.ndarray
    observed_mask: np.ndarray
    groups: np.ndarray
    columns: list[str]
    bias_params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.X_complete = np.asarray(self.X_complete, dtype=float)
        self.observed_mask = np.asarray(self.observed_mask, dtype=bool)
        self.groups = np.asarray(self.groups)
        if self.X_complete.shape != self.observed_mask.shape:
            raise ValueError("X_complete and observed_mask must have the same shape.")
        if self.X_complete.shape[0] != self.groups.shape[0]:
            raise ValueError("groups length must match number of rows.")
        if self.X_complete.shape[1] != len(self.columns):
            raise ValueError("columns length must match number of features.")

    def copy(self) -> "UQState":
        return UQState(
            X_complete=self.X_complete.copy(),
            observed_mask=self.observed_mask.copy(),
            groups=self.groups.copy(),
            columns=list(self.columns),
            bias_params=dict(self.bias_params) if self.bias_params is not None else None,
        )

    def to_dataframe(self, group_col: str = "region") -> pd.DataFrame:
        df = pd.DataFrame(self.X_complete, columns=self.columns)
        df.insert(0, group_col, self.groups)
        return df

    def validate_observed_fixed(self, original_observed_values: Any) -> bool:
        """Return True if all observed entries still match the original table.

        `original_observed_values` may contain NaN in missing locations.
        """

        original = np.asarray(original_observed_values, dtype=float)
        if original.shape != self.X_complete.shape:
            raise ValueError("original_observed_values shape does not match state.")
        if not np.allclose(
            self.X_complete[self.observed_mask],
            original[self.observed_mask],
            rtol=1e-10,
            atol=1e-10,
            equal_nan=False,
        ):
            return False
        return True
