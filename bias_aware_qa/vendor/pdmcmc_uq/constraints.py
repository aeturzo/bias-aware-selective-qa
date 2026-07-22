from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .state import UQState


@dataclass
class Constraint:
    name: str
    epsilon: float
    lambda_value: float = 0.0

    def score(self, state: UQState, context: dict[str, Any] | None = None) -> float:
        raise NotImplementedError


def _group_means(state: UQState, ref_group: str, target_group: str) -> tuple[np.ndarray, np.ndarray]:
    ref_mask = state.groups == ref_group
    target_mask = state.groups == target_group
    if not ref_mask.any() or not target_mask.any():
        raise ValueError("Both reference and target groups are required.")
    return state.X_complete[ref_mask].mean(axis=0), state.X_complete[target_mask].mean(axis=0)


@dataclass
class FairnessGapConstraint(Constraint):
    feature_cols: list[str] = field(default_factory=list)
    target_gap: np.ndarray | dict[str, float] | None = None
    ref_group: str = "EU"
    target_group: str = "GS"
    normalize: bool = False

    def __init__(
        self,
        feature_cols: list[str],
        epsilon: float,
        lambda_value: float = 0.0,
        target_gap: np.ndarray | dict[str, float] | None = None,
        ref_group: str = "EU",
        target_group: str = "GS",
        normalize: bool = False,
        name: str = "fairness_gap",
    ) -> None:
        super().__init__(name=name, epsilon=epsilon, lambda_value=lambda_value)
        self.feature_cols = list(feature_cols)
        self.target_gap = target_gap
        self.ref_group = ref_group
        self.target_group = target_group
        self.normalize = bool(normalize)

    def _target_vector(self, state: UQState) -> np.ndarray:
        if self.target_gap is None:
            return np.zeros(len(self.feature_cols), dtype=float)
        if isinstance(self.target_gap, dict):
            return np.array([float(self.target_gap[c]) for c in self.feature_cols], dtype=float)
        return np.asarray(self.target_gap, dtype=float)

    def score(self, state: UQState, context: dict[str, Any] | None = None) -> float:
        ref_mean, target_mean = _group_means(state, self.ref_group, self.target_group)
        col_pos = [state.columns.index(c) for c in self.feature_cols]
        gap = target_mean[col_pos] - ref_mean[col_pos]
        diff = gap - self._target_vector(state)
        if self.normalize:
            std = state.X_complete[:, col_pos].std(axis=0, ddof=1)
            std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
            diff = diff / std
        return float(np.linalg.norm(diff, ord=2))


@dataclass
class BiasPenaltyConstraint(Constraint):
    feature_cols: list[str] = field(default_factory=list)
    ref_group: str = "EU"
    target_group: str = "GS"
    target_gap: np.ndarray | dict[str, float] | None = None

    def __init__(
        self,
        feature_cols: list[str],
        epsilon: float,
        lambda_value: float = 0.0,
        ref_group: str = "EU",
        target_group: str = "GS",
        target_gap: np.ndarray | dict[str, float] | None = None,
        name: str = "bias_penalty",
    ) -> None:
        super().__init__(name=name, epsilon=epsilon, lambda_value=lambda_value)
        self.feature_cols = list(feature_cols)
        self.ref_group = ref_group
        self.target_group = target_group
        self.target_gap = target_gap

    def _target_vector(self) -> np.ndarray:
        if self.target_gap is None:
            return np.zeros(len(self.feature_cols), dtype=float)
        if isinstance(self.target_gap, dict):
            return np.array([float(self.target_gap.get(c, 0.0)) for c in self.feature_cols], dtype=float)
        return np.asarray(self.target_gap, dtype=float)

    def score(self, state: UQState, context: dict[str, Any] | None = None) -> float:
        ref_mask = state.groups == self.ref_group
        target_mask = state.groups == self.target_group
        if not ref_mask.any() or not target_mask.any():
            raise ValueError("Both reference and target groups are required.")

        target_gaps = self._target_vector()
        penalties: list[float] = []
        for idx, col in enumerate(self.feature_cols):
            j = state.columns.index(col)
            ref_vals = state.X_complete[ref_mask, j]
            target_vals = state.X_complete[target_mask, j]
            ref_mean = float(ref_vals.mean())
            target_mean = float(target_vals.mean())
            gap = target_mean - ref_mean
            allowed_gap = float(target_gaps[idx])
            std = float(state.X_complete[:, j].std(ddof=1))
            if not math.isfinite(std) or std <= 1e-6:
                std = 1.0

            lower_col = col.lower()
            adverse_high = (
                "carbon",
                "footprint",
                "emission",
                "salt",
                "sugar",
                "glyco",
                "glucose",
                "a1c",
                "total_cholesterol",
                "cholesterol_total",
                "bmi",
                "systolic",
                "diastolic",
                "blood_pressure",
            )
            protective_high = ("hdl",)
            if any(token in lower_col for token in protective_high):
                diff = max(0.0, allowed_gap - gap)
            elif any(token in lower_col for token in adverse_high):
                diff = max(0.0, gap - allowed_gap)
            else:
                diff = max(0.0, allowed_gap - gap)
            penalties.append(diff / std)

        if not penalties:
            return 0.0
        return float(np.mean(penalties))


@dataclass
class CoverageConstraint(Constraint):
    def __init__(self, epsilon: float = 0.0, lambda_value: float = 0.0, name: str = "coverage") -> None:
        super().__init__(name=name, epsilon=epsilon, lambda_value=lambda_value)

    def score(self, state: UQState, context: dict[str, Any] | None = None) -> float:
        return 0.0


@dataclass
class DistributionConstraint(Constraint):
    def __init__(self, epsilon: float = 0.0, lambda_value: float = 0.0, name: str = "distribution") -> None:
        super().__init__(name=name, epsilon=epsilon, lambda_value=lambda_value)

    def score(self, state: UQState, context: dict[str, Any] | None = None) -> float:
        return 0.0
