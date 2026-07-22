from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from .constraints import Constraint


def step_size(rho0: float, round_index: int, decreasing: bool = True) -> float:
    if not decreasing:
        return float(rho0)
    return float(rho0) / math.sqrt(float(round_index) + 1.0)


def update_lambdas(
    constraints: Iterable[Constraint],
    mean_scores: Mapping[str, float],
    round_index: int,
    rho0: float,
    decreasing: bool = True,
) -> dict[str, float]:
    """Apply projected dual ascent/descent for non-negative lambdas."""

    rho = step_size(rho0, round_index, decreasing=decreasing)
    updated: dict[str, float] = {}
    for constraint in constraints:
        score = float(mean_scores.get(constraint.name, 0.0))
        constraint.lambda_value = max(
            0.0,
            float(constraint.lambda_value) + rho * (score - float(constraint.epsilon)),
        )
        updated[constraint.name] = float(constraint.lambda_value)
    return updated

