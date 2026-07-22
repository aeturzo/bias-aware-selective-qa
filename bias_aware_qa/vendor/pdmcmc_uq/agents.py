from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class ConstraintAgent:
    """Small epsilon-greedy PD-style controller for proposal selection."""

    name: str
    constraint_name: str
    epsilon_greedy: float = 0.1
    q_values: dict[str, float] = field(default_factory=lambda: {"C": 0.0, "D": 0.0})
    action_counts: dict[str, int] = field(default_factory=lambda: {"C": 0, "D": 0})

    actions = ("C", "D")

    def choose_action(self, rng: np.random.Generator, frozen: bool = False) -> str:
        if (not frozen) and float(rng.random()) < self.epsilon_greedy:
            return str(rng.choice(self.actions))
        q_c = float(self.q_values.get("C", 0.0))
        q_d = float(self.q_values.get("D", 0.0))
        return "C" if q_c >= q_d else "D"

    def update(self, action: str, payoff: float) -> None:
        if action not in self.actions:
            raise ValueError(f"Unknown action {action!r}.")
        self.action_counts[action] = int(self.action_counts.get(action, 0)) + 1
        n = self.action_counts[action]
        old = float(self.q_values.get(action, 0.0))
        self.q_values[action] = old + (float(payoff) - old) / float(n)


def default_agents(epsilon_greedy: float = 0.1) -> list[ConstraintAgent]:
    # Role priors make the game non-vacuous from the first adaptation round:
    # fairness/calibration start mildly cooperative, while efficiency/coverage
    # must earn cooperation through payoff. These are proposal-policy priors,
    # not target-distribution changes; the final MH correction still targets the
    # frozen constrained posterior.
    return [
        ConstraintAgent(
            "FairnessAgent",
            "fairness_gap",
            epsilon_greedy=epsilon_greedy,
            q_values={"C": 0.15, "D": 0.0},
        ),
        ConstraintAgent(
            "CalibrationAgent",
            "bias_penalty",
            epsilon_greedy=epsilon_greedy,
            q_values={"C": 0.08, "D": 0.0},
        ),
        ConstraintAgent(
            "EfficiencyAgent",
            "width",
            epsilon_greedy=epsilon_greedy,
            q_values={"C": 0.0, "D": 0.05},
        ),
        ConstraintAgent(
            "CoverageAgent",
            "coverage",
            epsilon_greedy=epsilon_greedy,
            q_values={"C": 0.0, "D": 0.05},
        ),
    ]


def choose_proposal_from_actions(actions: Mapping[str, str]) -> str:
    """Map cooperative/defective agent actions to a frozen proposal mode."""

    n_coop = sum(1 for action in actions.values() if action == "C")
    if n_coop <= 1:
        return "random_cell"

    if actions.get("FairnessAgent") == "C" or n_coop >= 3:
        return "high_bias_cell"

    if actions.get("CalibrationAgent") == "C":
        return "uncertainty_cell"

    if actions.get("EfficiencyAgent") == "C":
        return "local"

    return "random_cell"


def disagreement_rate(actions: Mapping[str, str]) -> float:
    if not actions:
        return 0.0
    n_c = sum(1 for action in actions.values() if action == "C")
    n = len(actions)
    return float(min(n_c, n - n_c) / n)
