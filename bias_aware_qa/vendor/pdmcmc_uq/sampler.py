from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .agents import ConstraintAgent, choose_proposal_from_actions, default_agents, disagreement_rate
from .base_posterior import BaseImputationPosterior
from .constraints import Constraint
from .dual_update import update_lambdas
from .metrics import compute_group_gaps
from .state import UQState


def metropolis_accept_prob(
    current_log_target: float,
    proposed_log_target: float,
    log_q_forward: float = 0.0,
    log_q_reverse: float = 0.0,
) -> float:
    log_alpha = min(
        0.0,
        float(proposed_log_target)
        + float(log_q_reverse)
        - float(current_log_target)
        - float(log_q_forward),
    )
    if log_alpha <= -745.0:
        return 0.0
    return float(math.exp(log_alpha))


@dataclass
class SamplerConfig:
    method: str = "adaptive_single_agent"
    adapt_rounds: int = 10
    adapt_steps: int = 500
    chain_length: int = 8000
    burn_in: int = 1000
    rho0: float = 0.005
    decreasing_rho: bool = True
    proposal_type: str = "high_bias_cell"
    ref_group: str = "EU"
    target_group: str = "GS"
    seed: int = 0
    log_every: int = 1
    agent_epsilon: float = 0.1
    payoff_acceptance_weight: float = 1.0
    payoff_improvement_weight: float = 0.5
    payoff_constraint_weight: float = 1.0
    payoff_disagreement_weight: float = 0.1
    proposal_cost_random: float = 0.0
    proposal_cost_local: float = 0.01
    proposal_cost_high_bias: float = 0.02
    proposal_cost_uncertainty: float = 0.02
    final_mix_high_bias: float = 0.45
    final_mix_uncertainty: float = 0.25
    final_mix_local: float = 0.20
    final_mix_random: float = 0.10
    final_mixture_source: str = "fixed"
    final_proposal_weights: dict[str, float] = field(default_factory=dict)
    learned_mixture_floor: float = 0.05
    lambda_freeze_strategy: str = "last"
    store_samples: bool = True
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainResult:
    final_state: UQState
    samples: list[UQState]
    logs: pd.DataFrame
    agent_logs: pd.DataFrame = field(default_factory=pd.DataFrame)


class ConstraintCalibratedPDMCMCUQ:
    """Constraint-calibrated MH sampler for completed-table UQ states."""

    def __init__(
        self,
        base_posterior: BaseImputationPosterior,
        constraints: list[Constraint],
        config: SamplerConfig | dict[str, Any] | None = None,
        agents: list[ConstraintAgent] | None = None,
    ) -> None:
        self.base_posterior = base_posterior
        self.constraints = constraints
        if config is None:
            self.config = SamplerConfig()
        elif isinstance(config, SamplerConfig):
            self.config = config
        else:
            self.config = SamplerConfig(**config)
        self.lambda_trajectory_: pd.DataFrame = pd.DataFrame()
        self.agent_logs_: pd.DataFrame = pd.DataFrame()
        self.adaptation_logs_: pd.DataFrame = pd.DataFrame()
        self.learned_proposal_weights_: dict[str, float] = {}
        self.agents = agents if agents is not None else default_agents(self.config.agent_epsilon)

    def constraint_scores(self, state: UQState) -> dict[str, float]:
        return {
            constraint.name: float(constraint.score(state, self.config.context))
            for constraint in self.constraints
        }

    def log_target(self, state: UQState, scores: dict[str, float] | None = None) -> float:
        scores = scores if scores is not None else self.constraint_scores(state)
        penalty = sum(float(c.lambda_value) * float(scores[c.name]) for c in self.constraints)
        return float(self.base_posterior.log_prob(state) - penalty)

    def _quantity_log_fields(self, state: UQState) -> dict[str, float]:
        fields: dict[str, float] = {}
        gaps = compute_group_gaps(
            state,
            self.base_posterior.feature_cols,
            ref_group=self.config.ref_group,
            target_group=self.config.target_group,
        )
        for col, value in gaps.items():
            fields[f"gap_{col}"] = float(value)
        if "carbon_kg_per_kwh" in state.columns:
            j = state.columns.index("carbon_kg_per_kwh")
            fields["mean_carbon"] = float(state.X_complete[:, j].mean())
            fields["carbon_gap"] = float(gaps["carbon_kg_per_kwh"])
        return fields

    def mh_step(
        self,
        state: UQState,
        rng: np.random.Generator,
        proposal_type: str = "random_cell",
    ) -> tuple[UQState, dict[str, Any]]:
        old_scores = self.constraint_scores(state)
        old_log_target = self.log_target(state, old_scores)
        proposed_state, log_q_forward, log_q_reverse, proposal_info = self.base_posterior.propose(
            state,
            proposal_type=proposal_type,
            rng=rng,
        )
        new_scores = self.constraint_scores(proposed_state)
        new_log_target = self.log_target(proposed_state, new_scores)
        alpha = metropolis_accept_prob(old_log_target, new_log_target, log_q_forward, log_q_reverse)
        accepted = bool(rng.random() < alpha)
        next_state = proposed_state if accepted else state
        next_scores = new_scores if accepted else old_scores
        next_log_target = new_log_target if accepted else old_log_target

        log_row: dict[str, Any] = {
            "accepted": accepted,
            "acceptance_prob": float(alpha),
            "log_target": float(next_log_target),
            "proposal_type": proposal_type,
            **{f"old_C_{k}": float(v) for k, v in old_scores.items()},
            **{f"proposed_C_{k}": float(v) for k, v in new_scores.items()},
            **{f"C_{k}": float(v) for k, v in next_scores.items()},
            **{f"lambda_{c.name}": float(c.lambda_value) for c in self.constraints},
            **self._quantity_log_fields(next_state),
        }
        log_row.update({f"proposal_{k}": v for k, v in proposal_info.items() if isinstance(v, (str, int, float, bool))})
        return next_state, log_row

    def _proposal_cost(self, proposal_type: str) -> float:
        costs = {
            "random_cell": self.config.proposal_cost_random,
            "local": self.config.proposal_cost_local,
            "high_bias_cell": self.config.proposal_cost_high_bias,
            "uncertainty_cell": self.config.proposal_cost_uncertainty,
        }
        return float(costs.get(proposal_type, 0.02))

    def _fixed_final_proposal_weights(self) -> dict[str, float]:
        weights = {
            "high_bias_cell": float(self.config.final_mix_high_bias),
            "uncertainty_cell": float(self.config.final_mix_uncertainty),
            "local": float(self.config.final_mix_local),
            "random_cell": float(self.config.final_mix_random),
        }
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0.0:
            return {key: 0.25 for key in weights}
        return {key: max(value, 0.0) / total for key, value in weights.items()}

    def _final_proposal_weights(self) -> tuple[list[str], np.ndarray, str]:
        source = self.config.final_mixture_source
        if source not in {"fixed", "learned"}:
            raise ValueError("final_mixture_source must be 'fixed' or 'learned'.")
        weights = self._fixed_final_proposal_weights()
        if source == "learned":
            learned = self.config.final_proposal_weights or self.learned_proposal_weights_
            if learned:
                weights.update({key: float(value) for key, value in learned.items() if key in weights})
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0.0:
            weights = {key: 0.25 for key in weights}
            total = 1.0
        proposals = ["high_bias_cell", "uncertainty_cell", "local", "random_cell"]
        probs = np.array([max(weights[p], 0.0) / total for p in proposals], dtype=float)
        return proposals, probs, source

    def _learn_final_proposal_weights(self) -> dict[str, float]:
        if self.adaptation_logs_.empty or "proposal_type" not in self.adaptation_logs_:
            return {}
        proposals = ["high_bias_cell", "uncertainty_cell", "local", "random_cell"]
        df = self.adaptation_logs_[self.adaptation_logs_["proposal_type"].isin(proposals)].copy()
        if df.empty:
            return {}
        old_cols = [f"old_C_{constraint.name}" for constraint in self.constraints]
        proposed_cols = [f"proposed_C_{constraint.name}" for constraint in self.constraints]
        if all(col in df for col in old_cols + proposed_cols):
            gain = np.zeros(len(df), dtype=float)
            regression = np.zeros(len(df), dtype=float)
            for old_col, new_col in zip(old_cols, proposed_cols):
                delta = df[old_col].to_numpy(dtype=float) - df[new_col].to_numpy(dtype=float)
                gain += np.maximum(delta, 0.0)
                regression += np.maximum(-delta, 0.0)
            acceptance = df["acceptance_prob"].to_numpy(dtype=float) if "acceptance_prob" in df else 0.0
            df["proposal_utility"] = gain - regression + 0.15 * acceptance
            score_col = "proposal_utility"
        else:
            score_col = "payoff" if "payoff" in df else "acceptance_prob"
        grouped = df.groupby("proposal_type")[score_col].mean()
        scores = np.array([float(grouped.get(p, np.nan)) for p in proposals], dtype=float)
        if not np.isfinite(scores).any():
            return {}
        finite_min = float(np.nanmin(scores))
        scores = np.nan_to_num(scores, nan=finite_min)
        shifted = scores - float(np.min(scores))
        floor = max(float(self.config.learned_mixture_floor), 1e-6)
        weights = shifted + floor
        weights = weights / weights.sum()
        return {proposal: float(weight) for proposal, weight in zip(proposals, weights)}

    def _agent_payoff(self, row: dict[str, Any], actions: dict[str, str]) -> float:
        regression = 0.0
        improvement = 0.0
        for c in self.constraints:
            old = float(row.get(f"old_C_{c.name}", 0.0))
            new = float(row.get(f"proposed_C_{c.name}", old))
            regression += max(0.0, new - old)
            improvement += max(0.0, old - new)
        proposal_type = str(row.get("proposal_type", "random_cell"))
        return float(
            self.config.payoff_acceptance_weight * float(row.get("acceptance_prob", 0.0))
            + self.config.payoff_improvement_weight * improvement
            - self.config.payoff_constraint_weight * regression
            - self.config.payoff_disagreement_weight * disagreement_rate(actions)
            - self._proposal_cost(proposal_type)
        )

    def _agent_step(
        self,
        state: UQState,
        rng: np.random.Generator,
        frozen_agents: bool,
    ) -> tuple[UQState, dict[str, Any], list[dict[str, Any]]]:
        if frozen_agents:
            proposals, weights, mixture_source = self._final_proposal_weights()
            proposal_type = str(rng.choice(proposals, p=weights))
            actions = {
                "FairnessAgent": "C" if proposal_type == "high_bias_cell" else "D",
                "CalibrationAgent": "C" if proposal_type == "uncertainty_cell" else "D",
                "EfficiencyAgent": "C" if proposal_type == "local" else "D",
                "CoverageAgent": "D",
            }
        else:
            actions = {agent.name: agent.choose_action(rng, frozen=frozen_agents) for agent in self.agents}
            proposal_type = choose_proposal_from_actions(actions)
        next_state, row = self.mh_step(state, rng, proposal_type=proposal_type)
        payoff = self._agent_payoff(row, actions)

        agent_rows: list[dict[str, Any]] = []
        for agent in self.agents:
            action = actions[agent.name]
            if not frozen_agents:
                agent.update(action, payoff)
            agent_rows.append(
                {
                    "agent_name": agent.name,
                    "constraint_name": agent.constraint_name,
                    "action": action,
                    "q_C": float(agent.q_values.get("C", 0.0)),
                    "q_D": float(agent.q_values.get("D", 0.0)),
                    "payoff": float(payoff),
                    "proposal_type": proposal_type,
                    "mixture_source": mixture_source if frozen_agents else "adaptive_actions",
                    "frozen": bool(frozen_agents),
                }
            )
        row["payoff"] = float(payoff)
        row["n_cooperate"] = int(sum(1 for a in actions.values() if a == "C"))
        row["agent_disagreement"] = disagreement_rate(actions)
        row["proposal_mixture_source"] = mixture_source if frozen_agents else "adaptive_actions"
        return next_state, row, agent_rows

    def run_chain(
        self,
        initial_state: UQState,
        n_steps: int,
        burn_in: int = 0,
        proposal_type: str = "random_cell",
        collect: bool = True,
        seed: int | None = None,
        use_agents: bool = False,
        frozen_agents: bool = False,
    ) -> ChainResult:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        state = initial_state.copy()
        samples: list[UQState] = []
        logs: list[dict[str, Any]] = []
        agent_logs: list[dict[str, Any]] = []

        for iteration in range(int(n_steps)):
            if use_agents:
                state, row, step_agent_rows = self._agent_step(state, rng, frozen_agents=frozen_agents)
                for agent_row in step_agent_rows:
                    agent_row["iteration"] = int(iteration)
                    agent_row["phase"] = "sample" if iteration >= burn_in else "burn_in"
                    agent_logs.append(agent_row)
            else:
                state, row = self.mh_step(state, rng, proposal_type=proposal_type)
            row["iteration"] = int(iteration)
            row["phase"] = "sample" if iteration >= burn_in else "burn_in"
            if self.config.log_every <= 1 or iteration % self.config.log_every == 0:
                logs.append(row)
            if collect and iteration >= burn_in:
                samples.append(state.copy())

        return ChainResult(
            final_state=state.copy(),
            samples=samples,
            logs=pd.DataFrame(logs),
            agent_logs=pd.DataFrame(agent_logs),
        )

    def adapt_lambdas(
        self,
        initial_state: UQState,
        rounds: int | None = None,
        steps_per_round: int | None = None,
        proposal_type: str | None = None,
    ) -> tuple[UQState, pd.DataFrame]:
        rounds = self.config.adapt_rounds if rounds is None else int(rounds)
        steps_per_round = self.config.adapt_steps if steps_per_round is None else int(steps_per_round)
        proposal_type = self.config.proposal_type if proposal_type is None else proposal_type
        current = initial_state.copy()
        trajectory: list[dict[str, Any]] = []
        all_agent_logs: list[pd.DataFrame] = []
        all_adaptation_logs: list[pd.DataFrame] = []
        use_agents = self.config.method == "adaptive_multi_agent"

        for round_index in range(rounds):
            result = self.run_chain(
                current,
                n_steps=steps_per_round,
                burn_in=0,
                proposal_type=proposal_type,
                collect=False,
                seed=self.config.seed + 10_000 + round_index,
                use_agents=use_agents,
                frozen_agents=False,
            )
            current = result.final_state
            if not result.logs.empty:
                logs = result.logs.copy()
                logs["round"] = int(round_index)
                all_adaptation_logs.append(logs)
            if not result.agent_logs.empty:
                agent_logs = result.agent_logs.copy()
                agent_logs["round"] = int(round_index)
                all_agent_logs.append(agent_logs)
            mean_scores = {
                c.name: float(result.logs[f"C_{c.name}"].mean()) if f"C_{c.name}" in result.logs else 0.0
                for c in self.constraints
            }
            before = {c.name: float(c.lambda_value) for c in self.constraints}
            after = update_lambdas(
                self.constraints,
                mean_scores,
                round_index=round_index,
                rho0=self.config.rho0,
                decreasing=self.config.decreasing_rho,
            )
            for c in self.constraints:
                trajectory.append(
                    {
                        "round": int(round_index),
                        "constraint": c.name,
                        "mean_score": float(mean_scores[c.name]),
                        "epsilon": float(c.epsilon),
                        "lambda_before": float(before[c.name]),
                        "lambda_after": float(after[c.name]),
                    }
                )

        self.lambda_trajectory_ = pd.DataFrame(trajectory)
        self.agent_logs_ = pd.concat(all_agent_logs, ignore_index=True) if all_agent_logs else pd.DataFrame()
        self.adaptation_logs_ = (
            pd.concat(all_adaptation_logs, ignore_index=True) if all_adaptation_logs else pd.DataFrame()
        )
        self.learned_proposal_weights_ = self._learn_final_proposal_weights() if use_agents else {}
        if self.config.lambda_freeze_strategy == "max" and not self.lambda_trajectory_.empty:
            max_lambdas = self.lambda_trajectory_.groupby("constraint")["lambda_after"].max().to_dict()
            for constraint in self.constraints:
                if constraint.name in max_lambdas:
                    constraint.lambda_value = float(max_lambdas[constraint.name])
        elif self.config.lambda_freeze_strategy != "last":
            raise ValueError("lambda_freeze_strategy must be 'last' or 'max'.")
        return current, self.lambda_trajectory_

    def run(self, initial_state: UQState) -> dict[str, Any]:
        method = self.config.method
        if method not in {"fixed_lambda", "adaptive_single_agent", "adaptive_multi_agent"}:
            raise ValueError(
                "Unknown method. Use 'fixed_lambda', 'adaptive_single_agent', or 'adaptive_multi_agent'."
            )

        if method == "fixed_lambda":
            adapted_state = initial_state.copy()
            lambda_trajectory = pd.DataFrame()
            self.agent_logs_ = pd.DataFrame()
        else:
            adapted_state, lambda_trajectory = self.adapt_lambdas(initial_state)

        use_agents_final = method == "adaptive_multi_agent"
        final = self.run_chain(
            adapted_state,
            n_steps=self.config.chain_length,
            burn_in=self.config.burn_in,
            proposal_type=self.config.proposal_type,
            collect=self.config.store_samples,
            seed=self.config.seed + 999_999,
            use_agents=use_agents_final,
            frozen_agents=True,
        )
        if not final.agent_logs.empty:
            final_agent_logs = final.agent_logs.copy()
            final_agent_logs["round"] = -1
            self.agent_logs_ = pd.concat([self.agent_logs_, final_agent_logs], ignore_index=True)
        return {
            "adapted_state": adapted_state,
            "lambda_trajectory": lambda_trajectory,
            "final_result": final,
            "agent_logs": self.agent_logs_.copy(),
            "final_lambdas": {c.name: float(c.lambda_value) for c in self.constraints},
            "learned_proposal_weights": dict(self.learned_proposal_weights_),
        }
