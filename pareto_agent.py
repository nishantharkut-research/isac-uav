"""Baseline-anchored preference-conditioned vector-critic transfer agent."""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import cfg
from conservative_gate import GateDecision, select_residual_scale
from pareto_protocol import PNG_PREFERENCE, ParetoAnchors, validate_preferences
from source_integrity import policy_model_digest
from vector_replay import VectorReplayBuffer


ALGORITHM_ID = "ba_pcdtl_v1"


class ConditionedResidualActor(nn.Module):
    """Residual policy with state and two-objective preference inputs."""

    def __init__(self, state_dim: int, action_dim: int, source_actor=None) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.l1 = nn.Linear(self.state_dim + 2, 400)
        self.ln1 = nn.LayerNorm(400)
        self.l2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.out = nn.Linear(300, self.action_dim)
        self._initialize(source_actor)

    def _initialize(self, source_actor) -> None:
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.l1.weight[:, self.state_dim:].zero_()
            if source_actor is None:
                return
            if source_actor.l1.out_features != self.l1.out_features:
                raise ValueError("Baseline actor trunk is not shape-compatible.")
            source_state_dim = source_actor.l1.in_features
            if source_state_dim != self.state_dim:
                raise ValueError("Baseline and residual actor state dimensions differ.")
            self.l1.weight[:, :self.state_dim].copy_(source_actor.l1.weight)
            self.l1.bias.copy_(source_actor.l1.bias)
            self.ln1.load_state_dict(source_actor.ln1.state_dict())
            self.l2.load_state_dict(source_actor.l2.state_dict())
            self.ln2.load_state_dict(source_actor.ln2.state_dict())

    def forward(self, state: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([state, preference], dim=-1)
        hidden = torch.relu(self.ln1(self.l1(inputs)))
        hidden = torch.relu(self.ln2(self.l2(hidden)))
        return torch.tanh(self.out(hidden))


class VectorCritic(nn.Module):
    """Late-fusion two-component action-value function."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        source_critic=None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.s_l1 = nn.Linear(self.state_dim + 2, 400)
        self.s_ln1 = nn.LayerNorm(400)
        self.m_l2 = nn.Linear(400 + self.action_dim, 300)
        self.m_ln2 = nn.LayerNorm(300)
        self.out = nn.Linear(300, 2)
        self._initialize(source_critic)

    def _initialize(self, source_critic) -> None:
        nn.init.xavier_uniform_(self.s_l1.weight)
        nn.init.zeros_(self.s_l1.bias)
        nn.init.xavier_uniform_(self.m_l2.weight)
        nn.init.zeros_(self.m_l2.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.s_l1.weight[:, self.state_dim:].zero_()
            if source_critic is None:
                return
            if source_critic.s_l1.in_features != self.state_dim:
                raise ValueError("Baseline and vector critic state dimensions differ.")
            if source_critic.m_l2.in_features != self.m_l2.in_features:
                raise ValueError("Baseline and vector critic action dimensions differ.")
            self.s_l1.weight[:, :self.state_dim].copy_(source_critic.s_l1.weight)
            self.s_l1.bias.copy_(source_critic.s_l1.bias)
            self.s_ln1.load_state_dict(source_critic.s_ln1.state_dict())
            self.m_l2.load_state_dict(source_critic.m_l2.state_dict())
            self.m_ln2.load_state_dict(source_critic.m_ln2.state_dict())
            self.out.weight[0].copy_(source_critic.out.weight[0])
            self.out.bias[0].copy_(source_critic.out.bias[0])

    def forward(
        self, state: torch.Tensor, preference: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        state_inputs = torch.cat([state, preference], dim=-1)
        hidden = torch.relu(self.s_ln1(self.s_l1(state_inputs)))
        hidden = torch.relu(
            self.m_ln2(self.m_l2(torch.cat([hidden, action], dim=-1)))
        )
        return self.out(hidden)


class VectorTwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, source_critic=None) -> None:
        super().__init__()
        self.q1 = VectorCritic(state_dim, action_dim, source_critic)
        self.q2 = VectorCritic(state_dim, action_dim, source_critic)

    def forward(self, state, preference, action):
        return (
            self.q1(state, preference, action),
            self.q2(state, preference, action),
        )


def componentwise_bellman_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    q1_next: torch.Tensor,
    q2_next: torch.Tensor,
) -> torch.Tensor:
    """TD3 target with an independent minimum for each physical objective."""
    if reward.ndim != 2 or reward.shape[-1] != 2:
        raise ValueError("reward must have shape (B, 2).")
    if q1_next.shape != reward.shape or q2_next.shape != reward.shape:
        raise ValueError("Target critic tensors must match the vector reward shape.")
    if done.shape != (reward.shape[0], 1):
        raise ValueError("done must have shape (B, 1).")
    return reward + float(gamma) * torch.minimum(q1_next, q2_next) * (1.0 - done)


def scalarize_q_tensor(
    q_values: torch.Tensor,
    preferences: torch.Tensor,
    anchors: ParetoAnchors,
    augmentation: float = 0.05,
) -> torch.Tensor:
    """Use physical joint sum for PNG rows and Chebyshev otherwise."""
    if q_values.ndim != 2 or q_values.shape[-1] != 2:
        raise ValueError("q_values must have shape (B, 2).")
    if preferences.shape != q_values.shape:
        raise ValueError("preferences must match q_values shape.")
    validate_preferences(preferences.detach().cpu().numpy())
    balanced = torch.all(
        torch.isclose(
            preferences,
            torch.as_tensor(
                PNG_PREFERENCE, dtype=preferences.dtype, device=preferences.device
            ),
            atol=1e-7,
            rtol=0.0,
        ),
        dim=1,
    )
    result = q_values.sum(dim=1)
    if torch.all(balanced):
        return result
    if not anchors.frozen:
        raise RuntimeError("Non-PNG actor updates require frozen Pareto anchors.")
    ideal = torch.as_tensor(anchors.ideal, dtype=q_values.dtype, device=q_values.device)
    nadir = torch.as_tensor(anchors.nadir, dtype=q_values.dtype, device=q_values.device)
    normalized = (q_values - nadir) / (ideal - nadir)
    active = preferences > 0.0
    ratios = normalized / torch.clamp(preferences, min=1e-12)
    ratios = torch.where(active, ratios, torch.full_like(ratios, torch.inf))
    chebyshev = ratios.min(dim=1).values + float(augmentation) * (
        normalized * preferences
    ).sum(dim=1)
    return torch.where(balanced, result, chebyshev)


class BAPCDTLAgent:
    """Residual TD3-style multi-objective transfer agent anchored to a policy."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        baseline_agent,
        state_dim: int,
        action_dim: int,
        *,
        anchors: ParetoAnchors,
        residual_radius: float = 0.25,
        ensemble_size: int = 2,
        replay_capacity: int = cfg.replay_size,
        batch_size: int = cfg.B_batch,
        lr_actor: float = cfg.lr_actor,
        lr_critic: float = cfg.lr_critic,
        gamma: float = cfg.gamma,
        tau: float = cfg.tau,
        exploration_std: float = cfg.noise_std,
        target_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        residual_scales=(0.0, 0.25, 0.5, 0.75, 1.0),
        uncertainty_coefficient: float = 1.0,
        gate_margin: float = 0.0,
        transfer_trunks: bool = True,
        anchor_enabled: bool = True,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        if baseline_agent.actor.l1.in_features != self.state_dim:
            raise ValueError("Baseline actor state dimension mismatch.")
        if baseline_agent.actor.out.out_features != self.action_dim:
            raise ValueError("Baseline actor action dimension mismatch.")
        if int(ensemble_size) < 2:
            raise ValueError("BA-PCDTL requires at least two twin-critic pairs.")
        if not 0.0 < float(residual_radius) <= 1.0:
            raise ValueError("residual_radius must lie in (0, 1].")
        if int(batch_size) <= 0 or int(policy_delay) <= 0:
            raise ValueError("batch_size and policy_delay must be positive.")
        self.residual_radius = float(residual_radius)
        self.ensemble_size = int(ensemble_size)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.exploration_std = float(exploration_std)
        self.target_noise_std = float(target_noise_std)
        self.target_noise_clip = float(target_noise_clip)
        self.policy_delay = int(policy_delay)
        self.residual_scales = tuple(float(value) for value in residual_scales)
        self.uncertainty_coefficient = float(uncertainty_coefficient)
        self.gate_margin = float(gate_margin)
        self.transfer_trunks = bool(transfer_trunks)
        self.anchor_enabled = bool(anchor_enabled)
        if not self.anchor_enabled and self.transfer_trunks:
            raise ValueError("Unanchored agents cannot transfer baseline trunks.")
        self.anchors = ParetoAnchors.from_state_dict(anchors.state_dict())

        self.baseline_actor = copy.deepcopy(baseline_agent.actor).eval()
        if not self.anchor_enabled:
            with torch.no_grad():
                for parameter in self.baseline_actor.parameters():
                    parameter.zero_()
        for parameter in self.baseline_actor.parameters():
            parameter.requires_grad_(False)
        self.baseline_policy_digest = policy_model_digest(self.baseline_actor)

        source_actor = baseline_agent.actor if self.transfer_trunks else None
        source_critic = baseline_agent.critic if self.transfer_trunks else None
        self.residual_actor = ConditionedResidualActor(
            self.state_dim, self.action_dim, source_actor
        )
        self.residual_actor_target = copy.deepcopy(self.residual_actor)
        self.critic_ensembles = nn.ModuleList(
            [
                VectorTwinCritic(self.state_dim, self.action_dim, source_critic)
                for _ in range(self.ensemble_size)
            ]
        )
        self.critic_targets = copy.deepcopy(self.critic_ensembles)
        for target in self.critic_targets:
            for parameter in target.parameters():
                parameter.requires_grad_(False)

        self.actor_optimizer = optim.Adam(self.residual_actor.parameters(), lr=lr_actor)
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic)
            for critic in self.critic_ensembles
        ]
        self.replay = VectorReplayBuffer(
            replay_capacity, self.state_dim, self.action_dim, reward_dim=2
        )
        self.total_steps = 0

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)

    def _baseline_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.baseline_actor(state)

    def _critic_ensemble_values(
        self, state: torch.Tensor, preference: torch.Tensor, actions: torch.Tensor
    ) -> np.ndarray:
        values = []
        repeated_state = state.repeat(actions.shape[0], 1)
        repeated_preference = preference.repeat(actions.shape[0], 1)
        with torch.no_grad():
            for twin in self.critic_ensembles:
                q1, q2 = twin(repeated_state, repeated_preference, actions)
                conservative = torch.minimum(q1, q2)
                scalar = scalarize_q_tensor(
                    conservative, repeated_preference, self.anchors
                )
                values.append(scalar.cpu().numpy())
        return np.stack(values, axis=0)

    def select_action(
        self,
        state,
        preference,
        *,
        explore: bool = False,
        use_gate: bool = True,
        return_info: bool = False,
    ):
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dim,) or not np.all(np.isfinite(state_array)):
            raise ValueError("state has an invalid shape or nonfinite values.")
        preference_array = validate_preferences([preference])[0].astype(np.float32)
        state_tensor = torch.as_tensor(state_array).unsqueeze(0)
        preference_tensor = torch.as_tensor(preference_array).unsqueeze(0)
        with torch.no_grad():
            baseline = self.baseline_actor(state_tensor)
            residual = self.residual_actor(state_tensor, preference_tensor)

        if explore:
            noise = self.exploration_std * torch.randn_like(residual)
            explored_residual = torch.clamp(residual + noise, -1.0, 1.0)
            action = torch.clamp(
                baseline + self.residual_radius * explored_residual, -1.0, 1.0
            )
            info = {
                "residual_scale": 1.0,
                "fallback_reason": None,
                "gate_used": False,
            }
        elif not use_gate:
            action = torch.clamp(
                baseline + self.residual_radius * residual, -1.0, 1.0
            )
            info = {
                "residual_scale": 1.0,
                "fallback_reason": None,
                "gate_used": False,
            }
        else:
            scales = torch.as_tensor(
                self.residual_scales, dtype=state_tensor.dtype
            ).reshape(-1, 1)
            candidate_actions = torch.clamp(
                baseline + scales * self.residual_radius * residual, -1.0, 1.0
            )
            ensemble_values = self._critic_ensemble_values(
                state_tensor, preference_tensor, candidate_actions
            )
            decision: GateDecision = select_residual_scale(
                self.residual_scales,
                ensemble_values,
                self.uncertainty_coefficient,
                self.gate_margin,
            )
            action = candidate_actions[decision.selected_index].unsqueeze(0)
            info = {
                "residual_scale": decision.scale,
                "fallback_reason": decision.fallback_reason,
                "gate_used": True,
                "baseline_value": decision.baseline_value,
                "selected_lcb": decision.selected_lcb,
                "uncertainty": decision.uncertainty,
            }
        result = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)
        return (result, info) if return_info else result

    def update(self, preferences) -> dict:
        if len(self.replay) < self.batch_size:
            return {}
        preference_array = validate_preferences(preferences).astype(np.float32)
        if preference_array.shape != (self.batch_size, 2):
            raise ValueError(
                f"preferences must have shape ({self.batch_size}, 2)."
            )
        state, action, reward, next_state, done = self.replay.sample(self.batch_size)
        preference = torch.as_tensor(preference_array, dtype=torch.float32)

        with torch.no_grad():
            baseline_next = self.baseline_actor(next_state)
            residual_next = self.residual_actor_target(next_state, preference)
            smoothing = torch.clamp(
                self.target_noise_std * torch.randn_like(residual_next),
                -self.target_noise_clip,
                self.target_noise_clip,
            )
            next_action = torch.clamp(
                baseline_next
                + self.residual_radius * torch.clamp(residual_next + smoothing, -1.0, 1.0),
                -1.0,
                1.0,
            )

        critic_losses = []
        for critic, target, optimizer in zip(
            self.critic_ensembles, self.critic_targets, self.critic_optimizers
        ):
            with torch.no_grad():
                q1_next, q2_next = target(next_state, preference, next_action)
                bellman = componentwise_bellman_target(
                    reward, done, self.gamma, q1_next, q2_next
                )
            q1, q2 = critic(state, preference, action)
            loss = nn.functional.mse_loss(q1, bellman) + nn.functional.mse_loss(
                q2, bellman
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            critic_losses.append(loss.detach())

        self.total_steps += 1
        actor_loss_value = None
        if self.total_steps % self.policy_delay == 0:
            baseline_action = self.baseline_actor(state)
            residual_action = self.residual_actor(state, preference)
            proposed_action = torch.clamp(
                baseline_action + self.residual_radius * residual_action, -1.0, 1.0
            )
            vector_values = []
            for critic in self.critic_ensembles:
                q1, q2 = critic(state, preference, proposed_action)
                vector_values.append(torch.minimum(q1, q2))
            ensemble_value = torch.stack(vector_values, dim=0).mean(dim=0)
            actor_loss = -scalarize_q_tensor(
                ensemble_value, preference, self.anchors
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.residual_actor.parameters(), 1.0)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().item())

            self._soft_update(
                self.residual_actor_target, self.residual_actor, self.tau
            )
            for target, critic in zip(self.critic_targets, self.critic_ensembles):
                self._soft_update(target, critic, self.tau)

        result = {
            "critic_loss": float(torch.stack(critic_losses).mean().item()),
            "actor_updated": actor_loss_value is not None,
        }
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        return result

    def _contract(self) -> dict:
        return {
            "residual_radius": self.residual_radius,
            "ensemble_size": self.ensemble_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "exploration_std": self.exploration_std,
            "target_noise_std": self.target_noise_std,
            "target_noise_clip": self.target_noise_clip,
            "policy_delay": self.policy_delay,
            "residual_scales": list(self.residual_scales),
            "uncertainty_coefficient": self.uncertainty_coefficient,
            "gate_margin": self.gate_margin,
            "transfer_trunks": self.transfer_trunks,
            "anchor_enabled": self.anchor_enabled,
        }

    def training_state_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "algorithm_id": ALGORITHM_ID,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "baseline_policy_digest": self.baseline_policy_digest,
            "baseline_actor": self.baseline_actor.state_dict(),
            "contract": self._contract(),
            "residual_actor": self.residual_actor.state_dict(),
            "residual_actor_target": self.residual_actor_target.state_dict(),
            "critic_ensembles": self.critic_ensembles.state_dict(),
            "critic_targets": self.critic_targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [item.state_dict() for item in self.critic_optimizers],
            "replay": self.replay.state_dict(),
            "anchors": self.anchors.state_dict(),
            "total_steps": self.total_steps,
        }

    def load_training_state_dict(self, state: Mapping) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("Unsupported BA-PCDTL checkpoint schema.")
        if state.get("algorithm_id") != ALGORITHM_ID:
            raise ValueError("BA-PCDTL algorithm contract mismatch.")
        if (int(state["state_dim"]), int(state["action_dim"])) != (
            self.state_dim,
            self.action_dim,
        ):
            raise ValueError("BA-PCDTL checkpoint dimensions mismatch.")
        if state.get("baseline_policy_digest") != self.baseline_policy_digest:
            raise ValueError("BA-PCDTL baseline policy digest mismatch.")
        if dict(state.get("contract", {})) != self._contract():
            raise ValueError("BA-PCDTL hyperparameter contract mismatch.")

        baseline_candidate = copy.deepcopy(self.baseline_actor)
        baseline_candidate.load_state_dict(state["baseline_actor"])
        if policy_model_digest(baseline_candidate) != self.baseline_policy_digest:
            raise ValueError("Serialized baseline actor payload is inconsistent.")

        self.baseline_actor.load_state_dict(state["baseline_actor"])
        self.residual_actor.load_state_dict(state["residual_actor"])
        self.residual_actor_target.load_state_dict(state["residual_actor_target"])
        self.critic_ensembles.load_state_dict(state["critic_ensembles"])
        self.critic_targets.load_state_dict(state["critic_targets"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        optimizer_states = state["critic_optimizers"]
        if len(optimizer_states) != len(self.critic_optimizers):
            raise ValueError("BA-PCDTTL critic optimizer count mismatch.")
        for optimizer, optimizer_state in zip(
            self.critic_optimizers, optimizer_states
        ):
            optimizer.load_state_dict(optimizer_state)
        self.replay.load_state_dict(state["replay"])
        self.anchors = ParetoAnchors.from_state_dict(state["anchors"])
        self.total_steps = int(state["total_steps"])
        self.baseline_actor.eval()
        for parameter in self.baseline_actor.parameters():
            parameter.requires_grad_(False)
