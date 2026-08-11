"""
dtl_agent.py — Deep Transfer Learning (DTL) target agent.

Section IV of Saikia et al., IEEE Trans. Commun., 2025.

Key idea (Eq. 30):
  Q*(s^t, a^t) = υ^t · Q(s^t, a^t | ς^Q)         ← source agent knowledge
               + (1 − υ^t) · Q'(s^t, a^t | ς^{Q'}) ← target agent knowledge

The paper does not specify how Eq. (30) enters a complete Bellman target or how
the transfer rate changes over time. The evidence-ranked primary blends the
current source Q-value with the ordinary target Bellman target. The earlier
next-state continuation remains an explicit sensitivity branch.

The target agent is initialized from the source weights. A constant transfer
rate is primary; linear annealing is a separately fingerprinted sensitivity.
"""
from config import cfg
import torch
import torch.nn as nn

from ddpg_agent import DDPGAgent
from ambiguity_hypotheses import (
    get_transfer_schedule_hypothesis,
    get_transfer_update_hypothesis,
)


def build_eq30_critic_target(mode, upsilon, reward, done, gamma,
                             source_current, source_next, target_next):
    """Construct one of the two predeclared trainable readings of Eq. (30)."""
    get_transfer_update_hypothesis(mode)
    upsilon = float(upsilon)
    if not 0.0 <= upsilon <= 1.0:
        raise ValueError("upsilon must be in [0, 1]")
    if mode == "eq30_current_q_bellman_blend":
        bellman = reward + float(gamma) * target_next * (1.0 - done)
        blended = upsilon * source_current + (1.0 - upsilon) * bellman
        return torch.where(done.bool(), reward, blended)
    if mode == "eq30_next_state_continuation_blend":
        continuation = (
            upsilon * source_next + (1.0 - upsilon) * target_next
        )
        return reward + float(gamma) * continuation * (1.0 - done)
    raise ValueError(f"Unknown transfer update hypothesis: {mode}")


def next_upsilon(schedule, current, minimum, decay):
    """Advance the predeclared transfer schedule by one episode."""
    get_transfer_schedule_hypothesis(schedule)
    current = float(current)
    minimum = float(minimum)
    decay = float(decay)
    if not 0.0 <= minimum <= current <= 1.0:
        raise ValueError("upsilon values must satisfy 0 <= minimum <= current <= 1")
    if decay < 0.0:
        raise ValueError("upsilon decay must be nonnegative")
    if schedule == "constant_upsilon":
        return current
    if schedule == "linear_annealed_upsilon":
        return max(minimum, current - decay)
    raise ValueError(f"Unknown transfer schedule hypothesis: {schedule}")




class DTLAgent(DDPGAgent):
    """
    Transfer-learning enhanced DDPG target agent.

    Inherits the full DDPG implementation and overrides the critic update
    with the configured predeclared interpretation of Eq. (30).
    The source agent is frozen after
    pre-training; only its Q-values (forward pass) are used.

    Parameters
    ----------
    source_agent  : pre-trained DDPGAgent (source task)
    state_dim     : target task state dimension (may differ from source)
    action_dim    : target task action dimension
    """

    def __init__(self,
                 source_agent: DDPGAgent,
                 state_dim:    int,
                 action_dim:   int,
                 transfer_update_mode: str = None,
                 transfer_schedule: str = None):

        super().__init__(state_dim, action_dim)

        self.source  = source_agent          # frozen after pre-training
        self.upsilon = cfg.upsilon_0         # transfer weight  υ  (Eq. 30)
        self.transfer_update_mode = (
            cfg.dtl_transfer_mode
            if transfer_update_mode is None else str(transfer_update_mode)
        )
        self.transfer_schedule = (
            cfg.dtl_transfer_schedule_id
            if transfer_schedule is None else str(transfer_schedule)
        )
        get_transfer_update_hypothesis(self.transfer_update_mode)
        get_transfer_schedule_hypothesis(self.transfer_schedule)
        self._source_dims_match = (
            source_agent.actor.l1.in_features == state_dim
            and source_agent.actor.out.out_features == action_dim
        )
        for param in self.source.actor.parameters():
            param.requires_grad_(False)
        for param in self.source.critic.parameters():
            param.requires_grad_(False)
        self.source.actor.eval()
        self.source.critic.eval()
        self.source.actor_tgt.eval()
        self.source.critic_tgt.eval()

        # ── Warm-start target agent from source weights ───────────────────────
        # If dimensions match, copy directly; otherwise adapt (Sec. IV).
        self._transfer_weights(source_agent, state_dim, action_dim)

    # ── Weight Transfer ───────────────────────────────────────────────────────

    def _transfer_weights(self,
                          source:     DDPGAgent,
                          state_dim:  int,
                          action_dim: int):
        """
        Transfer source weights to target agent (Sec. IV).

        If dimensions are the same: direct copy (Sec. IV — "inherits weights
        from a well-trained model").
        If different: copy shared layers and randomly initialise extra neurons.
        """
        src_s_dim = source.actor.l1.in_features
        src_a_dim = source.actor.out.out_features

        if src_s_dim == state_dim and src_a_dim == action_dim:
            self._hard_copy(self.actor,      source.actor)
            self._hard_copy(self.critic,     source.critic)
            self._hard_copy(self.actor_tgt,  source.actor_tgt)
            self._hard_copy(self.critic_tgt, source.critic_tgt)
        else:
            # Partial transfer: copy hidden layers, re-initialise I/O layers
            self._partial_transfer(self.actor,  source.actor)
            self._partial_transfer(self.critic, source.critic)
            self._hard_copy(self.actor_tgt,  self.actor)
            self._hard_copy(self.critic_tgt, self.critic)

    @staticmethod
    def _partial_transfer(tgt_net: nn.Module, src_net: nn.Module):
        """Copy parameters where shapes match (hidden layers transfer, I/O layers re-init)."""
        for tp, sp in zip(tgt_net.parameters(), src_net.parameters()):
            if tp.shape == sp.shape:
                tp.data.copy_(sp.data)

    # ── DTL Update (Eq. 30) ───────────────────────────────────────────────────

    def update(self) -> dict:
        """
        One DTL gradient step.

        Execute the configured, predeclared trainable interpretation of Eq. 30.
        """
        if len(self.replay) < cfg.B_batch:
            return {}

        s, a, r, ns, done = self.replay.sample()

        # ── Hybrid target Q*  (Eq. 30) ────────────────────────────────────────
        with torch.no_grad():
            a_next   = self.actor_tgt(ns)

            q_target = self.critic_tgt(ns, a_next)
            if self._source_dims_match:
                q_source_current = self.source.critic(s, a)
                q_source_next = self.source.critic_tgt(ns, a_next)
            else:
                # Cross-dimensional transfer is the warm-started shared
                # weights. Evaluating source Q on incompatible vectors would
                # require an adapter that the paper does not define.
                q_source_current = r + self.gamma * q_target * (1.0 - done)
                q_source_next = q_target

            y_t = build_eq30_critic_target(
                self.transfer_update_mode,
                self.upsilon,
                r,
                done,
                self.gamma,
                q_source_current,
                q_source_next,
                q_target,
            )

        # ── Critic update ────────────────────────────────────────────────────
        q_pred      = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_pred, y_t)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # ── Actor update ─────────────────────────────────────────────────────
        a_curr = self.actor(s)
        actor_loss = -self.critic(s, a_curr).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # ── Soft target updates ───────────────────────────────────────────────
        self._soft_update(self.actor_tgt, self.actor, self.tau)
        self._soft_update(self.critic_tgt, self.critic, self.tau)

        self.total_steps += 1
        return dict(critic_loss=float(critic_loss.item()),
                    actor_loss=float(actor_loss.item()),
                    upsilon=self.upsilon)

    def decay_upsilon(self):
        """Advance the configured transfer schedule by one episode."""
        self.upsilon = next_upsilon(
            self.transfer_schedule,
            self.upsilon,
            minimum=cfg.upsilon_min,
            decay=cfg.upsilon_decay,
        )

    def training_state_dict(self) -> dict:
        state = super().training_state_dict()
        state.update({
            "upsilon": float(self.upsilon),
            "transfer_update_mode": self.transfer_update_mode,
            "transfer_schedule": self.transfer_schedule,
            "source_dims_match": bool(self._source_dims_match),
            "source_agent": self.source.training_state_dict(),
        })
        return state

    def load_training_state_dict(self, state: dict):
        super().load_training_state_dict(state)
        if state.get("transfer_update_mode") != self.transfer_update_mode:
            raise ValueError("DTL transfer-update hypothesis mismatch.")
        if state.get("transfer_schedule") != self.transfer_schedule:
            raise ValueError("DTL transfer-schedule hypothesis mismatch.")
        if bool(state["source_dims_match"]) != self._source_dims_match:
            raise ValueError("DTL source/target dimension relationship mismatch.")
        self.source.load_training_state_dict(state["source_agent"])
        self.upsilon = float(state["upsilon"])
        for network in (
            self.source.actor,
            self.source.critic,
            self.source.actor_tgt,
            self.source.critic_tgt,
        ):
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
