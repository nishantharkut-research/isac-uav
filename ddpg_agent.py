"""
ddpg_agent.py — Experience replay, PyTorch networks, and DDPG source agent.

Implements:
  • ReplayBuffer  (capacity W = 10 000, Table IV)
  • ActorNetwork  π(s | ς^μ) → a ∈ [-1,1]^{action_dim}
  • CriticNetwork Q(s, a | ς^Q) → scalar
  • DDPGAgent     source agent  (Eq. 26-29 of the paper)

Reference:
  Saikia et al., IEEE Trans. Commun., 2025.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from config import cfg


# ─── Experience Replay Buffer ─────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular replay buffer  W = 10 000  (Table IV)."""

    def __init__(self, capacity: int = cfg.replay_size):
        self.buffer   = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            np.array(action,     dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size: int = None):
        if batch_size is None:
            batch_size = cfg.B_batch
        batch   = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return (torch.FloatTensor(s),
                torch.FloatTensor(a),
                torch.FloatTensor(r).unsqueeze(1),
                torch.FloatTensor(ns),
                torch.FloatTensor(d).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def state_dict(self) -> dict:
        """Return every transition required for an exact training resume."""
        return {
            "capacity": int(self.buffer.maxlen),
            "transitions": list(self.buffer),
        }

    def load_state_dict(self, state: dict):
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")
        transitions = state.get("transitions", [])
        if len(transitions) > capacity:
            raise ValueError("Replay-buffer state exceeds its declared capacity.")
        self.buffer = deque(transitions, maxlen=capacity)


# ─── Actor Network  π(s | ς^μ) ────────────────────────────────────────────────

class ActorNetwork(nn.Module):
    """
    Deterministic policy  μ(s | ς^μ): s → a ∈ [-1, 1]^{action_dim}.

    Architecture: FC(s→400, LN, ReLU) → FC(400→300, LN, ReLU) → FC(300→a, Tanh)
    LayerNorm normalises activations across different-scale state features.
    Final layer uses small uniform init to keep initial actions near zero.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1   = nn.Linear(state_dim, 400)
        self.ln1  = nn.LayerNorm(400)
        self.l2   = nn.Linear(400, 300)
        self.ln2  = nn.LayerNorm(300)
        self.out  = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.l1.weight);  nn.init.zeros_(self.l1.bias)
        nn.init.xavier_uniform_(self.l2.weight);  nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.ln1(self.l1(s)))
        h = torch.relu(self.ln2(self.l2(h)))
        return self.tanh(self.out(h))


# ─── Critic Network  Q(s, a | ς^Q) ────────────────────────────────────────────

class CriticNetwork(nn.Module):
    """
    Late-fusion action-value function  Q(s, a | ς^Q): (s, a) → scalar.

    Architecture (Lillicrap et al. 2016, Sec. 7):
      Layer 1: FC(s→400, LN, ReLU)          — state features only
      Layer 2: FC(400+a→300, LN, ReLU)      — merge action at layer 2
      Layer 3: FC(300→1)                    — Q-value

    Concatenating action at layer 2 (not layer 1) gives a cleaner, stronger
    gradient ∂Q/∂a that directly informs the actor update.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.s_l1  = nn.Linear(state_dim, 400)
        self.s_ln1 = nn.LayerNorm(400)
        self.m_l2  = nn.Linear(400 + action_dim, 300)
        self.m_ln2 = nn.LayerNorm(300)
        self.out   = nn.Linear(300, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.s_l1.weight); nn.init.zeros_(self.s_l1.bias)
        nn.init.xavier_uniform_(self.m_l2.weight); nn.init.zeros_(self.m_l2.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.s_ln1(self.s_l1(s)))
        h = torch.relu(self.m_ln2(self.m_l2(torch.cat([h, a], dim=-1))))
        return self.out(h)


# ─── DDPG Source Agent ────────────────────────────────────────────────────────

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (source agent).

    Implements Eq. (26)-(29):
      Loss(ς^Q) = (1/B) Σ (y^t − Q(s^t, a^t | ς^Q))²          (Eq. 26)
      y^t = r^t + γ Q'(s^{t+1}, μ'(s^{t+1}|ς^{μ'}) | ς^{Q'})  (Eq. 26)
      ∇_{ς^μ} J ≈ (1/B) Σ ∇_a Q(s,a)|_{a=μ(s)} · ∇_{ς^μ} μ  (Eq. 27)
      ς^{Q'} ← τ ς^Q + (1−τ) ς^{Q'}                            (Eq. 28)
      ς^{μ'} ← τ ς^μ + (1−τ) ς^{μ'}                            (Eq. 29)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float  = cfg.lr_actor,
                 lr_critic: float = cfg.lr_critic,
                 gamma: float     = cfg.gamma,
                 tau: float       = cfg.tau,
                 noise_std: float = cfg.noise_std):

        self.gamma      = gamma
        self.tau        = tau
        self.noise_std  = noise_std
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Online networks
        self.actor  = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)

        # Target networks  ς' (Eq. 28, 29)
        self.actor_tgt  = ActorNetwork(state_dim, action_dim)
        self.critic_tgt = CriticNetwork(state_dim, action_dim)
        self._hard_copy(self.actor_tgt,  self.actor)
        self._hard_copy(self.critic_tgt, self.critic)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay      = ReplayBuffer()
        self.total_steps = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray,
                      explore: bool = True) -> np.ndarray:
        """
        μ(s) + Gaussian exploration noise  σ_n ~ N(0, noise_std²).
        Clipped to [-1, 1].
        """
        with torch.no_grad():
            s    = torch.FloatTensor(state).unsqueeze(0)
            a    = self.actor(s).squeeze(0).numpy()
        if explore:
            a += self.noise_std * np.random.randn(self.action_dim)
        return np.clip(a, -1.0, 1.0)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        One DDPG gradient step matching Eq. (26)-(29).
        """
        if len(self.replay) < cfg.B_batch:
            return {}

        s, a, r, ns, done = self.replay.sample()

        # ── Critic update, Eq. (26) ───────────────────────────────────────────
        with torch.no_grad():
            a_next = self.actor_tgt(ns)
            q_next = self.critic_tgt(ns, a_next)
            y_t    = r + self.gamma * q_next * (1 - done)

        q_pred      = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_pred, y_t)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # ── Actor update, Eq. (27) ────────────────────────────────────────────
        a_curr     = self.actor(s)
        actor_loss = -self.critic(s, a_curr).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # ── Soft target updates, Eq. (28)-(29) ────────────────────────────────
        self._soft_update(self.actor_tgt, self.actor, self.tau)
        self._soft_update(self.critic_tgt, self.critic, self.tau)

        self.total_steps += 1
        return dict(critic_loss=float(critic_loss.item()),
                    actor_loss=float(actor_loss.item()))

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _soft_update(tgt: nn.Module, src: nn.Module,
                     tau: float = cfg.tau):
        """ς' ← τ ς + (1-τ) ς'  (Eq. 28-29)."""
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    @staticmethod
    def _hard_copy(tgt: nn.Module, src: nn.Module):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(sp.data)

    def save(self, path: str):
        torch.save({
            'actor':      self.actor.state_dict(),
            'critic':     self.critic.state_dict(),
            'actor_tgt':  self.actor_tgt.state_dict(),
            'critic_tgt': self.critic_tgt.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_tgt.load_state_dict(ckpt.get('actor_tgt', ckpt['actor']))
        self.critic_tgt.load_state_dict(ckpt.get('critic_tgt', ckpt['critic']))

    def training_state_dict(self) -> dict:
        """Serialize all mutable agent state needed to continue training."""
        return {
            "schema_version": 1,
            "agent_type": type(self).__name__,
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "replay": self.replay.state_dict(),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "noise_std": float(self.noise_std),
            "total_steps": int(self.total_steps),
        }

    def load_training_state_dict(self, state: dict):
        """Restore a state produced by :meth:`training_state_dict`."""
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported agent checkpoint schema.")
        if state.get("agent_type") != type(self).__name__:
            raise ValueError(
                "Agent checkpoint type mismatch: "
                f"expected {type(self).__name__}, got {state.get('agent_type')}."
            )
        saved_dims = (int(state["state_dim"]), int(state["action_dim"]))
        current_dims = (int(self.state_dim), int(self.action_dim))
        if saved_dims != current_dims:
            raise ValueError(
                f"Agent dimensions mismatch: expected {current_dims}, got {saved_dims}."
            )

        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_tgt.load_state_dict(state["actor_tgt"])
        self.critic_tgt.load_state_dict(state["critic_tgt"])
        self.opt_actor.load_state_dict(state["opt_actor"])
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.replay.load_state_dict(state["replay"])
        self.gamma = float(state["gamma"])
        self.tau = float(state["tau"])
        self.noise_std = float(state["noise_std"])
        self.total_steps = int(state["total_steps"])
