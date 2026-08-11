"""
simulation.py — Full training pipeline and figure generation.

Reproduces Figs. 4-14 from:
  Saikia et al., "Hybrid-RIS Empowered UAV-Assisted ISAC Systems:
  Transfer Learning-Based DRL," IEEE Trans. Commun., 2025.
"""
from config import cfg, temporary_system_configuration
from physical_hypotheses import apply_power_hypothesis
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time, random, hashlib, json
from contextlib import contextmanager
from dataclasses import asdict
import platform
import sys

from channel_model import ChannelSet
from sinr_rate import (
    active_tx_power,
    compute_rates,
    select_feasible_beamformers,
)
from environment import HRISUAVISACEnv
from ddpg_agent import DDPGAgent
from dtl_agent import DTLAgent
from checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
    validate_training_histories,
)
from source_integrity import source_model_digest, source_tree_digest
from ao_optimizer import run_ao
from paper_contract import (
    FIGURE_6_SYSTEM_CONFIGURATION,
    active_power_penalty,
    cascade_penalty_16c,
)
from experiment_contract import FIGURE_CONTRACTS
from experiment_protocol import TRAINING_SEED_PROTOCOL, derive_training_seed
from parameter_profiles import active_profile_metadata
from ambiguity_hypotheses import ddpg_comparator_scee



OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
os.makedirs(OUT, exist_ok=True)


class FeasibilityEvidence:
    """Count every physical-feasibility decision used by a reported figure."""

    def __init__(self):
        self.verified_realizations = 0
        self.failed_realizations = 0

    def require(self, condition, label):
        self.verified_realizations += 1
        if not bool(condition):
            self.failed_realizations += 1
            raise RuntimeError(str(label))

    def as_metadata(self):
        if self.verified_realizations <= 0:
            raise ValueError("Feasibility evidence has no verified realizations.")
        passed = self.verified_realizations - self.failed_realizations
        return {
            "physical_feasibility": {
                "rate": float(passed / self.verified_realizations),
                "verified_realizations": int(self.verified_realizations),
                "failed_realizations": int(self.failed_realizations),
                "verification": "fail_closed_per_generated_realization",
            }
        }

def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _save_figure_data(figure_number, x, curves, metadata=None):
    """Persist the exact plotted arrays and their computation provenance."""
    os.makedirs(OUT, exist_ok=True)
    x_array = np.asarray(x)
    archive = {"x": x_array}
    curve_manifest = {}
    curve_keys = {}
    for index, (label, values) in enumerate(curves.items()):
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(
                f"Figure {figure_number} curve {label!r} contains non-finite values."
            )
        key = f"curve_{index:02d}"
        archive[key] = array
        curve_manifest[str(label)] = _json_ready(array)
        curve_keys[str(label)] = key

    stem = f"fig{int(figure_number)}_data"
    npz_path = os.path.join(OUT, f"{stem}.npz")
    json_path = os.path.join(OUT, f"{stem}.json")
    np.savez(npz_path, **archive)
    manifest = {
        "figure": int(figure_number),
        "configuration_fingerprint": _configuration_fingerprint(),
        "parameter_profile": active_profile_metadata(),
        "x": _json_ready(x_array),
        "curves": curve_manifest,
        "npz_keys": curve_keys,
        "experiment_contract": asdict(FIGURE_CONTRACTS[figure_number])
        if figure_number in FIGURE_CONTRACTS else None,
        "metadata": _json_ready(metadata or {}),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {"npz": npz_path, "json": json_path}


def set_reproducibility(seed=cfg.random_seed):
    """Seed all stochastic libraries used by the reproduction."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configuration_fingerprint(payload=None):
    """Stable identifier for a trained configuration; contains no results."""
    data = {
        "M": cfg.M, "N": cfg.N, "U": cfg.U, "K": cfg.K,
        "L": cfg.L, "La": cfg.La, "Lp": cfg.Lp,
        "PT": cfg.PT, "PT_dB": cfg.PT_dB,
        "PT_dBm": cfg.PT_dBm, "PT_dBW": cfg.PT_dBW,
        "PT_reference": cfg.PT_reference,
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "noise_hypothesis_id": cfg.noise_hypothesis_id,
        "dtl_transfer_mode": cfg.dtl_transfer_mode,
        "dtl_transfer_schedule_id": cfg.dtl_transfer_schedule_id,
        "wfb_hypothesis_id": cfg.wfb_hypothesis_id,
        "ddpg_comparator_csi_hypothesis_id":
            cfg.ddpg_comparator_csi_hypothesis_id,
        "los_probability_hypothesis_id": cfg.los_probability_hypothesis_id,
        "channel_coherence_hypothesis_id":
            cfg.channel_coherence_hypothesis_id,
        "cross_function_interference_hypothesis_id":
            cfg.cross_function_interference_hypothesis_id,
        "sigma2_u": cfg.sigma2_u,
        "sigma2_k": cfg.sigma2_k,
        "sigma2_a": cfg.sigma2_a,
        "rho_nmse": cfg.rho_nmse,
        "R_min": cfg.R_min, "Psi_max": cfg.Psi_max,
        "P_a_max": cfg.P_a_max,
        "partial_csi_fraction": cfg.partial_csi_fraction,
        "source_seed": cfg.source_scene_seed,
        "target_seed": cfg.target_scene_seed,
        "parameter_profile_digest": active_profile_metadata()["digest"],
    }
    if payload:
        data.update(payload)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _training_checkpoint_path(tag):
    safe_tag = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(tag)
    ).strip("_")
    return os.path.join(OUT, "checkpoints", f"{safe_tag}_latest.pt")


@contextmanager
def _temporary_paper_power_label(paper_label_db):
    """Apply the active power hypothesis for one sweep point and restore it."""
    fields = (
        "power_hypothesis_id", "PT_dB", "PT_reference", "PT_display_unit",
        "PT", "PT_dBm", "PT_dBW", "power_hypothesis_metadata",
    )
    original = {field: getattr(cfg, field) for field in fields}
    try:
        apply_power_hypothesis(
            cfg, original["power_hypothesis_id"], float(paper_label_db)
        )
        yield
    finally:
        for field, value in original.items():
            setattr(cfg, field, value)


def _training_identity(env, tag, algorithm, requested_episodes,
                       source_agent=None, training_seed=None):
    requested_episodes = int(requested_episodes)
    if requested_episodes <= 0:
        raise ValueError("requested episode target must be positive")
    is_dtl = str(algorithm) == "DTL"
    if is_dtl and source_agent is None:
        raise ValueError("DTL checkpoint identity requires its source agent")
    if not is_dtl and source_agent is not None:
        raise ValueError("A non-DTL checkpoint cannot declare a source agent")
    return {
        "run_id": str(tag),
        "algorithm": str(algorithm),
        "configuration_fingerprint": _configuration_fingerprint({
            "sigma2_scee": env.sigma2,
            "use_capon": env.use_capon,
            "sensing_beamformer_mode": getattr(
                env,
                "sensing_beamformer_mode",
                "capon" if env.use_capon else "matched_filter",
            ),
            "csi_observation_fraction": env.csi_observation_fraction,
            "observe_instantaneous_csi": getattr(
                env,
                "observe_instantaneous_csi",
                bool(float(
                    cfg.rho_nmse if env.sigma2 is None else env.sigma2
                ) <= 0.0),
            ),
            "allocation_mode": env.allocation_mode,
        }),
        "parameter_profile_digest": active_profile_metadata()["digest"],
        "objective": env.objective_spec.as_dict(),
        "requested_episode_target": requested_episodes,
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "source_tree_digest": source_tree_digest(),
        "source_model_digest": (
            source_model_digest(source_agent) if is_dtl else "not_applicable"
        ),
        "environment_seed": getattr(env, "seed", None),
        "training_seed": None if training_seed is None else int(training_seed),
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "dtl_transfer_mode": (
            cfg.dtl_transfer_mode if is_dtl else "not_applicable"
        ),
        "dtl_transfer_schedule_id": (
            cfg.dtl_transfer_schedule_id if is_dtl else "not_applicable"
        ),
        "noise_hypothesis_id": cfg.noise_hypothesis_id,
        "wfb_hypothesis_id": cfg.wfb_hypothesis_id,
        "ddpg_comparator_csi_hypothesis_id":
            cfg.ddpg_comparator_csi_hypothesis_id,
        "los_probability_hypothesis_id": cfg.los_probability_hypothesis_id,
        "channel_coherence_hypothesis_id":
            cfg.channel_coherence_hypothesis_id,
        "cross_function_interference_hypothesis_id":
            cfg.cross_function_interference_hypothesis_id,
    }


@contextmanager
def _temporary_hris_configuration(L=None, La=None):
    overrides = {}
    if L is not None:
        overrides = {"L": int(L), "La": int(L // 2 if La is None else La)}
    with temporary_system_configuration(**overrides):
        yield


@contextmanager
def _temporary_system_configuration(M=None, L=None, La=None):
    overrides = {}
    if M is not None:
        overrides["M"] = int(M)
    if L is not None:
        overrides.update({"L": int(L), "La": int(L // 2 if La is None else La)})
    with temporary_system_configuration(**overrides):
        yield


def train_experiment_bundle(n_episodes,
                            L=None,
                            La=None,
                            variants=("perfect", "imperfect", "partial"),
                            variant_sigmas=None,
                            sensing_rate_min=None,
                            sensing_only_sinr_min=0.0,
                            include_wfb=False,
                            include_ddpg=True,
                            source_result=None,
                            resume=True,
                            checkpoint_every=cfg.checkpoint_every):
    """Train source, target DDPG, and target DTL runs for one paper setting.

    The source and target scenes are deliberately distinct. The source run is
    transfer input only; the plotted DDPG baseline is independently initialized
    and trained on the target task.
    """
    if cfg.source_scene_seed == cfg.target_scene_seed:
        raise ValueError("Source and target tasks must use different scene seeds.")

    variant_settings = {
        "perfect": (0.0, True, 1.0, True),
        "imperfect": (cfg.rho_nmse, True, 1.0, False),
        "partial": (cfg.rho_nmse, True, cfg.partial_csi_fraction, True),
    }
    if variant_sigmas:
        variant_settings.update({
            name: (float(sigma2), True, 1.0, float(sigma2) <= 0.0)
            for name, sigma2 in variant_sigmas.items()
        })
    unknown = set(variants) - set(variant_settings)
    if unknown:
        raise ValueError(f"Unknown CSI variants: {sorted(unknown)}")

    with _temporary_hris_configuration(L=L, La=La):
        fingerprint = _configuration_fingerprint({
            "episodes": int(n_episodes),
            "variants": list(variants),
            "variant_sigmas": variant_sigmas or {},
            "sensing_rate_min": (
                float(cfg.R_min) if sensing_rate_min is None
                else float(sensing_rate_min)
            ),
            "sensing_only_sinr_min": float(sensing_only_sinr_min),
            "include_wfb": bool(include_wfb),
        })
        if source_result is None:
            source_training_seed = derive_training_seed(
                cfg.source_scene_seed, "source"
            )
            source_env = HRISUAVISACEnv(
                sigma2_scee=cfg.rho_nmse,
                seed=cfg.source_scene_seed,
                use_capon=True,
                sensing_rate_min=sensing_rate_min,
                sensing_only_sinr_min=sensing_only_sinr_min,
            )
            source = train_ddpg(
                source_env, int(n_episodes), f"Source-{fingerprint}",
                resume=resume, checkpoint_every=checkpoint_every,
                training_seed=source_training_seed,
            )
        else:
            source = source_result

        target_ddpg = None
        if include_ddpg:
            target_training_seed = derive_training_seed(
                cfg.target_scene_seed, "target"
            )
            target_env = HRISUAVISACEnv(
                sigma2_scee=ddpg_comparator_scee(cfg),
                seed=cfg.target_scene_seed,
                use_capon=True,
                sensing_rate_min=sensing_rate_min,
                sensing_only_sinr_min=sensing_only_sinr_min,
            )
            target_ddpg = train_ddpg(
                target_env, int(n_episodes), f"Target-DDPG-{fingerprint}",
                resume=resume, checkpoint_every=checkpoint_every,
                training_seed=target_training_seed,
            )

        dtl_runs = {}
        for name in variants:
            (sigma2, use_capon, observation_fraction,
             observe_instantaneous_csi) = variant_settings[name]
            target_env = HRISUAVISACEnv(
                sigma2_scee=sigma2,
                seed=cfg.target_scene_seed,
                use_capon=use_capon,
                sensing_rate_min=sensing_rate_min,
                sensing_only_sinr_min=sensing_only_sinr_min,
                csi_observation_fraction=observation_fraction,
                observe_instantaneous_csi=observe_instantaneous_csi,
            )
            dtl_runs[name] = train_dtl(
                source["agent"], target_env, int(n_episodes),
                f"Target-DTL-{name}-{fingerprint}",
                resume=resume, checkpoint_every=checkpoint_every,
                training_seed=derive_training_seed(
                    cfg.target_scene_seed, "target"
                ),
            )

        if include_wfb:
            wfb_env = HRISUAVISACEnv(
                sigma2_scee=0.0,
                seed=cfg.target_scene_seed,
                sensing_beamformer_mode=cfg.wfb_beamformer_mode,
                sensing_rate_min=sensing_rate_min,
                sensing_only_sinr_min=sensing_only_sinr_min,
            )
            dtl_runs["wfb"] = train_dtl(
                source["agent"], wfb_env, int(n_episodes),
                f"Target-DTL-WFB-{fingerprint}",
                resume=resume, checkpoint_every=checkpoint_every,
                training_seed=derive_training_seed(
                    cfg.target_scene_seed, "target"
                ),
            )

        return {
            "fingerprint": fingerprint,
            "source": source,
            "ddpg": target_ddpg,
            "dtl": dtl_runs,
            "configuration": {
                "L": cfg.L, "La": cfg.La, "Lp": cfg.Lp,
                "source_seed": cfg.source_scene_seed,
                "target_seed": cfg.target_scene_seed,
                "episodes": int(n_episodes),
                "sensing_rate_min": (
                    float(cfg.R_min) if sensing_rate_min is None
                    else float(sensing_rate_min)
                ),
                "sensing_only_sinr_min": float(sensing_only_sinr_min),
            },
        }


class TrainingRegistry:
    """In-memory cache keyed only by configuration, never by desired results."""

    def __init__(self, resume=True, checkpoint_every=cfg.checkpoint_every):
        self._bundles = {}
        self._sources = {}
        self.resume = bool(resume)
        self.checkpoint_every = int(checkpoint_every)

    def get_or_train(self, **kwargs):
        key = _configuration_fingerprint(kwargs)
        if key not in self._bundles:
            source_key = _configuration_fingerprint({
                "source_only": True,
                "episodes": int(kwargs["n_episodes"]),
                "L": kwargs.get("L"),
                "La": kwargs.get("La"),
                "sensing_rate_min": float(
                    cfg.R_min if kwargs.get("sensing_rate_min") is None
                    else kwargs["sensing_rate_min"]
                ),
                "sensing_only_sinr_min": float(
                    kwargs.get("sensing_only_sinr_min", 0.0)
                ),
            })
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("resume", self.resume)
            call_kwargs.setdefault("checkpoint_every", self.checkpoint_every)
            if source_key in self._sources:
                call_kwargs["source_result"] = self._sources[source_key]
            self._bundles[key] = train_experiment_bundle(**call_kwargs)
            self._sources[source_key] = self._bundles[key]["source"]
        return self._bundles[key]


plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.8,
    "figure.dpi": 130,
})
COLORS = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#8c564b","#e377c2"]
MARKS  = ["o","s","^","D","v","P","*"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fresh_scene(seed=None):
    rng = np.random.default_rng(seed)
    pos_ues = [
        rng.uniform(
            [cfg.ue_x_min, cfg.ue_y_min, 0.0],
            [cfg.ue_x_max, cfg.ue_y_max, 0.0],
        )
        for _ in range(cfg.U)
    ]
    pos_tes = [rng.uniform([0,0,0],[cfg.x_max,cfg.y_max,0]) for _ in range(cfg.K)]
    pos_uav = np.array([(cfg.x_min+cfg.x_max)/2,
                        (cfg.y_min+cfg.y_max)/2, cfg.z_UAV], float)
    return pos_uav, pos_ues, pos_tes


def _rand_phases(rng=None):
    uniform = np.random.uniform if rng is None else rng.uniform
    theta = np.exp(1j * uniform(0, 2*np.pi, cfg.Lp))
    phi = (cfg.alpha_max / 2) * np.exp(1j * uniform(0, 2*np.pi, cfg.La))
    return theta, phi


def _physical_baseline_rate(mode, sigma2=None, n_eval=5, seed_base=100,
                            n_steps=None, strict_feasibility=True,
                            feasibility_evidence=None):
    """Evaluate a physical baseline on seeded mobile episodes."""
    if mode not in {"no_ris", "random_phase"}:
        raise ValueError(f"Unknown physical baseline mode: {mode}")
    steps = cfg.N_slots if n_steps is None else int(n_steps)
    if steps <= 0:
        raise ValueError("n_steps must be positive")

    episode_rates = []
    for episode in range(int(n_eval)):
        episode_seed = int(seed_base) + episode
        env = HRISUAVISACEnv(sigma2_scee=sigma2, seed=episode_seed)
        env.reset()
        phase_rng = np.random.default_rng(
            np.random.SeedSequence([episode_seed, 99])
        )
        step_rates = []
        for slot_index in range(steps):
            # Match the target-motion and channel-refresh order in env.step().
            env._advance_targets()
            env._refresh_channel(slot_time_s=(slot_index + 1) * cfg.delta_T)
            if mode == "random_phase":
                theta, phi = _rand_phases(phase_rng)
                label = "Random Phase"
            else:
                theta = np.zeros(cfg.Lp, dtype=complex)
                phi = np.zeros(cfg.La, dtype=complex)
                label = "No-RIS"
            C, T, beam_info = select_feasible_beamformers(
                env.ch, theta, phi
            )
            if strict_feasibility:
                _require_physical_feasibility(
                    env.ch, theta, phi, C, T, beam_info, label,
                    feasibility_evidence,
                )
            ru, _, _, _ = compute_rates(env.ch, theta, phi, C, T)
            step_rates.append(float(np.sum(ru)))
        episode_rates.append(float(np.mean(step_rates)))
    return float(np.mean(episode_rates))


def _no_ris_rate(sigma2=None, n_eval=5, seed_base=100, n_steps=None,
                 strict_feasibility=True, feasibility_evidence=None):
    """Benchmark the direct link on the common seeded episode protocol."""
    return _physical_baseline_rate(
        "no_ris", sigma2, n_eval, seed_base, n_steps, strict_feasibility,
        feasibility_evidence,
    )


def _random_phase_rate(sigma2=None, n_eval=5, seed_base=100, n_steps=None,
                       strict_feasibility=True, feasibility_evidence=None):
    """Benchmark random HRIS phases on the common seeded episode protocol."""
    return _physical_baseline_rate(
        "random_phase", sigma2, n_eval, seed_base, n_steps,
        strict_feasibility, feasibility_evidence,
    )


def _eval_agent(agent, sigma2, n_eval=5, seed_base=100,
                sensing_rate_min=None, sensing_only_sinr_min=0.0,
                csi_observation_fraction=1.0,
                observe_instantaneous_csi=None,
                strict_feasibility=True, feasibility_evidence=None):
    """Run an agent greedily and return its computed communication rate."""
    return _eval_agent_metrics(
        agent, sigma2, n_eval=n_eval, seed_base=seed_base,
        sensing_rate_min=sensing_rate_min,
        sensing_only_sinr_min=sensing_only_sinr_min,
        csi_observation_fraction=csi_observation_fraction,
        observe_instantaneous_csi=observe_instantaneous_csi,
        strict_feasibility=strict_feasibility,
        feasibility_evidence=feasibility_evidence,
    )["comm_rate"]


def _eval_agent_metrics(agent, sigma2, n_eval=5, seed_base=100,
                        sensing_rate_min=None,
                        sensing_only_sinr_min=0.0,
                        csi_observation_fraction=1.0,
                        observe_instantaneous_csi=None,
                        strict_feasibility=True,
                        feasibility_evidence=None):
    """Return measured communication and sensing rates from agent rollouts."""
    comm_rates = []
    sensing_rates = []
    for t in range(n_eval):
        env = HRISUAVISACEnv(
            sigma2_scee=sigma2, seed=seed_base + t,
            sensing_rate_min=sensing_rate_min,
            sensing_only_sinr_min=sensing_only_sinr_min,
            csi_observation_fraction=csi_observation_fraction,
            observe_instantaneous_csi=observe_instantaneous_csi,
        )
        s = env.reset()
        ep_comm = 0.0
        ep_sensing = 0.0
        done = False
        while not done:
            a = agent.select_action(s, explore=False)
            s, _, done, info = env.step(a)
            if strict_feasibility:
                message = (
                    "Refusing to report an infeasible policy rollout: "
                    f"episode_seed={seed_base + t}, step={env.t}."
                )
                if feasibility_evidence is None:
                    if not info["physical_constraints_feasible"]:
                        raise RuntimeError(message)
                else:
                    feasibility_evidence.require(
                        info["physical_constraints_feasible"], message
                    )
            ep_comm += float(info["sum_rate"])
            ep_sensing += float(info["sum_sensing"])
        comm_rates.append(ep_comm / cfg.N_slots)
        sensing_rates.append(ep_sensing / cfg.N_slots)
    return {
        "comm_rate": float(np.mean(comm_rates)),
        "sensing_rate": float(np.mean(sensing_rates)),
    }


def _require_physical_feasibility(
        ch, theta, phi, C, T, beam_info, label, feasibility_evidence=None):
    _, cascade_ok, _ = cascade_penalty_16c(ch, theta, cfg.Psi_max)
    p_ai = active_tx_power(ch, T, phi)
    _, active_ok = active_power_penalty(p_ai, cfg.P_a_max)
    feasible = bool(
        beam_info.get("allocation_feasible", False)
        and beam_info["sensing_constraint_feasible"]
        and cascade_ok
        and active_ok
    )
    message = f"Refusing to report infeasible {label} baseline realization."
    if feasibility_evidence is None:
        if not feasible:
            raise RuntimeError(message)
    else:
        feasibility_evidence.require(feasible, message)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  DRL Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_ddpg(env: HRISUAVISACEnv,
               n_episodes: int = 1000,
               tag: str = "DDPG",
               checkpoint_path=None,
               resume: bool = True,
               checkpoint_every: int = cfg.checkpoint_every,
               training_seed: int = None) -> dict:
    """Train paper DDPG with atomic, full-state episode checkpoints."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if training_seed is not None:
        set_reproducibility(training_seed)
    agent = DDPGAgent(env.state_dim, env.action_dim)
    checkpoint_path = checkpoint_path or _training_checkpoint_path(tag)
    if not resume and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    identity = _training_identity(
        env, tag, "DDPG", n_episodes, training_seed=training_seed
    )
    histories = {
        "rewards": [],
        "comm_rates": [],
        "sensing_rates": [],
        "joint_rates": [],
        "feasibility_rates": [],
    }
    start_episode = 1
    previous_elapsed = 0.0
    if resume and os.path.isfile(checkpoint_path):
        restored = load_training_checkpoint(
            checkpoint_path, agent, env, expected_identity=identity
        )
        histories.update(validate_training_histories(
            restored.histories,
            completed_episode=restored.completed_episode,
            required_keys=histories,
            requested_episodes=n_episodes,
        ))
        start_episode = restored.completed_episode + 1
        previous_elapsed = float(restored.extra.get("elapsed_seconds", 0.0))
        print(f"  Resumed {tag} after episode {restored.completed_episode}.", flush=True)

    print(f"\n{'-'*55}")
    print(f"  Training {tag}  ({n_episodes} episodes)")
    print(f"{'-'*55}")
    print("  (paper DDPG random initialization)", flush=True)
    t0 = time.time() - previous_elapsed

    for ep in range(start_episode, n_episodes + 1):
        state = env.reset()
        episode = {
            "reward": 0.0,
            "communication": 0.0,
            "sensing": 0.0,
            "joint": 0.0,
            "feasible": 0.0,
        }
        done = False
        while not done:
            action = agent.select_action(state, explore=True)
            agent.noise_std = max(0.05, cfg.noise_std * (cfg.noise_decay ** ep))
            next_state, reward, done, info = env.step(action)
            agent.replay.push(
                state, info["proposed_action"], reward, next_state, done
            )
            agent.update()
            state = next_state
            episode["reward"] += float(reward)
            episode["communication"] += float(info["communication_sum_rate"])
            episode["sensing"] += float(info["sensing_sum_rate"])
            episode["joint"] += float(info["joint_rate"])
            episode["feasible"] += float(info["physical_constraints_feasible"])

        histories["rewards"].append(episode["reward"] / cfg.N_slots)
        histories["comm_rates"].append(episode["communication"] / cfg.N_slots)
        histories["sensing_rates"].append(episode["sensing"] / cfg.N_slots)
        histories["joint_rates"].append(episode["joint"] / cfg.N_slots)
        histories["feasibility_rates"].append(episode["feasible"] / cfg.N_slots)

        if ep % 50 == 0:
            print(f"  Ep {ep:4d}/{n_episodes}  |  "
                  f"avg_reward={np.mean(histories['rewards'][-50:]):.3f}  |  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

        if ep % checkpoint_every == 0 or ep == n_episodes:
            save_training_checkpoint(
                checkpoint_path, agent, env, ep, histories, identity,
                extra={"elapsed_seconds": time.time() - t0},
            )

    return dict(agent=agent, **histories,
                checkpoint_path=checkpoint_path,
                elapsed_seconds=time.time() - t0)


def train_dtl(source_agent: DDPGAgent,
              env: HRISUAVISACEnv,
              n_episodes: int = 1000,
              tag: str = "DTL",
              checkpoint_path=None,
              resume: bool = True,
              checkpoint_every: int = cfg.checkpoint_every,
              training_seed: int = None) -> dict:
    """Fine-tune DTL with atomic, full-state episode checkpoints."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if training_seed is not None:
        set_reproducibility(training_seed)
    agent = DTLAgent(source_agent, env.state_dim, env.action_dim)
    checkpoint_path = checkpoint_path or _training_checkpoint_path(tag)
    if not resume and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    identity = _training_identity(
        env, tag, "DTL", n_episodes, source_agent=source_agent,
        training_seed=training_seed,
    )
    histories = {
        "rewards": [],
        "comm_rates": [],
        "sensing_rates": [],
        "joint_rates": [],
        "feasibility_rates": [],
    }
    start_episode = 1
    previous_elapsed = 0.0
    if resume and os.path.isfile(checkpoint_path):
        restored = load_training_checkpoint(
            checkpoint_path, agent, env, expected_identity=identity
        )
        histories.update(validate_training_histories(
            restored.histories,
            completed_episode=restored.completed_episode,
            required_keys=histories,
            requested_episodes=n_episodes,
        ))
        start_episode = restored.completed_episode + 1
        previous_elapsed = float(restored.extra.get("elapsed_seconds", 0.0))
        print(f"  Resumed {tag} after episode {restored.completed_episode}.", flush=True)

    print(f"\n{'-'*55}")
    print(f"  Training {tag}  ({n_episodes} episodes, transfer warm-start)")
    print(f"{'-'*55}")
    t0 = time.time() - previous_elapsed

    for ep in range(start_episode, n_episodes + 1):
        state = env.reset()
        episode = {
            "reward": 0.0,
            "communication": 0.0,
            "sensing": 0.0,
            "joint": 0.0,
            "feasible": 0.0,
        }
        done = False
        while not done:
            action = agent.select_action(state, explore=True)
            agent.noise_std = max(0.05, cfg.noise_std * (cfg.noise_decay ** ep))
            next_state, reward, done, info = env.step(action)
            agent.replay.push(
                state, info["proposed_action"], reward, next_state, done
            )
            agent.update()
            state = next_state
            episode["reward"] += float(reward)
            episode["communication"] += float(info["communication_sum_rate"])
            episode["sensing"] += float(info["sensing_sum_rate"])
            episode["joint"] += float(info["joint_rate"])
            episode["feasible"] += float(info["physical_constraints_feasible"])

        agent.decay_upsilon()
        histories["rewards"].append(episode["reward"] / cfg.N_slots)
        histories["comm_rates"].append(episode["communication"] / cfg.N_slots)
        histories["sensing_rates"].append(episode["sensing"] / cfg.N_slots)
        histories["joint_rates"].append(episode["joint"] / cfg.N_slots)
        histories["feasibility_rates"].append(episode["feasible"] / cfg.N_slots)

        if ep % 50 == 0:
            print(f"  Ep {ep:4d}/{n_episodes}  |  "
                  f"avg_reward={np.mean(histories['rewards'][-50:]):.3f}  |  "
                  f"upsilon={agent.upsilon:.3f}  |  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

        if ep % checkpoint_every == 0 or ep == n_episodes:
            save_training_checkpoint(
                checkpoint_path, agent, env, ep, histories, identity,
                extra={"elapsed_seconds": time.time() - t0},
            )

    return dict(agent=agent, **histories,
                checkpoint_path=checkpoint_path,
                elapsed_seconds=time.time() - t0)


def smooth(x, w=20):
    """Causal moving-average smoother — no edge drop at start or end."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1): i + 1].mean()
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 4 — Rewards vs Episodes
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_rewards_vs_episodes(n_ep=None, resume=True,
                             checkpoint_every=cfg.checkpoint_every):
    print("\n===  Fig. 4: Rewards vs Episodes  ===")
    n_ep = cfg.fig4_episodes if n_ep is None else int(n_ep)
    bundle = train_experiment_bundle(
        n_episodes=n_ep,
        variants=("perfect", "imperfect"),
        include_wfb=True,
        include_ddpg=True,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )
    res_ddpg = bundle["ddpg"]
    res_dtl_p = bundle["dtl"]["perfect"]
    res_dtl_i = bundle["dtl"]["imperfect"]
    res_dtl_wfb = bundle["dtl"]["wfb"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    eps = np.arange(n_ep)
    w   = max(n_ep // 20, 5)
    plot_stride = max(1, int(cfg.fig4_plot_interval))
    for rewards, label, c, m in [
        (res_dtl_p["comm_rates"],   "DTL-pCSI", COLORS[0], MARKS[0]),
        (res_dtl_wfb["comm_rates"], "DTL-WFB",  COLORS[2], MARKS[2]),
        (res_dtl_i["comm_rates"],   "DTL-iCSI", COLORS[3], MARKS[3]),
        (res_ddpg["comm_rates"],    "DDPG",     COLORS[1], MARKS[1]),
    ]:
        ax.plot(eps[::plot_stride],
                smooth(rewards, w)[::plot_stride],
                marker=m, markersize=5, label=label, color=c)

    ax.set_xlabel("Number of Episodes")
    ax.set_ylabel("Average Reward (bits/s/Hz)")
    ax.set_title("Fig. 4 — Training Rewards vs Episodes")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT}/fig4_rewards_vs_episodes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")

    _save_figure_data(
        4, eps[::plot_stride],
        {
            "DTL-pCSI": smooth(res_dtl_p["comm_rates"], w)[::plot_stride],
            "DTL-WFB": smooth(res_dtl_wfb["comm_rates"], w)[::plot_stride],
            "DTL-iCSI": smooth(res_dtl_i["comm_rates"], w)[::plot_stride],
            "DDPG": smooth(res_ddpg["comm_rates"], w)[::plot_stride],
        },
        metadata={
            "paper_panel_uses": "communication sum-rate reward",
            "reward_contains_penalties": False,
            "smoothing": f"causal moving average, window={w}",
            "source_policy_is_not_plotted": True,
        },
    )

    return {
        "source":  bundle["source"]["agent"],
        "ddpg":    res_ddpg["agent"],
        "dtl_p":   res_dtl_p["agent"],
        "dtl_i":   res_dtl_i["agent"],
        "dtl_wfb": res_dtl_wfb["agent"],
        "histories": {
            "DDPG":     res_ddpg["comm_rates"],
            "DTL-pCSI": res_dtl_p["comm_rates"],
            "DTL-iCSI": res_dtl_i["comm_rates"],
            "DTL-WFB":  res_dtl_wfb["comm_rates"],
        },
        "training_runs": {
            40: {
                "ddpg": res_ddpg["agent"],
                "dtl": res_dtl_i["agent"],
                "times": {
                    "DDPG": res_ddpg["elapsed_seconds"],
                    "DTL": res_dtl_i["elapsed_seconds"],
                },
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 5 — R_u vs AO Iterations
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_ao_convergence(scene_seed=42, ao_seed=5_000):
    """Fig. 5 - AO sum-rate vs iterations for L = 10, 20, 30."""
    print("\n=== Fig. 5: AO Convergence vs Iterations ===")
    pos_uav, pos_ues, pos_tes = _fresh_scene(int(scene_seed))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = {}
    feasibility = FeasibilityEvidence()

    for L_val, c, ls in [(30, COLORS[0], "-"),
                          (20, COLORS[1], "--"),
                          (10, COLORS[2], ":")]:
        with _temporary_hris_configuration(L=L_val, La=L_val // 2):
            res = run_ao(
                pos_uav, pos_ues, pos_tes,
                sigma2=cfg.rho_nmse, n_iter=cfg.ao_max_iter,
                fix_uav=False, stop_on_convergence=False,
                seed=int(ao_seed),
            )
            feasibility.require(
                res["is_feasible"],
                f"Fig. 5 AO solution is infeasible for L={L_val}.",
            )
            for iteration, is_feasible in enumerate(res["history_feasible"]):
                feasibility.require(
                    is_feasible,
                    "Fig. 5 AO history contains an infeasible iterate for "
                    f"L={L_val}, iteration={iteration}.",
                )
            history = res["history_sum_rate"][1:cfg.ao_max_iter + 1]
            curves[f"L={L_val}"] = history

        ax.plot(range(1, cfg.ao_max_iter + 1), history,
                color=c, linestyle=ls, marker=MARKS[0],
                markersize=5, label=f"L={L_val}")

    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("R_u (bits/s/Hz)")
    ax.set_title("Fig. 5 - AO Convergence vs Iterations")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT}/fig5_ao_convergence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        5, range(1, cfg.ao_max_iter + 1), curves,
        metadata={
            **feasibility.as_metadata(),
            "method": "SciPy numerical AO surrogate",
            "early_stopping": False,
            "uav_fixed": False,
            "trajectory_disclosure": (
                "one-step numerical trajectory block, not the unavailable "
                "author CVX trajectory solver"
            ),
        },
    )
    return {"x": list(range(1, cfg.ao_max_iter + 1)), "curves": curves}
# ==============================================================================
#  Fig. 6 - R_u vs Transmit Power P_T
# ==============================================================================

def fig6_rate_vs_power(training_registry=None, n_ep=None, n_eval=4,
                       evaluation_seed_base=6_000, scene_seed=6,
                       ao_seed_base=6_000):
    """Fig. 6 - Sum rate vs P_T using MC compute_rates (matches paper)."""
    print("\n=== Fig. 6: R_u vs Transmit Power ===")
    registry = TrainingRegistry() if training_registry is None else training_registry
    PT_dB_range = np.arange(10, 31, 5)

    # Paper Table III + Fig. 6 caption: M=4, L=20 (La=10, Lp=10).
    pos_uav, pos_ues, pos_tes = _fresh_scene(int(scene_seed))
    n_mc = 6
    feasibility = FeasibilityEvidence()

    def sweep_pt(PT_dB, sigma2, use_ao=False):
        with temporary_system_configuration(**FIGURE_6_SYSTEM_CONFIGURATION):
            with _temporary_paper_power_label(PT_dB):
                rates = []
                for mc_index in range(n_mc):
                    seed = int(ao_seed_base) + mc_index
                    if use_ao:
                        res = run_ao(
                            pos_uav, pos_ues, pos_tes, sigma2=sigma2,
                            n_iter=cfg.fig6_ao_iter, fix_uav=False, seed=seed,
                        )
                        feasibility.require(
                            res["is_feasible"],
                            "Fig. 6 AO solution is infeasible at "
                            f"P_T={PT_dB} dB.",
                        )
                        rates.append(res["sum_rate"])
                    else:
                        ch = ChannelSet(
                            pos_uav, pos_ues, pos_tes, sigma2, seed=seed
                        )
                        theta, phi = _rand_phases(
                            np.random.default_rng([seed, 99])
                        )
                        C, T, beam_info = select_feasible_beamformers(
                            ch, theta, phi
                        )
                        _require_physical_feasibility(
                            ch, theta, phi, C, T, beam_info, "Fig. 6",
                            feasibility,
                        )
                        ru, _, _, _ = compute_rates(ch, theta, phi, C, T)
                        rates.append(float(np.sum(ru)))
                return float(np.mean(rates))

    # P_T is not part of the policy observation. Train a separately keyed
    # target bundle at every power point instead of silently reusing the
    # baseline policy outside its training task.
    train_ep = cfg.fig6_train_episodes if n_ep is None else int(n_ep)
    dtl_p, dtl_i, dtl_par, ddpg_r, ao_p, ao_i = ([] for _ in range(6))
    baseline_bundle = None
    baseline_power_label = cfg.PT_dB
    for PT_dB in PT_dB_range:
        with temporary_system_configuration(**FIGURE_6_SYSTEM_CONFIGURATION):
            with _temporary_paper_power_label(PT_dB):
                bundle6 = registry.get_or_train(
                    n_episodes=train_ep,
                    L=20,
                    La=10,
                    variants=("perfect", "imperfect", "partial"),
                    include_ddpg=True,
                )
                if int(PT_dB) == int(baseline_power_label):
                    baseline_bundle = bundle6

                dtl_p.append(_eval_agent(
                    bundle6["dtl"]["perfect"]["agent"], 0.0,
                    n_eval=n_eval, seed_base=int(evaluation_seed_base),
                    feasibility_evidence=feasibility,
                ))
                dtl_i.append(_eval_agent(
                    bundle6["dtl"]["imperfect"]["agent"], cfg.rho_nmse,
                    n_eval=n_eval, seed_base=int(evaluation_seed_base),
                    feasibility_evidence=feasibility,
                ))
                dtl_par.append(_eval_agent(
                    bundle6["dtl"]["partial"]["agent"], cfg.rho_nmse,
                    n_eval=n_eval, seed_base=int(evaluation_seed_base),
                    csi_observation_fraction=cfg.partial_csi_fraction,
                    observe_instantaneous_csi=True,
                    feasibility_evidence=feasibility,
                ))
                ddpg_r.append(_eval_agent(
                    bundle6["ddpg"]["agent"], ddpg_comparator_scee(cfg),
                    n_eval=n_eval, seed_base=int(evaluation_seed_base),
                    feasibility_evidence=feasibility,
                ))

        print(f"  sweeping AO at P_T={PT_dB} dB ...")
        ao_p.append(sweep_pt(PT_dB, 0.0, use_ao=True))
        ao_i.append(sweep_pt(PT_dB, cfg.rho_nmse, use_ao=True))

    if baseline_bundle is None:
        raise RuntimeError("Fig. 6 did not retain the baseline-power training run.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for data, label, c, m in [
        (dtl_p,   "DTL-pCSI",   COLORS[0], MARKS[0]),
        (ddpg_r,  "DDPG",       COLORS[5], MARKS[5]),
        (ao_p,    "AO-pCSI",    COLORS[2], MARKS[2]),
        (dtl_i,   "DTL-iCSI",   COLORS[3], MARKS[3]),
        (ao_i,    "AO-iCSI",    COLORS[4], MARKS[4]),
        (dtl_par, "DTL-ParCSI", COLORS[1], MARKS[1]),
    ]:
        ax.plot(PT_dB_range, data, marker=m,
                markersize=5, label=label, color=c)

    ax.set_xlabel("P_T (dB)")
    ax.set_ylabel("R_u (bits/s/Hz)")
    ax.set_title("Fig. 6 - Sum Rate vs Transmit Power  [M=4, L=20]")
    ax.legend(ncol=2)
    fig.tight_layout()
    path = f"{OUT}/fig6_rate_vs_power.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        6, PT_dB_range,
        {
            "DTL-pCSI": dtl_p, "DDPG": ddpg_r, "AO-pCSI": ao_p,
            "DTL-iCSI": dtl_i, "AO-iCSI": ao_i,
            "DTL-ParCSI": dtl_par,
        },
        metadata={
            **feasibility.as_metadata(),
            "power_hypothesis_id": cfg.power_hypothesis_id,
            "power_conversion": (
                f"paper dB label interpreted as {cfg.PT_display_unit} "
                f"with reference {cfg.PT_reference}"
            ),
            "ao_disclosure": "SciPy numerical surrogate, only on AO-labeled curves",
        },
    )
    return {
        20: {
            "ddpg": baseline_bundle["ddpg"]["agent"],
            "dtl": baseline_bundle["dtl"]["imperfect"]["agent"],
            "times": {
                "DDPG": baseline_bundle["ddpg"]["elapsed_seconds"],
                "DTL": baseline_bundle["dtl"]["imperfect"]["elapsed_seconds"],
            },
        },
    }


# ==============================================================================
#  Fig. 7 - R_u vs Number of HRIS Elements
# ==============================================================================

def fig7_rate_vs_L(training_registry=None, n_ep=None, n_eval=5,
                   evaluation_seed_base=7_000):
    """Fig. 7 - trained HRIS policies and physical baseline methods vs L."""
    print("\n=== Fig. 7: R_u vs HRIS Elements ===")
    registry = TrainingRegistry() if training_registry is None else training_registry
    n_ep = cfg.E_episodes if n_ep is None else int(n_ep)
    L_range = [20, 30, 40, 50, 60]
    curves = {
        "HRIS-pCSI": [], "HRIS-iCSI": [], "HRIS-ParCSI": [],
        "Passive RIS": [], "Random Phase": [], "No-RIS": [],
    }
    feasibility = FeasibilityEvidence()

    for L_val in L_range:
        active_count = L_val // 2
        hris_bundle = registry.get_or_train(
            n_episodes=n_ep, L=L_val, La=active_count,
            variants=("perfect", "imperfect", "partial"),
            include_ddpg=False,
        )
        with _temporary_hris_configuration(L=L_val, La=active_count):
            curves["HRIS-pCSI"].append(_eval_agent(
                hris_bundle["dtl"]["perfect"]["agent"], 0.0,
                n_eval=n_eval, seed_base=int(evaluation_seed_base),
                feasibility_evidence=feasibility,
            ))
            curves["HRIS-iCSI"].append(_eval_agent(
                hris_bundle["dtl"]["imperfect"]["agent"], cfg.rho_nmse,
                n_eval=n_eval, seed_base=int(evaluation_seed_base),
                feasibility_evidence=feasibility,
            ))
            curves["HRIS-ParCSI"].append(_eval_agent(
                hris_bundle["dtl"]["partial"]["agent"], cfg.rho_nmse,
                n_eval=n_eval, seed_base=int(evaluation_seed_base),
                csi_observation_fraction=cfg.partial_csi_fraction,
                observe_instantaneous_csi=True,
                feasibility_evidence=feasibility,
            ))
            curves["Random Phase"].append(_random_phase_rate(
                cfg.rho_nmse, n_eval,
                seed_base=int(evaluation_seed_base),
                feasibility_evidence=feasibility,
            ))
            curves["No-RIS"].append(_no_ris_rate(
                cfg.rho_nmse, n_eval,
                seed_base=int(evaluation_seed_base),
                feasibility_evidence=feasibility,
            ))

        passive_bundle = registry.get_or_train(
            n_episodes=n_ep, L=L_val, La=0,
            variants=("imperfect",), include_ddpg=False,
        )
        with _temporary_hris_configuration(L=L_val, La=0):
            curves["Passive RIS"].append(_eval_agent(
                passive_bundle["dtl"]["imperfect"]["agent"], cfg.rho_nmse,
                n_eval=n_eval, seed_base=int(evaluation_seed_base),
                feasibility_evidence=feasibility,
            ))

    hris_p = curves["HRIS-pCSI"]
    hris_i = curves["HRIS-iCSI"]
    hris_par = curves["HRIS-ParCSI"]
    passive = curves["Passive RIS"]
    rand_ph = curves["Random Phase"]
    no_ris_r = curves["No-RIS"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for data, label, c, m in [
        (hris_p,   "HRIS-pCSI",   COLORS[0], MARKS[0]),
        (hris_i,   "HRIS-iCSI",   COLORS[1], MARKS[1]),
        (hris_par, "HRIS-ParCSI", COLORS[2], MARKS[2]),
        (passive,  "Passive RIS",  COLORS[3], MARKS[3]),
        (rand_ph,  "Random Phase", COLORS[4], MARKS[4]),
        (no_ris_r, "No-RIS",       COLORS[5], MARKS[5]),
    ]:
        ax.plot(L_range, data, marker=m,
                markersize=6, label=label, color=c)

    ax.set_xlabel("Number of HRIS Elements L")
    ax.set_ylabel("R_u (bits/s/Hz)")
    ax.set_title("Fig. 7 - Sum Rate vs Number of HRIS Elements")
    ax.legend(ncol=2)
    fig.tight_layout()
    path = f"{OUT}/fig7_rate_vs_L.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        7, L_range, curves,
        metadata=feasibility.as_metadata(),
    )
    return {"x": L_range, "curves": curves, "training_registry": registry}


#  Fig. 8 — R_u vs SCEE (ρ)
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_rate_vs_scee(training_registry=None, n_ep=None, n_eval=5,
                      evaluation_seed_base=8_000):
    print("\n===  Fig. 8: R_u vs SCEE  ===")
    registry = TrainingRegistry() if training_registry is None else training_registry
    n_ep = cfg.E_episodes if n_ep is None else int(n_ep)
    rho_range = np.arange(0.0, 1.01, 0.2)
    feasibility = FeasibilityEvidence()

    def scee_sweep(L_val, La):
        rates = []
        for rho in rho_range:
            sigma2 = float(rho)
            bundle = registry.get_or_train(
                n_episodes=n_ep, L=L_val, La=La,
                variants=("sweep",), variant_sigmas={"sweep": sigma2},
                include_ddpg=False,
            )
            with _temporary_hris_configuration(L=L_val, La=La):
                rates.append(_eval_agent(
                    bundle["dtl"]["sweep"]["agent"], sigma2,
                    n_eval=n_eval,
                    seed_base=int(evaluation_seed_base),
                    feasibility_evidence=feasibility,
                ))
        return rates

    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = {}
    for L_val, c, ls in [(40, COLORS[0], "-"), (20, COLORS[1], "--")]:
        hris_r = scee_sweep(L_val, La=L_val // 2)
        passive_r = scee_sweep(L_val, La=0)
        curves[f"HRIS, L={L_val}"] = hris_r
        curves[f"Passive RIS, L={L_val}"] = passive_r
        ax.plot(rho_range, hris_r,    color=c, linestyle=ls,
                marker=MARKS[0], markersize=5, label=f"HRIS, $L={L_val}$")
        ax.plot(rho_range, passive_r, color=c, linestyle=":",
                marker=MARKS[2], markersize=5, label=f"Passive RIS, $L={L_val}$")

    ax.set_xlabel("SCEE Parameter $\\varrho$")
    ax.set_ylabel("$R_u$ (bits/s/Hz)")
    ax.set_title("Fig. 8 — Sum Rate vs SCEE")
    ax.legend(ncol=2)
    fig.tight_layout()
    path = f"{OUT}/fig8_rate_vs_scee.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        8, rho_range, curves,
        metadata=feasibility.as_metadata(),
    )
    return {"x": rho_range.tolist(), "curves": curves,
            "training_registry": registry}


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 9 — R_u vs Active Element Budget ν
# ═══════════════════════════════════════════════════════════════════════════════

def fig9_active_budget(training_registry=None, n_ep=None, n_eval=5,
                       evaluation_seed_base=9_000):
    print("\n===  Fig. 9: R_u vs Active Element Budget  ===")
    registry = TrainingRegistry() if training_registry is None else training_registry
    n_ep = cfg.E_episodes if n_ep is None else int(n_ep)
    nu_vals = [0.1, 0.3, 0.5]    # fraction of L allocated to active
    L_vals  = [40, 30]

    results = {}   # (L, nu) -> rate
    feasibility = FeasibilityEvidence()
    for L_val in L_vals:
        for nu in nu_vals:
            active_count = int(round(L_val * nu))
            bundle = registry.get_or_train(
                n_episodes=n_ep, L=L_val, La=active_count,
                variants=("imperfect",), include_ddpg=False,
            )
            with _temporary_hris_configuration(L=L_val, La=active_count):
                results[(L_val, nu)] = _eval_agent(
                    bundle["dtl"]["imperfect"]["agent"], cfg.rho_nmse,
                    n_eval=n_eval,
                    seed_base=int(evaluation_seed_base),
                    feasibility_evidence=feasibility,
                )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x       = np.arange(len(L_vals))
    width   = 0.22
    offsets = np.linspace(-(len(nu_vals)-1)*width/2,
                           (len(nu_vals)-1)*width/2, len(nu_vals))
    for i, nu in enumerate(nu_vals):
        vals = [results[(L, nu)] for L in L_vals]
        ax.bar(x + offsets[i], vals, width,
               label=f"$\\nu={nu}$", color=COLORS[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$L={L}$" for L in L_vals])
    ax.set_ylabel("$R_u$ (bits/s/Hz)")
    ax.set_title("Fig. 9 — Sum Rate vs Active Element Budget $\\nu$")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT}/fig9_active_budget.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        9, L_vals,
        {f"nu={nu}": [results[(L_val, nu)] for L_val in L_vals]
         for nu in nu_vals},
        metadata=feasibility.as_metadata(),
    )
    return {"L": L_vals, "nu": nu_vals, "rates": results,
            "training_registry": registry}


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 10 — UAV Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_uav_trajectory(agent=None):
    print("\n===  Fig. 10: UAV Trajectory  ===")

    if agent is None:
        print("  Skipping Fig. 10: no trained agent trajectory is available.")
        return None

    env_agent = HRISUAVISACEnv(sigma2_scee=cfg.rho_nmse, seed=12)
    state = env_agent.reset()
    traj_agent = [env_agent.uav_pos[:2].copy()]
    done = False
    while not done:
        action = agent.select_action(state, explore=False)
        state, _, done, info = env_agent.step(action)
        traj_agent.append(info["uav_pos"][:2].copy())

    rng = np.random.default_rng(77)
    env_random = HRISUAVISACEnv(sigma2_scee=cfg.rho_nmse, seed=12)
    state = env_random.reset()
    traj_random = [env_random.uav_pos[:2].copy()]
    done = False
    while not done:
        action = agent.select_action(state, explore=False)
        random_phase = rng.uniform(-1.0, 1.0, cfg.Lp + cfg.La)
        action[:cfg.Lp + cfg.La] = random_phase
        state, _, done, info = env_random.step(action)
        traj_random.append(info["uav_pos"][:2].copy())

    th = np.array(traj_agent)
    tr = np.array(traj_random)
    plot_interval = max(1, int(cfg.fig10_plot_interval))
    th_plot = th[::plot_interval]
    tr_plot = tr[::plot_interval]
    ues = np.array(env_agent.pos_ues)
    ues_projected = ues.copy()
    ues_projected[:, 2] = cfg.z_UAV

    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(th_plot[:, 0], th_plot[:, 1],
            cfg.z_UAV * np.ones(len(th_plot)),
            "-o", markersize=4, color=COLORS[0], label="DTL trajectory")
    ax.plot(tr_plot[:, 0], tr_plot[:, 1],
            cfg.z_UAV * np.ones(len(tr_plot)),
            "--s", markersize=4, color=COLORS[1], label="Random phase")
    ax.scatter(ues[:, 0], ues[:, 1], cfg.z_UAV * np.ones(len(ues)),
               marker="x", s=80, color=COLORS[2], zorder=5, label="Users")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_zlim(cfg.z_UAV - 1.0, cfg.z_UAV + 1.0)
    ax.set_title("Fig. 10 — HRIS-Aided UAV Trajectory")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT}/fig10_uav_trajectory.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        10, range(len(th_plot)),
        {"DTL trajectory XY": th_plot,
         "Random-phase trajectory XY": tr_plot,
         "UE markers projected XYZ": ues_projected},
        metadata={"altitude_m": cfg.z_UAV,
                  "UE_ground_positions_XYZ": ues,
                  "source_slots": len(th),
                  "display_interval_slots": plot_interval},
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 11 — Beampattern Gain
# ═══════════════════════════════════════════════════════════════════════════════

def fig11_beampattern():
    print("\n===  Fig. 11: Beampattern Gain  ===")
    from channel_model import upa_steering
    from sinr_rate import capon_beamformer

    pos_uav, _, pos_tes = _fresh_scene(11)
    target_directions = []
    for target in pos_tes:
        direction = np.asarray(target) - pos_uav
        direction /= np.linalg.norm(direction) + 1e-12
        target_directions.append(direction)
    steering = [
        upa_steering(cfg.Lax, cfg.Lay, direction[0], direction[1])
        for direction in target_directions
    ]
    a_tgt = steering[0]
    R_y = cfg.rho_nmse * np.eye(cfg.La, dtype=complex)
    for other in steering[1:]:
        R_y += np.outer(other, other.conj())
    T_cap = capon_beamformer(R_y, a_tgt)
    T_cap /= np.linalg.norm(T_cap) + 1e-12

    phi_range = np.linspace(np.deg2rad(-80.0), np.deg2rad(0.0), 60)
    eta_range = np.linspace(np.deg2rad(0.0), np.deg2rad(80.0), 40)
    PHI, ETA = np.meshgrid(phi_range, eta_range)

    def beam_gain(phi_s, eta_s, T_w):
        pb1 = np.sin(phi_s) * np.cos(eta_s)
        pb2 = np.sin(phi_s) * np.sin(eta_s)
        a   = upa_steering(cfg.Lax, cfg.Lay, pb1, pb2)
        return float(np.abs(T_w.conj() @ a)**2)

    # iCSI beampattern
    gain_i = np.zeros_like(PHI)
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            gain_i[i, j] = beam_gain(PHI[i,j], ETA[i,j], T_cap)
    gain_i /= (gain_i.max() + 1e-12)

    fig = plt.figure(figsize=(8, 5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(np.degrees(PHI), np.degrees(ETA), gain_i,
                    cmap="viridis", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Azimuth $\\varphi$ (°)")
    ax.set_ylabel("Elevation $\\eta$ (°)")
    ax.set_zlabel("Beampattern Gain")
    ax.set_title("Fig. 11 — Beampattern Gain (iCSI)")
    fig.tight_layout()
    path = f"{OUT}/fig11_beampattern.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        11, np.degrees(phi_range), {"iCSI beampattern": gain_i},
        metadata={"elevation_degrees": np.degrees(eta_range)},
    )
    return {"azimuth_degrees": np.degrees(phi_range),
            "elevation_degrees": np.degrees(eta_range), "gain": gain_i}


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 12 — Transmission Time: DTL vs DDPG
# ═══════════════════════════════════════════════════════════════════════════════

def fig12_transmission_time(training_runs=None, n_ep=None,
                            training_registry=None):
    """Plot local measured training time; values are hardware-dependent."""
    print("\n===  Fig. 12: Transmission Time  ===")
    n_ep = cfg.E_episodes if n_ep is None else int(n_ep)
    runs = {} if training_runs is None else dict(training_runs)
    registry = TrainingRegistry() if training_registry is None else training_registry

    for L_val in [40, 30]:
        if L_val in runs:
            continue
        bundle = registry.get_or_train(
            n_episodes=n_ep, L=L_val, La=L_val // 2,
            variants=("imperfect",), include_ddpg=True,
        )
        ddpg = bundle["ddpg"]
        dtl = bundle["dtl"]["imperfect"]
        runs[L_val] = {
            "ddpg": ddpg["agent"],
            "dtl": dtl["agent"],
            "times": {
                "DDPG": ddpg["elapsed_seconds"],
                "DTL": dtl["elapsed_seconds"],
            },
        }

    L_vals = [40, 30]
    algorithms = ["DTL", "DDPG"]
    x = np.arange(2)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, L_val in enumerate(L_vals):
        values = [runs[L_val]["times"][name] for name in algorithms]
        ax.bar(x + (index - 0.5) * width, values, width,
               label=f"$L={L_val}$", color=COLORS[3 - index])
    ax.set_xticks(x, algorithms)
    ax.set_yscale("log")
    ax.set_xlabel("Data size")
    ax.set_ylabel("Transmission time (secs)")
    ax.set_title("Fig. 12 — Transmission Time for DTL and DDPG")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT}/fig12_transmission_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        12, algorithms,
        {f"L={L_val}": [runs[L_val]["times"][name] for name in algorithms]
         for L_val in L_vals},
        metadata={
            "measurement": "local Python wall-clock training time",
            "disclosure": "hardware-dependent; not directly comparable to paper bars",
            "episodes": n_ep,
        },
    )
    return runs


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 13 — Communication & Target Rate vs δ (sensing SINR threshold)
# ═══════════════════════════════════════════════════════════════════════════════

def fig13_rate_vs_delta(training_registry=None, n_ep=None, n_eval=5,
                        evaluation_seed_base=13_000):
    print("\n===  Fig. 13: Comm. & Target Rate vs delta  ===")
    registry = TrainingRegistry() if training_registry is None else training_registry
    n_ep = cfg.E_episodes if n_ep is None else int(n_ep)
    delta_range = np.arange(0, 21, 5)   # minimum sensing SINR, linear
    L_vals = [40, 30, 20]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    curves = {}
    feasibility = FeasibilityEvidence()

    for L_val, c in zip(L_vals, COLORS):
        ru_list, rk_list = [], []
        for delta in delta_range:
            threshold = float(delta)
            sensing_rate_min = float(np.log2(1.0 + threshold))
            active_count = L_val // 2
            bundle = registry.get_or_train(
                n_episodes=n_ep, L=L_val, La=active_count,
                variants=("imperfect",),
                sensing_rate_min=sensing_rate_min,
                include_ddpg=False,
            )
            with _temporary_hris_configuration(L=L_val, La=active_count):
                metrics = _eval_agent_metrics(
                    bundle["dtl"]["imperfect"]["agent"], cfg.rho_nmse,
                    n_eval=n_eval,
                    seed_base=int(evaluation_seed_base),
                    sensing_rate_min=sensing_rate_min,
                    feasibility_evidence=feasibility,
                )
            ru_list.append(metrics["comm_rate"])
            rk_list.append(metrics["sensing_rate"])

        curves[f"R_u, L={L_val}"] = ru_list
        if L_val in [30, 20]:
            curves[f"R_k, L={L_val}"] = rk_list

        idx = L_vals.index(L_val)
        ax1.plot(delta_range, ru_list, marker=MARKS[idx], markersize=5,
                 label=f"$R_u$, $L={L_val}$", color=c)
        if L_val in [30, 20]:
            ax2.plot(delta_range, rk_list, marker=MARKS[idx + 2],
                     markersize=5, linestyle="--",
                     label=f"$R_k$, $L={L_val}$", color=COLORS[idx + 3])

    ax1.set_xlabel("Minimum SINR required for the target $\\delta$")
    ax1.set_ylabel("$R_u$ (bits/s/Hz)")
    ax2.set_ylabel("$R_k$ (bits/s/Hz)")
    ax1.set_title("Fig. 13 — Communication and Target Rate vs $\\delta$")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], ncol=2)

    fig.tight_layout()
    path = f"{OUT}/fig13_rate_vs_delta.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        13, delta_range, curves,
        metadata={
            **feasibility.as_metadata(),
            "delta_units": "linear full target SINR from Eq. (14)",
            "rate_floor_conversion": "R_k >= log2(1 + delta)",
        },
    )
    return {"x": delta_range.tolist(), "curves": curves,
            "training_registry": registry}


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 14 — Communication Rate vs δ̄ (min sensing power)
# ═══════════════════════════════════════════════════════════════════════════════

def fig14_rate_vs_delta_bar(training_registry=None, n_ep=None, n_eval=4,
                            evaluation_seed_base=14_000):
    print("\n===  Fig. 14: Comm. Rate vs delta-bar (sensing power)  ===")
    delta_bar_range = np.arange(0, 21, 5)
    L_vals = [30, 20]
    n_ep = cfg.E_episodes if n_ep is None else int(n_ep)
    registry = TrainingRegistry() if training_registry is None else training_registry
    curves = {}
    feasibility = FeasibilityEvidence()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for L_index, L_val in enumerate(L_vals):
        dtl_rates = []
        ddpg_rates = []
        active_count = L_val // 2
        for delta_bar_db in delta_bar_range:
            sensing_only_sinr_min = 10 ** (float(delta_bar_db) / 10.0)
            bundle = registry.get_or_train(
                n_episodes=n_ep, L=L_val, La=active_count,
                variants=("imperfect",), include_ddpg=True,
                sensing_only_sinr_min=sensing_only_sinr_min,
            )
            with _temporary_hris_configuration(L=L_val, La=active_count):
                dtl_rates.append(_eval_agent(
                    bundle["dtl"]["imperfect"]["agent"], cfg.rho_nmse,
                    n_eval=n_eval,
                    seed_base=int(evaluation_seed_base),
                    sensing_only_sinr_min=sensing_only_sinr_min,
                    feasibility_evidence=feasibility,
                ))
                ddpg_rates.append(_eval_agent(
                    bundle["ddpg"]["agent"], ddpg_comparator_scee(cfg),
                    n_eval=n_eval,
                    seed_base=int(evaluation_seed_base),
                    sensing_only_sinr_min=sensing_only_sinr_min,
                    feasibility_evidence=feasibility,
                ))

        curves[f"DTL, L={L_val}"] = dtl_rates
        curves[f"DDPG, L={L_val}"] = ddpg_rates
        ax.plot(delta_bar_range, dtl_rates,
                color=COLORS[2 * L_index], linestyle="-",
                marker=MARKS[L_index], markersize=5,
                label=f"DTL, $L={L_val}$")
        ax.plot(delta_bar_range, ddpg_rates,
                color=COLORS[2 * L_index + 1], linestyle="--",
                marker=MARKS[L_index + 2], markersize=5,
                label=f"DDPG, $L={L_val}$")

    ax.set_xlabel("Min Sensing Power $\\bar{{\\delta}}$ (dB)")
    ax.set_ylabel("$R_u$ (bits/s/Hz)")
    ax.set_title("Fig. 14 — Communication Rate vs $\\bar{\\delta}$")
    ax.legend(ncol=2)
    fig.tight_layout()
    path = f"{OUT}/fig14_rate_vs_delta_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    _save_figure_data(
        14, delta_bar_range, curves,
        metadata={
            **feasibility.as_metadata(),
            "threshold_units": "dB converted to linear sensing-only SINR",
            "constraint": "Remark 1, communication interference excluded",
        },
    )
    return {"x": delta_bar_range.tolist(), "curves": curves,
            "training_registry": registry}


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary Dashboard (all figures in one grid)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_summary():
    """Assemble individual PNG files into one summary image."""
    import matplotlib.image as mpimg
    figs_order = [
        "fig4_rewards_vs_episodes.png",
        "fig5_ao_convergence.png",
        "fig6_rate_vs_power.png",       "fig7_rate_vs_L.png",
        "fig8_rate_vs_scee.png",        "fig9_active_budget.png",
        "fig10_uav_trajectory.png",     "fig11_beampattern.png",
        "fig12_transmission_time.png",  "fig13_rate_vs_delta.png",
        "fig14_rate_vs_delta_bar.png",
    ]
    available = [f for f in figs_order if os.path.exists(f"{OUT}/{f}")]
    ncols = 3
    nrows = int(np.ceil(len(available) / ncols))
    fig = plt.figure(figsize=(ncols * 6, nrows * 4.5))
    for i, fname in enumerate(available):
        img = mpimg.imread(f"{OUT}/{fname}")
        ax  = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(img); ax.axis("off")
    fig.tight_layout(pad=0.5)
    path = f"{OUT}/SUMMARY_all_figures.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  OK Summary dashboard saved -> {path}")
    return path
