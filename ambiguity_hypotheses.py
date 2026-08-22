"""Predeclared executable hypotheses for details omitted by the journal."""

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from types import MappingProxyType

from physical_hypotheses import dbm_to_watts


@dataclass(frozen=True)
class NoiseHypothesis:
    identifier: str
    role: str
    ue_receiver_dbm: float
    target_receiver_dbm: float
    active_amplifier_dbm: float
    rationale: str


@dataclass(frozen=True)
class TransferUpdateHypothesis:
    identifier: str
    role: str
    rationale: str


@dataclass(frozen=True)
class TransferScheduleHypothesis:
    identifier: str
    role: str
    rationale: str


@dataclass(frozen=True)
class WFBHypothesis:
    identifier: str
    role: str
    beamformer_mode: str
    rationale: str


@dataclass(frozen=True)
class DDPGComparatorCSIHypothesis:
    identifier: str
    role: str
    scee_multiplier: float
    rationale: str


@dataclass(frozen=True)
class LoSProbabilityHypothesis:
    identifier: str
    role: str
    formula: str
    rationale: str


@dataclass(frozen=True)
class ChannelCoherenceHypothesis:
    identifier: str
    role: str
    resample_each_slot: bool
    rationale: str


@dataclass(frozen=True)
class CSIObservationHypothesis:
    identifier: str
    role: str
    perfect_observes_channels: bool
    imperfect_observes_channels: bool
    partial_observes_channels: bool
    rationale: str


@dataclass(frozen=True)
class RateModelHypothesis:
    identifier: str
    role: str
    matrix_at_zero_error: bool
    rationale: str


@dataclass(frozen=True)
class CrossFunctionInterferenceHypothesis:
    identifier: str
    role: str
    receiver_consistent: bool
    rationale: str


PRIMARY_NOISE_HYPOTHESIS_ID = "description_literal_target_matches_ue"
PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID = "eq30_next_state_continuation_blend"
PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID = "constant_upsilon"
PRIMARY_WFB_HYPOTHESIS_ID = "adaptive_matched_filter_no_capon"
PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID = "fair_pcsi_comparator"
PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID = "paper_unnumbered_los_literal_roles"
PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID = "independent_slot_fading"
PRIMARY_CSI_OBSERVATION_HYPOTHESIS_ID = (
    "instantaneous_estimates_all_csi_qualities"
)
PRIMARY_RATE_MODEL_HYPOTHESIS_ID = "continuous_matrix_scee_all_csi_qualities"
PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID = (
    "paper_eq12_eq14_literal_cross_channels"
)


NOISE_HYPOTHESES = MappingProxyType({
    PRIMARY_NOISE_HYPOTHESIS_ID: NoiseHypothesis(
        identifier=PRIMARY_NOISE_HYPOTHESIS_ID,
        role="primary",
        ue_receiver_dbm=-94.0,
        target_receiver_dbm=-94.0,
        active_amplifier_dbm=-70.0,
        rationale=(
            "Follow the Table III description for active-element noise and "
            "use the only reported receiver-noise floor for the unreported "
            "target receiver."
        ),
    ),
    "dual_minus70_conservative": NoiseHypothesis(
        identifier="dual_minus70_conservative",
        role="sensitivity",
        ue_receiver_dbm=-94.0,
        target_receiver_dbm=-70.0,
        active_amplifier_dbm=-70.0,
        rationale=(
            "Honor both the sigma_k symbol and active-noise description in "
            "the conflicting Table III row."
        ),
    ),
})


TRANSFER_UPDATE_HYPOTHESES = MappingProxyType({
    "eq30_literal_current_q_blend": TransferUpdateHypothesis(
        identifier="eq30_literal_current_q_blend",
        role="sensitivity",
        rationale=(
            "Implement Eq. (30) literally as the current-state source/target "
            "Q blend. This omits a reward-bearing Bellman update and is kept "
            "only to report the printed formula's behavior."
        ),
    ),
    "eq30_current_q_bellman_blend": TransferUpdateHypothesis(
        identifier="eq30_current_q_bellman_blend",
        role="sensitivity",
        rationale=(
            "Blend current source Q with the standard DDPG Bellman target. "
            "This reward-bearing interpretation attenuates the Bellman target "
            "when transfer is high and is retained only as a sensitivity."
        ),
    ),
    PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID: TransferUpdateHypothesis(
        identifier=PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
        role="primary",
        rationale=(
            "Preserve the full immediate reward and blend source and target "
            "next-state continuation values inside the Bellman target. This "
            "is the operational integration of Eq. (30); the paper does not "
            "publish a complete reward-bearing update."
        ),
    ),
})


TRANSFER_SCHEDULE_HYPOTHESES = MappingProxyType({
    PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID: TransferScheduleHypothesis(
        identifier=PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
        role="primary",
        rationale=(
            "Use a constant transfer rate because the paper publishes no "
            "annealing schedule."
        ),
    ),
    "linear_annealed_upsilon": TransferScheduleHypothesis(
        identifier="linear_annealed_upsilon",
        role="sensitivity",
        rationale=(
            "Linearly reduce the transfer rate to its configured floor as an "
            "explicit schedule sensitivity."
        ),
    ),
})


WFB_HYPOTHESES = MappingProxyType({
    PRIMARY_WFB_HYPOTHESIS_ID: WFBHypothesis(
        identifier=PRIMARY_WFB_HYPOTHESIS_ID,
        role="primary",
        beamformer_mode="matched_filter",
        rationale=(
            "Remove Capon covariance inversion while retaining an adaptive, "
            "physically meaningful target matched filter."
        ),
    ),
    "channel_independent_dft_no_capon": WFBHypothesis(
        identifier="channel_independent_dft_no_capon",
        role="sensitivity",
        beamformer_mode="channel_independent_dft",
        rationale=(
            "Use deterministic channel-independent DFT sensing directions to "
            "bound a lower-information reading of DTL-WFB."
        ),
    ),
})


DDPG_COMPARATOR_CSI_HYPOTHESES = MappingProxyType({
    PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID:
        DDPGComparatorCSIHypothesis(
            identifier=PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
            role="primary",
            scee_multiplier=0.0,
            rationale=(
                "Use pCSI for the independently initialized DDPG comparator, "
                "matching DTL-pCSI so the algorithm comparison does not "
                "confound transfer learning with channel-estimation error."
            ),
        ),
    "icsi_comparator": DDPGComparatorCSIHypothesis(
        identifier="icsi_comparator",
        role="sensitivity",
        scee_multiplier=1.0,
        rationale=(
            "Use the configured iCSI NMSE for DDPG as an explicit sensitivity "
            "because the journal does not disclose the comparator CSI state."
        ),
    ),
})


LOS_PROBABILITY_HYPOTHESES = MappingProxyType({
    PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID: LoSProbabilityHypothesis(
        identifier=PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
        role="primary",
        formula="1/(1+b_env*exp(-a_env*(elevation-b_env)))",
        rationale=(
            "Follow the paper's unnumbered LoS formula as printed. With the "
            "declared urban constants this "
            "also matches the paper's stated high-LoS/free-space assumption "
            "for UAV-ground communication links."
        ),
    ),
    "al_hourani_standard_mapped_roles": LoSProbabilityHypothesis(
        identifier="al_hourani_standard_mapped_roles",
        role="sensitivity",
        formula="1/(1+a_env*exp(-b_env*(elevation-a_env)))",
        rationale=(
            "Map the conventional Al-Hourani constant roles into the paper's "
            "reversed symbols. This external-convention interpretation is "
            "reported as a non-selecting sensitivity."
        ),
    ),
})


CHANNEL_COHERENCE_HYPOTHESES = MappingProxyType({
    "episode_block_fading": ChannelCoherenceHypothesis(
        identifier="episode_block_fading",
        role="sensitivity",
        resample_each_slot=False,
        rationale=(
            "Hold one latent realization for an episode as a block-fading "
            "sensitivity; the printed Algorithm 1 samples inside the time-step "
            "loop instead."
        ),
    ),
    PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID: ChannelCoherenceHypothesis(
        identifier=PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
        role="primary",
        resample_each_slot=True,
        rationale=(
            "Follow Algorithm 1 by sampling a realization for every time-step. "
            "The environment exposes that realization before its action is "
            "evaluated, preserving instantaneous-CSI observability."
        ),
    ),
})


CSI_OBSERVATION_HYPOTHESES = MappingProxyType({
    "pcsi_instantaneous_icsi_statistical": CSIObservationHypothesis(
        identifier="pcsi_instantaneous_icsi_statistical",
        role="sensitivity",
        perfect_observes_channels=True,
        imperfect_observes_channels=False,
        partial_observes_channels=True,
        rationale=(
            "Use instantaneous channel coefficients for pCSI and masked ParCSI "
            "states, while treating the iCSI curve as statistical CSI. This "
            "alternative is retained because the paper discusses a separate "
            "statistical-CSI state but does not label such a curve explicitly."
        ),
    ),
    PRIMARY_CSI_OBSERVATION_HYPOTHESIS_ID: CSIObservationHypothesis(
        identifier=PRIMARY_CSI_OBSERVATION_HYPOTHESIS_ID,
        role="primary",
        perfect_observes_channels=True,
        imperfect_observes_channels=True,
        partial_observes_channels=True,
        rationale=(
            "Expose true channels for pCSI, estimated channels for iCSI, and "
            "masked estimated channels for ParCSI. The SCEE section defines "
            "iCSI through channel estimates, so estimation quality remains "
            "separate from instantaneous state observability."
        ),
    ),
})


RATE_MODEL_HYPOTHESES = MappingProxyType({
    PRIMARY_RATE_MODEL_HYPOTHESIS_ID: RateModelHypothesis(
        identifier=PRIMARY_RATE_MODEL_HYPOTHESIS_ID,
        role="primary",
        matrix_at_zero_error=True,
        rationale=(
            "Evaluate Eqs. (21)/(25) continuously at zero and positive SCEE. "
            "This avoids changing the receiver-rate functional solely because "
            "the NMSE moves above zero."
        ),
    ),
    "paper_split_scalar_pcsi_matrix_icsi": RateModelHypothesis(
        identifier="paper_split_scalar_pcsi_matrix_icsi",
        role="sensitivity",
        matrix_at_zero_error=False,
        rationale=(
            "Use scalar Eqs. (12)-(15) at pCSI and matrix Eqs. (21)/(25) at "
            "positive SCEE. This follows the paper's section split literally "
            "but changes the numerical receiver model at zero NMSE."
        ),
    ),
})


CROSS_FUNCTION_INTERFERENCE_HYPOTHESES = MappingProxyType({
    PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID:
        CrossFunctionInterferenceHypothesis(
            identifier=PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID,
            role="primary",
            receiver_consistent=False,
            rationale=(
                "Reproduce the cross-function interference terms in Eqs. "
                "(12), (14), (21), and (25) exactly as printed."
            ),
        ),
    "receiver_consistent_cross_channels":
        CrossFunctionInterferenceHypothesis(
            identifier="receiver_consistent_cross_channels",
            role="sensitivity",
            receiver_consistent=True,
            rationale=(
                "Route both communication and sensing waveforms through the "
                "channel of the receiver whose SINR is being evaluated; this "
                "matches the physical receiver convention used in Remark 1."
            ),
        ),
})


AMBIGUITY_IDENTITY_REGISTRIES = MappingProxyType({
    "noise_hypothesis_id": NOISE_HYPOTHESES,
    "dtl_transfer_mode": TRANSFER_UPDATE_HYPOTHESES,
    "dtl_transfer_schedule_id": TRANSFER_SCHEDULE_HYPOTHESES,
    "wfb_hypothesis_id": WFB_HYPOTHESES,
    "ddpg_comparator_csi_hypothesis_id": DDPG_COMPARATOR_CSI_HYPOTHESES,
    "los_probability_hypothesis_id": LOS_PROBABILITY_HYPOTHESES,
    "channel_coherence_hypothesis_id": CHANNEL_COHERENCE_HYPOTHESES,
    "csi_observation_hypothesis_id": CSI_OBSERVATION_HYPOTHESES,
    "rate_model_hypothesis_id": RATE_MODEL_HYPOTHESES,
    "cross_function_interference_hypothesis_id":
        CROSS_FUNCTION_INTERFERENCE_HYPOTHESES,
})

AMBIGUITY_FIELD_TO_AXIS = MappingProxyType({
    "noise_hypothesis_id": "noise",
    "dtl_transfer_mode": "transfer_update",
    "dtl_transfer_schedule_id": "transfer_schedule",
    "wfb_hypothesis_id": "wfb",
    "ddpg_comparator_csi_hypothesis_id": "ddpg_comparator_csi",
    "los_probability_hypothesis_id": "los_probability",
    "channel_coherence_hypothesis_id": "channel_coherence",
    "csi_observation_hypothesis_id": "csi_observation",
    "rate_model_hypothesis_id": "rate_model",
    "cross_function_interference_hypothesis_id":
        "cross_function_interference",
})


def _lookup(registry, identifier, label):
    try:
        return registry[str(identifier)]
    except KeyError as error:
        raise ValueError(f"Unknown {label} hypothesis: {identifier}") from error


def get_noise_hypothesis(identifier):
    return _lookup(NOISE_HYPOTHESES, identifier, "noise")


def get_transfer_update_hypothesis(identifier):
    return _lookup(TRANSFER_UPDATE_HYPOTHESES, identifier, "transfer update")


def get_transfer_schedule_hypothesis(identifier):
    return _lookup(
        TRANSFER_SCHEDULE_HYPOTHESES, identifier, "transfer schedule"
    )


def get_wfb_hypothesis(identifier):
    return _lookup(WFB_HYPOTHESES, identifier, "WFB")


def get_ddpg_comparator_csi_hypothesis(identifier):
    return _lookup(
        DDPG_COMPARATOR_CSI_HYPOTHESES, identifier, "DDPG comparator CSI"
    )


def get_los_probability_hypothesis(identifier):
    return _lookup(LOS_PROBABILITY_HYPOTHESES, identifier, "LoS probability")


def get_channel_coherence_hypothesis(identifier):
    return _lookup(CHANNEL_COHERENCE_HYPOTHESES, identifier, "channel coherence")


def get_csi_observation_hypothesis(identifier):
    return _lookup(CSI_OBSERVATION_HYPOTHESES, identifier, "CSI observation")


def get_rate_model_hypothesis(identifier):
    return _lookup(RATE_MODEL_HYPOTHESES, identifier, "rate model")


def use_matrix_scee_rate(config, scee_nmse):
    """Resolve the predeclared zero-error rate-functional interpretation."""
    hypothesis = get_rate_model_hypothesis(config.rate_model_hypothesis_id)
    return bool(hypothesis.matrix_at_zero_error or float(scee_nmse) > 0.0)


def get_cross_function_interference_hypothesis(identifier):
    return _lookup(
        CROSS_FUNCTION_INTERFERENCE_HYPOTHESES,
        identifier,
        "cross-function interference",
    )


def ddpg_comparator_scee(config):
    """Return the comparator NMSE without result-dependent selection."""
    hypothesis = get_ddpg_comparator_csi_hypothesis(
        config.ddpg_comparator_csi_hypothesis_id
    )
    return float(hypothesis.scee_multiplier * config.rho_nmse)


def csi_observes_instantaneous(config, quality):
    """Resolve state observability independently of CSI estimation quality."""
    hypothesis = get_csi_observation_hypothesis(
        config.csi_observation_hypothesis_id
    )
    fields = {
        "perfect": hypothesis.perfect_observes_channels,
        "imperfect": hypothesis.imperfect_observes_channels,
        "partial": hypothesis.partial_observes_channels,
    }
    try:
        return bool(fields[str(quality)])
    except KeyError as error:
        raise ValueError(f"Unknown CSI quality: {quality}") from error


def apply_ambiguity_hypotheses(
        config,
        noise_hypothesis_id=PRIMARY_NOISE_HYPOTHESIS_ID,
        transfer_update_hypothesis_id=PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
        transfer_schedule_hypothesis_id=PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
        wfb_hypothesis_id=PRIMARY_WFB_HYPOTHESIS_ID,
        ddpg_comparator_csi_hypothesis_id=
        PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
        los_probability_hypothesis_id=
        PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
        channel_coherence_hypothesis_id=
        PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
        csi_observation_hypothesis_id=
        PRIMARY_CSI_OBSERVATION_HYPOTHESIS_ID,
        rate_model_hypothesis_id=PRIMARY_RATE_MODEL_HYPOTHESIS_ID,
        cross_function_interference_hypothesis_id=
        PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID):
    """Apply one complete, predeclared ambiguity configuration."""
    noise = get_noise_hypothesis(noise_hypothesis_id)
    update = get_transfer_update_hypothesis(transfer_update_hypothesis_id)
    schedule = get_transfer_schedule_hypothesis(
        transfer_schedule_hypothesis_id
    )
    wfb = get_wfb_hypothesis(wfb_hypothesis_id)
    ddpg_csi = get_ddpg_comparator_csi_hypothesis(
        ddpg_comparator_csi_hypothesis_id
    )
    los_probability = get_los_probability_hypothesis(
        los_probability_hypothesis_id
    )
    coherence = get_channel_coherence_hypothesis(
        channel_coherence_hypothesis_id
    )
    csi_observation = get_csi_observation_hypothesis(
        csi_observation_hypothesis_id
    )
    rate_model = get_rate_model_hypothesis(rate_model_hypothesis_id)
    cross_interference = get_cross_function_interference_hypothesis(
        cross_function_interference_hypothesis_id
    )

    config.noise_hypothesis_id = noise.identifier
    config.sigma2_u_dBm = noise.ue_receiver_dbm
    config.sigma2_k_dBm = noise.target_receiver_dbm
    config.sigma2_a_dBm = noise.active_amplifier_dbm
    config.sigma2_u = dbm_to_watts(noise.ue_receiver_dbm)
    config.sigma2_k = dbm_to_watts(noise.target_receiver_dbm)
    config.sigma2_a = dbm_to_watts(noise.active_amplifier_dbm)
    config.dtl_transfer_mode = update.identifier
    config.dtl_transfer_schedule_id = schedule.identifier
    config.wfb_hypothesis_id = wfb.identifier
    config.wfb_beamformer_mode = wfb.beamformer_mode
    config.ddpg_comparator_csi_hypothesis_id = ddpg_csi.identifier
    config.los_probability_hypothesis_id = los_probability.identifier
    config.channel_coherence_hypothesis_id = coherence.identifier
    config.csi_observation_hypothesis_id = csi_observation.identifier
    config.rate_model_hypothesis_id = rate_model.identifier
    config.cross_function_interference_hypothesis_id = (
        cross_interference.identifier
    )
    return ambiguity_metadata(config)


@contextmanager
def applied_ambiguity_hypotheses(config, **overrides):
    """Temporarily apply a complete ambiguity contract and restore exactly."""
    fields = (
        "noise_hypothesis_id", "sigma2_u_dBm", "sigma2_k_dBm",
        "sigma2_a_dBm", "sigma2_u", "sigma2_k", "sigma2_a",
        "dtl_transfer_mode", "dtl_transfer_schedule_id",
        "wfb_hypothesis_id", "wfb_beamformer_mode",
        "ddpg_comparator_csi_hypothesis_id",
        "los_probability_hypothesis_id", "channel_coherence_hypothesis_id",
        "csi_observation_hypothesis_id",
        "rate_model_hypothesis_id",
        "cross_function_interference_hypothesis_id",
    )
    saved = {field: getattr(config, field) for field in fields}
    try:
        yield apply_ambiguity_hypotheses(config, **overrides)
    finally:
        for field, value in saved.items():
            setattr(config, field, value)


def ambiguity_metadata(config):
    """Return assumption identities only; this structure contains no results."""
    noise = get_noise_hypothesis(config.noise_hypothesis_id)
    update = get_transfer_update_hypothesis(config.dtl_transfer_mode)
    schedule = get_transfer_schedule_hypothesis(
        config.dtl_transfer_schedule_id
    )
    wfb = get_wfb_hypothesis(config.wfb_hypothesis_id)
    ddpg_csi = get_ddpg_comparator_csi_hypothesis(
        config.ddpg_comparator_csi_hypothesis_id
    )
    los_probability = get_los_probability_hypothesis(
        config.los_probability_hypothesis_id
    )
    coherence = get_channel_coherence_hypothesis(
        config.channel_coherence_hypothesis_id
    )
    csi_observation = get_csi_observation_hypothesis(
        config.csi_observation_hypothesis_id
    )
    rate_model = get_rate_model_hypothesis(config.rate_model_hypothesis_id)
    cross_interference = get_cross_function_interference_hypothesis(
        config.cross_function_interference_hypothesis_id
    )
    return {
        "schema_version": 1,
        "kind": "ambiguity_hypothesis_contract",
        "selection_rule": "predeclared_not_result_selected",
        "axes": {
            "noise": asdict(noise),
            "transfer_update": asdict(update),
            "transfer_schedule": asdict(schedule),
            "wfb": asdict(wfb),
            "ddpg_comparator_csi": asdict(ddpg_csi),
            "los_probability": asdict(los_probability),
            "channel_coherence": asdict(coherence),
            "csi_observation": asdict(csi_observation),
            "rate_model": asdict(rate_model),
            "cross_function_interference": asdict(cross_interference),
        },
    }


def apply_ambiguity_manifest_fields(payload, config):
    """Add immutable ambiguity identities to a run/result payload."""
    payload["noise_hypothesis_id"] = config.noise_hypothesis_id
    payload["dtl_transfer_mode"] = config.dtl_transfer_mode
    payload["dtl_transfer_schedule_id"] = config.dtl_transfer_schedule_id
    payload["wfb_hypothesis_id"] = config.wfb_hypothesis_id
    payload["ddpg_comparator_csi_hypothesis_id"] = (
        config.ddpg_comparator_csi_hypothesis_id
    )
    payload["los_probability_hypothesis_id"] = (
        config.los_probability_hypothesis_id
    )
    payload["channel_coherence_hypothesis_id"] = (
        config.channel_coherence_hypothesis_id
    )
    payload["csi_observation_hypothesis_id"] = (
        config.csi_observation_hypothesis_id
    )
    payload["rate_model_hypothesis_id"] = config.rate_model_hypothesis_id
    payload["cross_function_interference_hypothesis_id"] = (
        config.cross_function_interference_hypothesis_id
    )
    payload["ambiguity_contract"] = ambiguity_metadata(config)
    return payload


def valid_ambiguity_manifest_fields(payload):
    """Return true only for canonical, internally consistent identities."""
    if not isinstance(payload, dict) or not all(
        payload.get(field) in registry
        for field, registry in AMBIGUITY_IDENTITY_REGISTRIES.items()
    ):
        return False
    expected_axes = {
        AMBIGUITY_FIELD_TO_AXIS[field]: asdict(registry[payload[field]])
        for field, registry in AMBIGUITY_IDENTITY_REGISTRIES.items()
    }
    contract = payload.get("ambiguity_contract")
    if contract is None:
        return True
    return contract == {
        "schema_version": 1,
        "kind": "ambiguity_hypothesis_contract",
        "selection_rule": "predeclared_not_result_selected",
        "axes": expected_axes,
    }


def require_matching_ambiguity_contract(payload, config, label="Baseline"):
    if (
        "ambiguity_contract" not in payload
        or not valid_ambiguity_manifest_fields(payload)
    ):
        raise ValueError(f"{label} ambiguity contract is invalid or inconsistent.")
    expected = {
        "noise_hypothesis_id": config.noise_hypothesis_id,
        "dtl_transfer_mode": config.dtl_transfer_mode,
        "dtl_transfer_schedule_id": config.dtl_transfer_schedule_id,
        "wfb_hypothesis_id": config.wfb_hypothesis_id,
        "ddpg_comparator_csi_hypothesis_id":
            config.ddpg_comparator_csi_hypothesis_id,
        "los_probability_hypothesis_id":
            config.los_probability_hypothesis_id,
        "channel_coherence_hypothesis_id":
            config.channel_coherence_hypothesis_id,
        "csi_observation_hypothesis_id":
            config.csi_observation_hypothesis_id,
        "rate_model_hypothesis_id": config.rate_model_hypothesis_id,
        "cross_function_interference_hypothesis_id":
            config.cross_function_interference_hypothesis_id,
    }
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"{label} ambiguity contract does not match: "
            + ", ".join(mismatches)
        )
    return True


def add_ambiguity_cli_arguments(parser):
    parser.add_argument(
        "--noise-hypothesis",
        choices=tuple(NOISE_HYPOTHESES),
        default=PRIMARY_NOISE_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--transfer-update-hypothesis",
        choices=tuple(TRANSFER_UPDATE_HYPOTHESES),
        default=PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--transfer-schedule-hypothesis",
        choices=tuple(TRANSFER_SCHEDULE_HYPOTHESES),
        default=PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--wfb-hypothesis",
        choices=tuple(WFB_HYPOTHESES),
        default=PRIMARY_WFB_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--ddpg-comparator-csi-hypothesis",
        choices=tuple(DDPG_COMPARATOR_CSI_HYPOTHESES),
        default=PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--los-probability-hypothesis",
        choices=tuple(LOS_PROBABILITY_HYPOTHESES),
        default=PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--channel-coherence-hypothesis",
        choices=tuple(CHANNEL_COHERENCE_HYPOTHESES),
        default=PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--csi-observation-hypothesis",
        choices=tuple(CSI_OBSERVATION_HYPOTHESES),
        default=PRIMARY_CSI_OBSERVATION_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--rate-model-hypothesis",
        choices=tuple(RATE_MODEL_HYPOTHESES),
        default=PRIMARY_RATE_MODEL_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--cross-function-interference-hypothesis",
        choices=tuple(CROSS_FUNCTION_INTERFERENCE_HYPOTHESES),
        default=PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID,
    )
    return parser


def apply_ambiguity_arguments(args, config):
    return apply_ambiguity_hypotheses(
        config,
        noise_hypothesis_id=args.noise_hypothesis,
        transfer_update_hypothesis_id=args.transfer_update_hypothesis,
        transfer_schedule_hypothesis_id=args.transfer_schedule_hypothesis,
        wfb_hypothesis_id=args.wfb_hypothesis,
        ddpg_comparator_csi_hypothesis_id=
            args.ddpg_comparator_csi_hypothesis,
        los_probability_hypothesis_id=args.los_probability_hypothesis,
        channel_coherence_hypothesis_id=args.channel_coherence_hypothesis,
        csi_observation_hypothesis_id=args.csi_observation_hypothesis,
        rate_model_hypothesis_id=args.rate_model_hypothesis,
        cross_function_interference_hypothesis_id=
            args.cross_function_interference_hypothesis,
    )
