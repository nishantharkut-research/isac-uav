# Paper-to-Code Audit

## Baseline equation mapping

| Paper item | Code | Status and interpretation |
|---|---|---|
| Eqs. (1)-(2), UAV closure and step | `paper_contract.project_uav_step`, `environment._controls_from_action`, `environment._trajectory_diagnostics` | Every candidate is checked for bounds, step length, remaining return reachability, and exact final-slot closure. Feasibility fallback keeps the already projected UAV position, so a signal-control fallback cannot undo closure. |
| Eqs. (3)-(9), path loss and Rician channels | `channel_model.py` | Seeded BS-HRIS, HRIS-ground, and direct channels with ULA/UPA responses. Eq. (5) maps `zeta_e=zeta_los-zeta_nlos`, `zeta_0=zeta_nlos`. Printed Eq. (6) symbol roles are primary and the standard Al-Hourani mapping is sensitivity. Eq. (8) Doppler uses target radial velocity and is applied only to the moving-target LoS term. Omitted constants are profile assumptions. |
| Eqs. (10)-(11), composite channels | `ChannelSet.composite_ue`, `ChannelSet.composite_te` | Passive path serves communication users and active path serves sensing targets, plus direct links. |
| Eqs. (12)-(15), SINR/rates | `sinr_rate.compute_rates` | Per-user and per-target rates. The printed cross-function channels are reproduction-primary; receiver-consistent channels are a predeclared sensitivity. Eq. (12)'s target-indexed active-noise channel in a UE denominator is represented by the mean target-channel active-noise power because the paper gives no UE-target index mapping. |
| Eq. (16), baseline objective | `ObjectiveSpec.paper`, `evaluate_objective` | Exactly `sum_u R_u`; sensing remains a hard rate constraint. |
| Eqs. (16a)-(16b), phase/amplification | `environment._controls_from_action`, `_find_feasible_controls` | Unit passive phases, bounded active phases and amplitudes. Infeasible proposals use a deterministic descending path scan plus refinement; no monotone-feasibility assumption is made. If both endpoints fail, bounded zero/half/full-gain phase templates are tested on the same channel. Replay receives the policy proposal; deterministic projection is part of the environment transition and the executed control remains diagnostic evidence. |
| Eq. (16c), passive cascade | `paper_contract.cascade_power_16c` | Checked as the squared Frobenius norm of the coherent passive-element sum, including cross-element terms. |
| Eqs. (16d), (18), active power | `sinr_rate.active_tx_power` | Aggregate active power checked separately. Beam columns already contain square-root watts, so BS power is not multiplied twice. |
| Eq. (16e), target rate | `ObjectiveSpec.target_rate_min`, beam allocation | Per-target floor. The paper omits its numeric value. |
| Eq. (16f), total BS power | `beamformers_from_powers` | Per-stream nonnegative powers sum to at most `P_T`. |
| Eq. (17), Capon | `sinr_rate.capon_beamformer` | Deterministic covariance assumption because paper snapshots are unavailable. |
| Eq. (19), coherence rate | `sinr_rate.coherence_corrected_rate` | Validated utility with the corrected overhead guard. It is not applied to paper curves because the paper omits the required coherence interval. |
| Eqs. (20)-(25), SCEE | `ChannelSet`, `_compute_scee_rates` | Estimated channels plus effective error covariance and matrix log-determinant rates. The matrix model is used at both zero and positive estimation error so no formula-switch discontinuity occurs. |
| Eqs. (26)-(29), DDPG | `ddpg_agent.DDPGAgent` | Replay, Bellman critic update, deterministic actor update, and soft targets using `self.tau`. |
| Eq. (30), transfer value | `dtl_agent.DTLAgent` | Primary: current source Q blended with the Eq. (26) target Bellman value. Sensitivity: source/target next-state continuation blend. Terminal transitions use raw reward and never reintroduce bootstrap value. |
| Algorithms 2-3, AO | `ao_optimizer.py` | Disclosed numerical SciPy surrogate because the author CVX implementation and executable auxiliary details are unavailable. It uses beam direction/power allocation, direct phase optimization, and a one-step trajectory block; it is not the full MMSE/CVX/SCA algorithm. Published convergence histories are raw iterates; best-feasible tracking affects only the returned final solution. |
| Fig. 4 reward | `simulation.train_ddpg`, `train_dtl`, `plot_baseline.py` | Raw communication rate only. No penalty plot is produced. Endpoints use the final 100 training episodes; full-curve acceptance independently recomputes a fixed 20-episode causal average at all 15 digitized paper markers. |

## Objective separation

| Mode | Objective | Constraints |
|---|---|---|
| `paper_baseline` | `sum_u R_u` | target floors and physical constraints |
| `png_joint_sum` | `sum_u R_u + sum_k R_k` | user/target floors and physical constraints |
| `comm_retention_sensing` | `sum_k R_k` | communication retention, user/target floors, physical constraints |
| `normalized_pareto` | weighted normalized communication and sensing | user/target floors and physical constraints |

The four rows above describe physical objective modes, not the new algorithm.
The supplied source already implemented sensing rates and fixed weight sweeps.
The research contribution is `ba_pcdtl_v1` in `pareto_agent.py`: a frozen
accepted actor, preference-conditioned residual policy, vector twin-critic
ensemble, training-only anchors, and validation-locked conservative fallback.
The exact PNG point always uses the unnormalized physical `Q_c + Q_s` value.

No mode silently adds a penalty to the reported reward. Constraint deficits are
diagnostics. SciPy AO uses an internal numerical merit function for constrained
search, which is not exported as a rate.

The supplied PNG's positive UE floor is active in literal `png_joint_sum` mode.
Its omitted numerical value is the disclosed `R_u_min=0.5 bits/s/Hz` primary
assumption and can be changed only through an explicit extension sensitivity.

## Table III noise ambiguity

Table III prints `sigma_k^2` but describes the row as active-element noise.
The received-signal equations need distinct target-receiver and active-amplifier
terms. The implementation keeps `sigma2_k` and `sigma2_a` separate. The primary
description-led mapping uses active noise `-70 dBm` and target-receiver noise
`-94 dBm`, matching the only reported receiver floor. The sensitivity mapping
uses `-70 dBm` for both unresolved terms, honoring the printed symbol and row
description simultaneously. Neither is presented as unpublished author fact.

## Non-power hypothesis protocol

`ambiguity_hypotheses.py` defines closed registries for noise mapping,
Equation 30 update, transfer schedule, DTL-WFB replacement, the undisclosed CSI
condition of the standalone DDPG comparator, LoS symbol roles, small-scale
channel coherence, and cross-function interference channels. Primary
branches minimize unsupported mechanics or follow the strongest semantic
evidence. `run_ambiguity_sensitivity.py` executes one-factor alternatives by
default and can execute the full 256-combination factorial.

Printed Eq. (6) is primary because it also agrees with the paper's high-LoS
text; the standard Al-Hourani role mapping is sensitivity. Algorithm 1's
per-time-step channel sampling is primary, but the realization is exposed before
its action is scored. Episode-block fading is sensitivity. Statistical-CSI
states omit channel coefficients; instantaneous CSI adds them, and ParCSI uses a
fixed mask.

The primary DDPG comparator uses pCSI, matching DTL-pCSI so an algorithmic
comparison does not also change channel-estimation quality. The earlier iCSI
DDPG interpretation is retained only as `icsi_comparator` sensitivity. This
choice is not made from the observed DDPG reward.

Every output is checked against its predeclared hypothesis identity. Reports
contain paired deltas but no selected-hypothesis field. These alternatives are
not calibrated or selected against Figure 4 or another holdout.

## Power hypotheses

Table III and Fig. 6 label transmitter power in `dB`; receiver and active noise
are explicitly `dBm`. Because the transmitter reference is not stated, the
implementation evaluates both executable interpretations:

```text
paper_db_as_dbm: 10 dB label -> 10 dBm -> 0.01 W
paper_db_as_dbw: 10 dB label -> 10 dBW -> 10 W
```

Both use common Figure 6 calibration seeds. Infeasible branches, pCSI/iCSI
ordering violations, material power-direction reversals, less than half the
paper's total power response, and curve RMSE above three digitization-uncertainty
units are hard rejections. Remaining branches are ranked by normalized curve
error and slope error. The selection is locked before Figure 4 or any other
holdout is evaluated. A larger simulated reward is not a selection criterion.

The calibration system is `M=4`, `N=2`, `U=K=3`, `L=20`, `La=Lp=10`. The
equal split is declared rather than reported: Figure 6's `L=20` conflicts with
the general `La=20` statement and the paper's `La <= Lp` constraint. Profile,
report, lock, and preparation authority are bound to the current source-tree
digest, so earlier artifacts cannot authorize changed code.

## Values not numerically disclosed by the paper

- `omega_0`, carrier frequency, element spacing, Rician factors, and path-loss
  exponents;
- `R_min`, `Psi_max`, and `P_a_max`;
- source and target transfer domains;
- network hidden widths and exploration process;
- usable scalar `tau` (Table IV prints `{0,1}`);
- transfer weight and any time schedule;
- exact replacement beamformer used by DTL-WFB;
- active/passive split for Figure 6, standalone DDPG comparator CSI, and the
  construction of the partial-CSI observation mask;
- the usable coherence interval needed to apply Eq. (19);
- exact Capon sample covariance and AO/CVX implementation;
- UE-region origin and target speed;
- Table III's target-noise versus active-noise symbol mapping;
- the numerical UE-rate floor in the separately supplied extension PNG.

The paper does not define executable source and target transfer domains. The
implementation uses separately seeded geometries and channel streams, with the
source policy trained under the declared iCSI source profile. This is an
explicit reproducibility assumption, not a recovered author setting. The
current profile schema cannot override these seeds; a transfer-domain study
therefore requires a separately preregistered protocol and code change.

These values are marked in `parameter_evidence.py`. Reported values are locked.
Only the declared channel/geometry subset is structurally selected and then
calibrated against the Fig. 6 `AO-pCSI` and `AO-iCSI` curves. The undisclosed
`R_min`, `Psi_max`, and `P_a_max` remain explicit assumptions/sensitivities;
they are not derived from a favorable candidate. Figures 4, 5, 7, 8, 9, 13,
and 14 are holdouts and never enter selection. In particular, the AO solver is
not used to fit Figure 5 convergence or the Figure 7 DTL/HRIS curve.

The calibrated profile is accepted by full runners only beside a matching
successful `calibration_report.json`. The calibration CLI accepts only a
prepared profile authorized by the structural search, holdout, stress, and
preflight artifacts.

`omega_0` is included in the bounded structural and Figure 6 calibration path.
Element spacing is fixed at the cited half-wavelength convention
`d_spacing=0.5`; carrier frequency is the declared `3 GHz` assumption. Spacing
and carrier frequency are sensitivity-only profile values: they must be
predeclared and all outcomes reported, never chosen because one gives a larger
reward. This avoids adding weakly identifiable parameters to a single-figure
calibration and then overstating the fit as recovery of author settings.

`calibrate_reproduction.py --fixed-sensitivity KEY=VALUE` creates a conditional
calibration branch. The fixed value must belong to `SENSITIVITY_ONLY_KEYS`, is
recorded as `predeclared_not_result_selected`, and is cross-checked between the
calibrated profile and report. Other omitted nuisance parameters may be
recalibrated on Figure 6 in that branch; the fixed value itself is not optimized.
Journal-reported antenna, power, and HRIS dimensions remain locked and are swept
only by the corresponding paper-figure implementations.

`run_holdouts.py` evaluates the selected Figure 6 branch and the untouched
Figures 5, 7, 8, 9, 13, and 14 over five seeds. `validate_holdouts.py` checks
exact raw-array aggregation, qualitative paper contracts, and numerical
tolerance where digitized reference points are available. Figures 10 and 11
remain descriptive, and Figure 12 remains hardware-dependent.

## Checkpoint contract

`checkpointing.py` stores full mutable training state and RNG state with an
identity containing profile, physical configuration, objective, dimensions,
CSI mode, allocator, power convention, noise mapping, Equation 30 update,
transfer schedule, WFB interpretation, LoS mapping, channel coherence, and
cross-function interference interpretation. Atomic
replacement preserves the previous file if a write is interrupted. Resume
begins at the next unfinished episode. Restore also validates episode boundary,
array shapes, finite values, history lengths, and requested episode count.
The run seed function enables deterministic PyTorch/cuDNN execution and records
CUDA RNG state when CUDA is available.

For each research seed, source and target library RNG seeds are derived by the
recorded `sha256_v1_common_target_seed_assignment` protocol. All four target
methods receive the same target-training seed and are reseeded before
construction, so method loop order cannot alter a run. Different architectures
are not claimed to consume identical random draws. Source and
target streams remain distinct, and checkpoint restore supersedes the initial
seed with the exact saved RNG state.

## Acceptance boundary

`validate_results.py` does not modify arrays. Baseline acceptance requires:

- generated-result provenance;
- 100% physical feasibility;
- at least five independent seeds;
- explicit training-reward endpoint provenance;
- paper Fig. 4 full-curve and endpoint tolerances plus paired method ordering.

The legacy `comm_retention_sensing` mode alone checks its declared communication
retention and paired sensing gain. Literal `png_joint_sum` has no retention
field; its legacy validator checks raw joint-rate identity and both rate floors.

BA-PCDTL preliminary acceptance uses at least ten paired policy-training seeds
and requires a strictly positive paired 95% confidence-interval lower bound for
joint-rate improvement over both same-objective source/target DTL and the
accepted paper-objective `DTL-pCSI` baseline evaluated on the same PNG pCSI
test scenes. Publication acceptance additionally requires the complete fair
independent-policy grid and a strictly positive paired Pareto-hypervolume
interval. Both levels require
100% physical feasibility, every UE/target floor, immutable baseline/anchor/gate
provenance, and untouched test scenes.

The holdout summary and validation are hash-bound to their raw result payload,
source tree, selected profile, reference data, and baseline evidence. Feasibility
is counted from the actual AO decisions and policy rollout steps and fails
closed; it is never inserted as an uncounted constant success value. Per-seed
JSON and NPZ arrays must match exactly and both raw files are hash-bound to the
completed summary.

Every extension run requires a passing baseline-validation report whose payload
hash matches the exact baseline summary. Until the colleague-side completed
runs pass these checks, the package is an implementation, not proof of
numerical reproduction or superiority.
