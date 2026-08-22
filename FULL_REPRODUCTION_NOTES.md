# Reproduction Claim Boundary

This is the isolated full-reproduction copy; the original folder is unchanged.
The full multi-seed DRL commands are long training runs and use checkpoints.

- Paper Fig. 4 reward is the raw communication sum rate. No penalty is added
  to or subtracted from the DRL reward in the paper-baseline mode.
- Constraint violations are stored as feasibility diagnostics. Numerical
  merit functions used inside the SciPy AO solver are not reported as rates.
- The supplied PNG objective is implemented separately as communication plus
  sensing sum rate with positive per-UE and per-target floors. Their omitted
  numerical values are disclosed extension assumptions.
- The primary research extension implements the supplied PNG literally:
  unweighted communication plus sensing sum rate with hard UE and target
  floors. Communication-retention sensing is retained only as a labeled legacy
  sensitivity and is not the BA-PCDTL contribution.
- Paper-reference points are visually digitized, labeled as references, and
  never emitted as generated simulation results. Publication CLIs accept only
  the repository's canonical reference payload content.
- AO calibration uses only the Fig. 6 AO-pCSI and AO-iCSI curves. Fig. 7 and
  all other declared holdouts are excluded from calibration.
- Tables III and IV do not disclose every required parameter. Reported values
  are locked; only omitted or ambiguous values can be structurally selected or
  calibrated.
- Omitted-value sensitivity branches use repeatable predeclared
  `--fixed-sensitivity` inputs. Their values are not optimized from curve or RL
  quality and must match in the calibrated profile and report.
- The journal labels transmitter power as dB without defining its reference.
  The implementation evaluates both `dBm` and `dBW` interpretations on Figure
  6, rejects infeasible branches, and locks the best reference-curve match
  before any holdout result is viewed. The primary prior is `dBm`; `dBW` is a
  declared sensitivity branch, not an assumed paper fact.
- Eq. (30) omits a complete executable Bellman integration and transfer
  schedule. The primary preserves the full immediate reward and blends source
  and target next-state continuation values inside the Bellman target. Literal
  current-Q blending and current-source/Bellman blending remain executable
  sensitivities, as does a linear schedule. None is presented as unpublished
  author code.
- Fig. 4 validation uses final-window training communication rewards.
  It also checks every digitized learning-curve marker from raw histories using
  a fixed causal smoother. Post-training evaluation is retained separately for
  paired extension tests.
- Figure 5 convergence histories contain raw AO iterates, not running maxima.
- Target methods receive the same recorded deterministic seed assignment and
  are reseeded before construction, so method loop order cannot alter a run.
  Different architectures are not claimed to consume identical random draws.
- Action projection scans the nonconvex interpolation path before refinement;
  it does not assume rate/power feasibility is monotone in HRIS phase.
- A failed fast power allocation is retried with SciPy on the identical
  channel and control before that control is declared infeasible. This is a
  feasibility fallback, not an objective bonus or channel retry.
- Training, validation, and test use that same fast-plus-SciPy-fallback
  allocator contract. Replay stores the executed physical action; the policy
  proposal is available per step and persisted results retain aggregate
  projection/recovery diagnostics.
- The primary follows Eq. (9)'s `La=Lax*Lay` active rectangular steering and
  places disjoint active/passive rectangular subarrays next to one another on
  one physical aperture. The earlier uniformly interleaved full-UPA reading is
  retained only as a named sensitivity; no mode uses overlapping apertures.
- Every candidate transition is checked for UAV bounds, step length, ability to
  return within the remaining slots, and exact final-slot closure. If both the
  proposal and previous control are signal-infeasible on the sampled channel,
  a bounded deterministic same-channel control search is attempted. The code
  never redraws a channel or scene to manufacture a feasible transition.
- Eq. (8) is followed literally: its printed propagation-distance phase and
  additional `exp(j 2 pi f_rt t)` factor are both applied to moving-target LoS
  channels. This is a reproduction choice, not a claim that the two factors are
  a unique kinematically derived Doppler model. The Table III
  target/active-noise symbol conflict is represented by separate
  description-led and dual-`-70 dBm` hypotheses. DTL-WFB likewise has
  matched-filter and channel-independent no-Capon branches.
- The journal does not disclose the CSI condition of standalone DDPG. The
  primary comparator uses pCSI to match DTL-pCSI and avoid a confounded
  algorithm comparison; DDPG under iCSI is a non-selecting sensitivity.
- The one-factor ambiguity study reuses an accepted primary baseline and trains
  eleven isolated alternatives, including both nonprimary Eq. (30) readings,
  LoS roles, channel coherence, statistical-iCSI observability, the literal paper
  scalar-pCSI/matrix-iCSI rate split, and cross-function interference.
  Its report contains no result-based winner.
- Legacy extension normalization references come from the accepted baseline's
  final 100 training episodes. BA-PCDTL Pareto anchors instead come from locked
  scalar-comparator training tails; vector critic returns are converted to the
  same rate units. Validation and test scenes cannot set references, gate
  parameters, anchors, preferences, or hyperparameters.
- Instantaneous/partial CSI state includes the episode scale used to normalize
  primitive coefficients, preserving absolute channel magnitude needed for
  receiver-noise-limited SINR.
- Every configured evaluation episode is executed and retained. Projection
  attribution must pass the fixed `0.10` projection, `0.01` deterministic
  recovery, and `0.95` retained-action thresholds.
- A SHA-256-derived global evaluation schedule gives every research seed a
  disjoint scene block while pairing all compared methods on exactly the same
  scenes. Evaluation scenes cannot overlap environment or training seeds.
- Scalar and Pareto evaluations retain action-projection, retained-action,
  deterministic-recovery, and allocator-fallback rates so post-projection
  feasibility cannot conceal dependence on environment repair.
- Full DRL training was not run while preparing this source package. Numerical
  agreement or improvement is established only by completed multi-seed runs:
  `validate_results.py` for the paper baseline and
  `validate_pareto_extension.py` for BA-PCDTL.
- Because source identity is part of every manifest and checkpoint, results or
  checkpoints created by an older source tree cannot be reused with this one.
- Completed-result validation opens each checkpoint and requires its internal
  identity, completed episode count, history keys, and finite raw histories to
  equal the serialized result evidence. File hashes and sidecars alone are not
  accepted as proof of checkpoint authority.
- A sensing extension may start only from the exact baseline payload named in a
  passing baseline-validation report.
- The publication sweep runner evaluates Figure 6 plus untouched Figures 5, 7,
  8, 9, 13, and 14 on five seeds with episode and per-figure resume state.
- Figures 10-12 are retained only as descriptive diagnostics because the paper
  does not provide a defensible quantitative trajectory/beampattern reference
  or a portable runtime protocol.
