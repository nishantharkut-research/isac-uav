# End-to-End Reproduction Correction Report

## Purpose

This document records the corrective work performed after review of
`E2E_PROBLEM_ANALYSIS.md`. The exact previously delivered source is preserved
on GitHub `main` at commit `7b035dd`. All changes described here are developed
on `agent/e2e-reproduction-corrections` so the baseline and correction remain
independently auditable.

This source revision does not contain a newly trained result and does not claim
that the paper has already been numerically reproduced. The old calibration,
checkpoints, and plots were generated under a different physical contract and
must not be reused. Numerical reproduction is established only by a fresh
calibration, multi-seed run, and passing validators.

## Findings and Disposition

| ID | Review finding | Disposition |
|---|---|---|
| A | SCEE rates jump when NMSE changes from zero | Confirmed symptom. The immediate cause was a switch from scalar Eqs. (12)/(14) at zero error to matrix Eqs. (21)/(25) at any positive error, not the paper-required use of estimated CSI in the numerator. Continuous matrix SCEE is primary; the paper's literal scalar-pCSI/matrix-iCSI section split is a separate report-all sensitivity. |
| B | Unnumbered LoS-formula constants appear swapped | Confirmed paper/model ambiguity, not a uniquely recoverable author value. The paper's printed form is the reproduction primary because it also matches the high-LoS/free-space text; the standard Al-Hourani role mapping is a predeclared non-selecting sensitivity. |
| C | Eqs. (12)/(14) cross-function channels are likely inconsistent | Confirmed ambiguity. The printed equations remain primary for reproduction; receiver-consistent cross channels are a report-all sensitivity. Both scalar and matrix rate paths implement the same selected hypothesis. |
| D | Power selection accepts pCSI/iCSI inversion | Confirmed and fixed. Any candidate with an ordering violation is inadmissible. |
| E | Constraint values were derived from one arbitrary pre-calibration candidate | Confirmed and removed from the identification path. `R_min`, `Psi_max`, and `P_a_max` are disclosed omitted-parameter assumptions/sensitivities, not result-derived structural parameters. |
| F/G | Power response and policy-action response are nearly flat | Confirmed and exposed. Structural preparation and Figure 6 calibration retain declared acceptance checks. The baseline launcher records a deterministic pre-training action-response diagnostic, but weak response is advisory and never terminates training. |
| J | SLSQP repeatedly recomputes identical assessments | Confirmed and fixed with exact-input assessment reuse. |
| K | BA-PCDTL resume does not bind Pareto anchors | Confirmed and fixed. The canonical anchor digest is part of both agent and run checkpoint contracts. |
| L | Pareto anchors include early random episodes and extrema | Confirmed risk and fixed. Anchors use the converged training tail and 5th/95th quantiles, with estimator provenance frozen into the anchor identity. |
| I | Sensing and cascade-error loops are slow | Confirmed and vectorized. Direct elementwise-equivalence tests protect the numerical equations. |
| M | Eq. (19) helper has an off-by-one guard and is not in the live pipeline | Guard fixed. The helper remains an explicit utility because the paper does not publish the coherence interval needed to apply it to the reported curves. |

## Additional Defects Found During Correction

The policy state exposed one phase-dependent composite realization while the
environment changed geometry before evaluating the action. The primary state
now exposes all estimated primitive channel tensors, so alternative HRIS phases
remain identifiable. Phases/gains are scored on the observed current channel;
UAV/target motion determines the next geometry; the next channel and precoders
are constructed atomically. Episode-block fading remains a declared
sensitivity, and coherence identity plus latent seed are checkpoint-bound.
The normalization scale for primitive instantaneous/partial CSI is also part
of the observation, so two different absolute channel magnitudes cannot become
the same normalized state while producing different noise-limited SINRs.

CSI estimation quality and state observability were previously coupled. They
are now separate, fingerprinted axes. The primary exposes zero-NMSE primitive
estimates for pCSI, nonzero-NMSE estimates for iCSI, and masked estimates for
ParCSI because the SCEE section defines iCSI through channel estimates. The
alternative statistical-iCSI reading omits channel coefficients and is retained
as a non-selecting sensitivity because Section III also describes a statistical
state without labeling which plotted CSI curve uses it.

Equation (16c) places the squared norm inside the sum over passive elements.
The earlier coherent-sum implementation incorrectly introduced cross-element
terms and allowed opposite phases to cancel physical element powers. The code
now sums the per-element squared Frobenius norms and uses the worst user because
the printed constraint omits its communication-user index.

Replay previously mixed proposal and feasibility-projected action semantics.
Replay now stores the executed physical action that produced the recorded
reward, so critic training remains consistent with the environment transition.
The raw proposal and executed action are available in each in-memory step
diagnostic; persisted results retain aggregate projection/recovery evidence.

The original channel construction also generated active and passive HRIS paths
as separate overlapping apertures. The corrected primary implements Eq. (9)'s
`La=Lax*Lay` active rectangular response and places disjoint active/passive
rectangular subarrays next to one another on one physical aperture. Both slices
come from the same BS-HRIS and HRIS-ground realization. The former interleaved
full-UPA reading remains an explicit sensitivity.

Cross-dimensional DTL previously skipped source-Q evaluation when pCSI added
state features, which silently removed the Eq. (30) source term. The source
network is now evaluated on the exact shared state prefix, added target input
columns start at zero, and incompatible action dimensions fail closed.

## Calibration Acceptance

Power-reference selection uses Figure 6 only. A branch is now admissible only
when all of the following hold:

1. all AO calibration points are feasible;
2. no pCSI/iCSI ordering violation occurs;
3. no material communication-rate reversal occurs as transmitter power rises;
4. each AO curve recovers at least 50% of the paper's 10-to-30 dB response;
5. Figure 6 curve RMSE is at most three digitization-uncertainty units
   (`0.24 bits/s/Hz` for the declared `0.08` uncertainty).

If neither dBm nor dBW interpretation passes, calibration stops. It does not
select the less-bad branch and does not proceed to DRL training.

Calibration now uses the Figure 6 system contract (`M=4`, `N=2`, `U=K=3`,
`L=20`, `La=Lp=10`) instead of silently inheriting the default `L=40` system.
The equal active/passive split is a disclosed feasible assumption used to reconcile
Figure 6's `L=20` with the paper's otherwise contradictory `La=20` statement and
`La <= Lp` constraint. Calibration artifacts are also source-tree bound and
explicitly identify the SciPy AO surrogate; an artifact from old executable
source cannot authorize a new run.

## Reward and Constraints

The paper-baseline reward is exactly `sum_u R_u`. The literal extension reward
is exactly `sum_u R_u + sum_k R_k`. Neither reported reward contains a penalty.
Constraint deficits are diagnostics and constrained-optimizer internals only.
Rate floors, HRIS power, BS power, phase/gain, UAV motion, return reachability,
and closure are enforced as hard feasibility conditions.

## Ambiguities

The primary is fixed before results are viewed. Closed sensitivity registries
cover noise mapping, transfer update, transfer schedule, WFB replacement, DDPG
CSI, LoS probability, channel coherence, CSI observability, zero-error rate
functional, and cross-function interference. The one-at-a-time design has one
primary plus eleven alternatives. The optional full factorial has 1536
scenarios. Sensitivity reports contain paired deltas but no
winner-selection field.

## Checkpoint and Evidence Integrity

Scalar DDPG/DTL and BA-PCDTL checkpoints contain model, optimizer, replay,
environment, histories, and RNG state. BA-PCDTL additionally binds frozen
Pareto anchors and preference-stream position. Run manifests and plot sidecars
include every physical ambiguity identity. A source, profile, anchor, seed,
objective, or hypothesis mismatch fails resume and evidence validation.
Completed-result validation also opens every checkpoint and binds its internal
identity, requested/completed episode count, history keys, and finite raw
histories to the serialized result row. Recomputed outer hashes cannot conceal
an unrelated or edited checkpoint payload.
For DTL, the source-network digest is recomputed from both the accepted source
checkpoint and the source agent embedded in the target checkpoint; both must
match the target identity.
Evaluation evidence separately records action-projection rate, mean retained
action fraction, deterministic-recovery rate, and allocation-fallback rate.
Training, validation, and test use the same fast-grid allocator with a
same-channel SciPy feasibility fallback. Validators reject evidence above
`0.10` action projection or `0.01` deterministic recovery, or below `0.95`
mean retained-action fraction.

Extension objective references are reconstructed from the accepted baseline's
final 100 training episodes, not held-out evaluation scenes. Each requested
evaluation episode is actually executed and preserved in nested raw evidence.
All methods for a research seed use the same SHA-256-derived scene block, and
blocks are globally unique across research seeds and disjoint from training.
The conservative ensemble gate is explicitly a deterministic mean-minus-spread
score, not a calibrated confidence interval. Publication hypervolume compares
ungated BA-PCDTL with ungated independent policies so fallback cannot inflate
the algorithmic Pareto claim. Non-balanced critic values are converted from
discounted-return units to rate-equivalent units before frozen-anchor
scalarization. At the balanced PNG preference, the rate-equivalent joint value
is divided only by one frozen positive span, preserving exactly the physical
`Q_c+Q_s` argmax while keeping actor utilities dimensionless. The hypervolume
reference is locked below normalized physical zero, and validators reconstruct
both anchors and the complete gate grid.

Preparation status, structural search, holdout, stress, preflight, calibrated
profile, calibration report, and power lock now carry and validate the same
source-tree digest. Figure 14 evaluates the DDPG comparator with the same locked
CSI hypothesis used during its training. Legacy extension validators now require
a positive paired confidence lower bound over the corresponding baseline; mere
feasibility can no longer be reported as improvement.

## Remaining Reproduction Boundary

The code does not implement the complete MMSE/CVX/SCA AO procedure in paper
Algorithms 2-3. Its AO path is a fixed-geometry constrained SciPy surrogate
with constructed ZF/matched-filter directions, stream-power allocation, direct
HRIS-control optimization, and terminal-iterate reporting. The paper also omits the executable coherence interval for
Eq. (19), the exact DDPG comparator CSI, partial-CSI mask construction, transfer
domains, and enough detail to recover Eq. (30) as unique code. These are locked
assumptions or report-all sensitivities, not hidden fitted choices.

Therefore this revision is an auditable implementation and experiment protocol,
not a guarantee that the paper's numerical curves have already been reproduced.
That claim requires fresh runs and passing validators, and AO agreement would
still be agreement of the disclosed surrogate rather than proof of author-solver
equivalence.

## Required Fresh Execution

Follow `REPRODUCTION_GUIDE.md` from preparation onward in new empty `runs`
directories. Do not copy the earlier calibration or checkpoints. The baseline
runner automatically records a no-training action-response and feasibility
diagnostic before creating training checkpoints; weak response never stops the
run. A failed calibration, baseline validator, or holdout validator means
reproduction has not been established and the extension must not be presented
as an improvement.
