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
| A | SCEE rates jump when NMSE changes from zero | Confirmed symptom. The immediate cause was a switch from scalar Eqs. (12)/(14) at zero error to matrix Eqs. (21)/(25) at any positive error, not the paper-required use of estimated CSI in the numerator. The matrix model is now used continuously, including at zero error. |
| B | Eq. (6) LoS constants appear swapped | Confirmed paper/model ambiguity, not a uniquely recoverable author value. Printed Eq. (6) is the reproduction primary because it also matches the paper's high-LoS/free-space text; the standard Al-Hourani role mapping is a predeclared non-selecting sensitivity. |
| C | Eqs. (12)/(14) cross-function channels are likely inconsistent | Confirmed ambiguity. The printed equations remain primary for reproduction; receiver-consistent cross channels are a report-all sensitivity. Both scalar and matrix rate paths implement the same selected hypothesis. |
| D | Power selection accepts pCSI/iCSI inversion | Confirmed and fixed. Any candidate with an ordering violation is inadmissible. |
| E | Constraint values were derived from one arbitrary pre-calibration candidate | Confirmed and removed from the identification path. `R_min`, `Psi_max`, and `P_a_max` are disclosed omitted-parameter assumptions/sensitivities, not result-derived structural parameters. |
| F/G | Power response and policy-action response are nearly flat | Confirmed as required fail-closed invariants. Structural selection checks action response, Figure 6 calibration checks power response and curve fidelity, and a deterministic pre-training action-response gate runs before any real baseline training. |
| J | SLSQP repeatedly recomputes identical assessments | Confirmed and fixed with exact-input assessment reuse. |
| K | BA-PCDTL resume does not bind Pareto anchors | Confirmed and fixed. The canonical anchor digest is part of both agent and run checkpoint contracts. |
| L | Pareto anchors include early random episodes and extrema | Confirmed risk and fixed. Anchors use the converged training tail and 5th/95th quantiles, with estimator provenance frozen into the anchor identity. |
| I | Sensing and cascade-error loops are slow | Confirmed and vectorized. Direct elementwise-equivalence tests protect the numerical equations. |
| M | Eq. (19) helper has an off-by-one guard and is not in the live pipeline | Guard fixed. The helper remains an explicit utility because the paper does not publish the coherence interval needed to apply it to the reported curves. |

## Additional Defects Found During Correction

The policy state exposed one instantaneous CSI realization, but the environment
sampled another small-scale realization before evaluating the action. The
transition order now evaluates the proposal on the realization represented by
the current state, then samples the next realization before returning the next
instantaneous-CSI state. This follows the order in Algorithm 1 without making
the observed channel irrelevant to the action. Episode-block fading remains a
declared sensitivity, and coherence identity plus latent seed are
checkpoint-bound.

The implementation also exposed estimated channel coefficients in every iCSI
state even though Section III says statistical CSI contains HRIS phases and
precoders, with coefficients added for instantaneous CSI. Statistical states
now omit channel coefficients; pCSI includes them; ParCSI explicitly opts into
a deterministic masked channel observation.

Equation (16c) was implemented as a sum of per-element powers. The paper places
the passive-element sum inside the squared norm, so cross-element coherent terms
were missing. The implementation now evaluates the coherent matrix sum before
the squared Frobenius norm.

Replay previously stored the feasibility-projected action while actor and target
updates queried the critic at the raw policy proposal. Replay now stores the
proposal and treats deterministic projection as environment dynamics. Both the
proposal and executed action remain in transition diagnostics.

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
The equal active/passive split is a disclosed assumption required to reconcile
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
CSI, LoS probability, channel coherence, and cross-function interference. The
one-at-a-time design has one primary plus eight alternatives. The optional full
factorial has 256 scenarios. Sensitivity reports contain paired deltas but no
winner-selection field.

## Checkpoint and Evidence Integrity

Scalar DDPG/DTL and BA-PCDTL checkpoints contain model, optimizer, replay,
environment, histories, and RNG state. BA-PCDTL additionally binds frozen
Pareto anchors and preference-stream position. Run manifests and plot sidecars
include every physical ambiguity identity. A source, profile, anchor, seed,
objective, or hypothesis mismatch fails resume and evidence validation.

Preparation status, structural search, holdout, stress, preflight, calibrated
profile, calibration report, and power lock now carry and validate the same
source-tree digest. Figure 14 evaluates the DDPG comparator with the same locked
CSI hypothesis used during its training. Legacy extension validators now require
a positive paired confidence lower bound over the corresponding baseline; mere
feasibility can no longer be reported as improvement.

## Remaining Reproduction Boundary

The code does not implement the complete MMSE/CVX/SCA AO procedure in paper
Algorithms 2-3. Its AO path is a constrained SciPy surrogate with beam
direction/power allocation, direct phase optimization, and a one-step UAV
trajectory block. The paper also omits the executable coherence interval for
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
runner automatically performs a no-training action-response and feasibility
gate before creating training checkpoints. A failed calibration, pre-training
gate, baseline validator, or holdout validator means reproduction has not been
established and the extension must not be presented as an improvement.
