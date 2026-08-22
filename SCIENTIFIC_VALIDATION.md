# Development Scientific Validation Record

## Purpose

This record distinguishes implementation verification from numerical paper
reproduction. Tests can detect broken contracts, but they cannot recover values
or algorithms that the journal does not publish.

## Rejected Intermediate Protocols

The earlier profile protocol used middle quantiles and accepted only `0.50`
joint feasibility. It achieved approximately `0.75` joint feasibility in an
independent stress check while full policy evaluation required every step to be
feasible. That protocol was internally inconsistent and was retired.

The first conservative revision used `1.25` headroom on upper constraints. A
48-step moving-channel development preflight found four analytical active-power
envelope violations, although actual sampled policy power remained feasible.
No figure or training result from that attempt was accepted. The upper-limit
headroom was changed to a declared factor of `2`, and fresh preflight seeds were
reserved before final validation.

## Revised Structural Protocol

- Candidate generator: deterministic scrambled SciPy Sobol prefix.
- Candidates: `32`.
- Active-gain probes per channel: zero, half, and full gain.
- Calibration seeds: `21000` through `21007`.
- Holdout seeds: `22000` through `22007`.
- Stress seeds: `23000` through `23031`.
- Dynamic preflight seeds: `25001` through `25004`, 12 steps each.
- Required feasibility fraction on every partition: `1.0`.
- Published curve ordinates used: no.
- Communication-rate magnitude used in identification score: no.

The active-power stress check uses an analytical Eq. (18) envelope valid for
every allowed active gain and every sensing beamformer satisfying the BS power
budget. `R_min`, `P_a_max`, and `Psi_max` remain disclosed fixed assumptions;
they are not derived from a candidate's observed rates or powers.

This structural preparation stage deliberately does not use paper curve
ordinates. The later `calibrate_reproduction.py` stage starts only from its
authorized identified profile and fits omitted channel parameters against the
Fig. 6 AO-pCSI/AO-iCSI curves. Fig. 7 and the other declared holdouts remain
unseen by calibration. The two stages must not be conflated.

## Development Profile

Profile SHA-256:

```text
2b5d4c3ecca57b1fe7b78aa3acd8bbc88ca0033e9c80e2346ae6cf4d8ecf3803
```

The digest above identifies the local development profile generated before the
final source-only handoff. It is evidence from debugging, not a paper result or
an author-provided profile. The receiving researcher must regenerate the
canonical `result/parameter_identification/identified_profile.json` from the
final code. That newly generated JSON becomes the numeric authority for her run.

## Development Outcomes

| Partition | Finite | BS power | Sensing | Active envelope | Cascade | Joint |
|---|---:|---:|---:|---:|---:|---:|
| Calibration | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Holdout | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 32-seed stress | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Moving-target preflight | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Thirty of 32 candidates passed strict calibration acceptance. The selected
candidate then passed the disjoint holdout, stress, and dynamic preflight.

## Claim Boundary

These local outcomes supported debugging of the revised protocol. They do not
replace the required colleague-side rerun, prove global feasibility over
an unbounded fading distribution, recover the authors' missing parameters, or
establish numerical agreement with Figs. 4-14.

The development outcomes above predate the final equation-level Doppler,
raw-AO-history, target-seed-assignment, and nonmonotone-projection corrections.
They are intentionally not reused as evidence for the final source tree.

The complete DRL sweep was not run during package preparation. Numerical paper
reproduction remains pending until the full run completes, every rollout stays
feasible, generated data pass cross-figure validation, and results are checked
over additional independent random seeds.

## BA-PCDTL Extension Verification

The extension implementation was verified without launching full DRL training.
Focused tests cover vector-reward replay, exact PNG scalarization, immutable
training-only anchors, Pareto hypervolume, paired intervals, zero-residual
baseline equivalence, vector Bellman targets, critic uncertainty fallback,
executed-physical-action replay with aggregate projection diagnostics,
deterministic Sobol resume, seed-role separation,
primitive-CSI state sufficiency, atomic next-channel/precoder transitions,
discounted-return/rate-unit Pareto normalization, validation-only gate locking,
full checkpoint identity plus internal history/completion and DTL-source-model
binding, globally
unique paired evaluation-scene assignment, publication-level
independent-policy comparison, deterministic same-channel feasibility recovery,
UAV return reachability/final closure, exact Eq. (9) active-subarray steering
with a one-aperture disjoint HRIS partition,
nonleaking training-tail objective references, repeated evaluation evidence,
and hashed plot evidence.

The validator rejects nonfinite arrays, objective-component inconsistencies,
post-hoc seed overlap, incomplete preferences, missing or invalid projection
evidence, violated UE/target floors,
physical infeasibility, checkpoint or gate-lock drift, pointwise envelopes,
nonpositive joint-rate confidence bounds, and nonpositive hypervolume confidence
bounds. These checks establish implementation integrity only. They do not imply
that the pending numerical experiment will pass.

## Final Source Verification

The final source was checked without DRL training using the complete fast test
suite and Python bytecode compilation. The suite covers equation contracts,
power units, SCEE covariance, objective separation, hard feasibility,
nonmonotone action projection, same-channel recovery, final-slot UAV closure,
per-stream allocation, per-element-power Eq. (16c), CSI-state semantics, source-bound
preparation/calibration, raw curve validation,
paired seeds, checkpoint/RNG restoration, holdout provenance, and delivery
hashes. It also covers both noise mappings, all three Equation 30 readings, both
transfer schedules, both WFB replacements, both DDPG comparator CSI readings,
both CSI-observation readings, both zero-error rate-function readings,
hypothesis fingerprints, and the
non-selecting sensitivity plan/report. Hashed plot sidecars cannot mix source,
profile, seed, power, noise, transfer, schedule, WFB, DDPG-comparator CSI,
CSI-observation, rate-model, or cross-interference provenance within one
bundle. Numerical curve agreement is deliberately outside this static-test
claim.

The final audit also checks that all paper-declared holdouts (Figures 4, 5, 7,
8, 9, 13, and 14) are excluded from calibration, and that any omitted-value
`--fixed-sensitivity` branch is predeclared and identical in its calibrated
profile and calibration report.
