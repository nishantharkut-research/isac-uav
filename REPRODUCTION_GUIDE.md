# HRIS-UAV-ISAC Reproduction and BA-PCDTL Guide

## 1. Scope

This package supports two sequential studies:

1. **Journal reproduction:** communication sum-rate maximization with sensing
   rate and physical constraints.
2. **Research extension:** exact joint communication-sensing optimization with
   BA-PCDTL and paired Pareto evaluation.

The package contains source code, tests, the paper PDF, the supplied problem
PNG, profiles, and documentation. It contains no accepted trained result.
Numerical conclusions become valid only after the declared runs finish and the
corresponding validators return exit code `0`.

## 2. Scientific Boundary

- Journal baseline reward: `sum_u R_u` in `bits/s/Hz`.
- Supplied PNG objective: `sum_u R_u + sum_k R_k` in `bits/s/Hz`.
- Reported rewards contain no constraint penalty.
- The PNG objective contains no fixed `95%` communication-retention rule.
- Per-UE and per-target floors remain hard constraints.
- Phase, active gain, BS power, active-HRIS power, passive cascade, UAV bounds,
  UAV step, and UAV closure remain hard physical constraints.
- Replay stores the action executed after hard feasibility projection.
- Every transition checks UAV return reachability and exact final-slot closure.
  A failed proposal is recovered deterministically on the same sampled channel;
  neither channel nor scene is resampled to obtain a better result.
- Publication evaluation uses the SciPy constrained stream-power allocator.
- Training uses the deterministic global fast grid first and retries SciPy on
  the same channel/control when fast allocation alone cannot establish
  feasibility. Previous stream powers are warm starts only.
- Test scenes never select a model, normalizer, gate parameter, preference, or
  hyperparameter.
- A larger plotted curve alone is not evidence of improvement.

No neural method can guarantee strict improvement on unseen stochastic
channels. The code therefore uses a fail-closed statistical acceptance rule
instead of a guaranteed-improvement claim.

## 3. What BA-PCDTL Adds

The supplied implementations already modeled target sensing and swept fixed
communication-sensing weights. BA-PCDTL adds the following algorithmic layer:

- a frozen accepted `DTL-pCSI` baseline actor;
- a bounded preference-conditioned residual actor;
- exact zero-residual initialization, reproducing the baseline action;
- two or more twin vector-critic pairs with separate communication and sensing
  outputs;
- communication-trunk transfer without copying a communication scalar into the
  sensing output;
- deterministic Sobol preference relabeling;
- unnormalized `Q_comm + Q_sense` actor updates for the exact PNG point;
- frozen-anchor Chebyshev scalarization for non-PNG Pareto preferences;
- a current-state critic lower-confidence gate with baseline fallback;
- validation-only gate selection and an immutable lock before test evaluation;
- fair same-objective, scratch, independent-policy, and ablation comparisons.

The conservative gate is a neural decision rule, not a theorem about the true
return. Nonfinite estimates or insufficient conservative advantage execute the
frozen baseline proposal.

## 4. Installation

Open PowerShell in the extracted package directory.

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

SciPy is required. It supplies constrained allocation, statistical tests, and
the deterministic Sobol preference sequence.

Run source verification. These commands do not start DRL training.

```powershell
python -m pytest -q
python -m compileall -q .
```

Do not continue if either command fails.

This source tree has its own immutable digest. Do not copy checkpoints or result
folders from an older package into these `runs` directories. Any source change
requires new baseline and extension runs.

## 5. Prepare Omitted Parameters

The paper omits several channel, constraint, network, and transfer values. The
preparation stage searches only bounded structural assumptions and does not use
paper reward curves.

```powershell
python prepare_reproduction.py `
  --output runs\01_prepared `
  --candidates 32 `
  --seed 2025
```

Required output includes `identified_profile.json` and successful structural,
stress, holdout, and preflight reports. Do not edit a failed status file.

`omega_0` is searched within its declared evidence bounds. The primary element
spacing remains the cited half-wavelength value `d_spacing=0.5`, and carrier
frequency remains the disclosed code assumption `fc=3e9`. Do not change either
to maximize reward. Section 16 shows how to predeclare conditional sensitivity
branches and report every branch.

## 6. Calibrate the Printed Power Ambiguity

The paper labels transmitter power in `dB` without stating dBm or dBW. The code
evaluates both predeclared interpretations on Figure 6 calibration data and
locks the feasible branch before any holdout or RL result is viewed.

```powershell
python calibrate_reproduction.py `
  --base-profile runs\01_prepared\identified_profile.json `
  --reference references\paper_reference_curves.json `
  --output runs\02_calibration `
  --seeds 21000,21001,21002,21003,21004 `
  --ao-iterations 6 `
  --max-iterations 30
```

Resolve the selected profile:

```powershell
$calibrationRoot = Resolve-Path runs\02_calibration
$lock = Get-Content "$calibrationRoot\power_hypothesis_lock.json" -Raw |
  ConvertFrom-Json
$profile = Join-Path $calibrationRoot $lock.selected_profile
$profile
```

Do not switch the locked branch after inspecting another result.

Calibration fails instead of writing a selection lock when neither branch has
the required pCSI/iCSI ordering, monotone power direction, at least half the
paper's total power response, and normalized curve error at most `0.15`. A
failure here is a model-identification failure; do not proceed to training.

For an omitted-parameter sensitivity branch, repeat the complete calibration in
a new empty output directory and fix the value before any simulated curve is
seen. For example:

```powershell
python calibrate_reproduction.py `
  --base-profile runs\01_prepared\identified_profile.json `
  --reference references\paper_reference_curves.json `
  --output runs\02_calibration_fc_2p4ghz `
  --seeds 21000,21001,21002,21003,21004 `
  --ao-iterations 6 `
  --max-iterations 30 `
  --fixed-sensitivity fc=2.4e9
```

`--fixed-sensitivity` is repeatable and accepts only parameters classified as
sensitivity-only in `parameter_evidence.py`. It rejects reported values and
structural calibration variables. The fixed value is hash-bound into both the
profile and calibration report as `predeclared_not_result_selected`. This is a
conditional calibration branch: nuisance parameters are recalibrated on Figure
6, but the fixed parameter itself is never chosen from result quality.

## 7. Lock the Seed Roles

Use at least ten independent policy-training seeds, five validation scene
seeds, and ten untouched test scene seeds. The three sets must not overlap.

```powershell
$researchSeeds = "2025,2026,2027,2028,2029,2030,2031,2032,2033,2034"
$validationScenes = "12001,12002,12003,12004,12005"
$testScenes = "13001,13002,13003,13004,13005,13006,13007,13008,13009,13010"
```

Twenty test scenes are preferred for a final journal submission. Changing seed
roles requires a new output directory and a new experiment declaration.

## 8. Reproduce the Journal Baseline

The research extension requires a passing baseline generated by this exact
final source tree and the same ten research seeds.

First create the manifest without training:

```powershell
python run_baseline.py `
  --profile $profile `
  --output runs\03_baseline `
  --seeds $researchSeeds `
  --episodes 1500 `
  --eval-episodes 5 `
  --checkpoint-every 1 `
  --dry-run
```

Then rerun without `--dry-run`:

```powershell
python run_baseline.py `
  --profile $profile `
  --output runs\03_baseline `
  --seeds $researchSeeds `
  --episodes 1500 `
  --eval-episodes 5 `
  --checkpoint-every 1
```

Before the first training checkpoint is created, this command runs a bounded
no-training physical sanity gate on three deterministic scenes. Every probe
must be feasible and the median communication response across gain/phase
actions must be at least `0.005` relative span. A failure prevents an expensive
run whose action space is effectively flat.

Validate the complete raw baseline:

```powershell
python validate_results.py `
  --baseline runs\03_baseline\baseline_summary.json `
  --reference references\paper_reference_curves.json `
  --output runs\03_baseline\baseline_validation.json
```

A nonzero exit code means the journal reproduction has not passed. The
research extension will refuse a failed or mismatched baseline.

Optional Figure 4 evidence:

```powershell
python plot_baseline.py `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --reference references\paper_reference_curves.json `
  --output runs\03_baseline\figure4_evidence
```

## 9. Reproduce the Remaining Quantitative Paper Figures

The paper-level reproduction also requires the declared quantitative holdouts.
Figure 6 is the only calibration figure. Figures 5, 7, 8, 9, 13, and 14 remain
untouched until this stage. First create and inspect the resumable run plan:

```powershell
python run_holdouts.py `
  --profile $profile `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --output runs\04_paper_holdouts `
  --seeds $researchSeeds `
  --episodes 1000 `
  --eval-episodes 5 `
  --checkpoint-every 1 `
  --dry-run
```

Rerun the same command without `--dry-run` to execute or resume:

```powershell
python run_holdouts.py `
  --profile $profile `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --output runs\04_paper_holdouts `
  --seeds $researchSeeds `
  --episodes 1000 `
  --eval-episodes 5 `
  --checkpoint-every 1
```

Validate raw aggregation, provenance, qualitative contracts, and available
digitized references before plotting:

```powershell
python validate_holdouts.py `
  --summary runs\04_paper_holdouts\holdout_summary.json `
  --reference references\paper_reference_curves.json `
  --output runs\04_paper_holdouts\holdout_validation.json

python plot_holdouts.py `
  --summary runs\04_paper_holdouts\holdout_summary.json `
  --reference references\paper_reference_curves.json `
  --output runs\04_paper_holdouts\figures
```

Figures 10-12 are descriptive only: the paper does not provide enough numeric
trajectory/beampattern authority or a portable runtime protocol for defensible
quantitative acceptance. The holdout runner is a long training/optimization
workflow; it checkpoints each declared unit and never chooses a scenario from
its observed result.

The paper does not provide the authors' CVX implementation or every auxiliary
matrix needed to reproduce Algorithms 2-3 byte-for-byte. The bundled AO method
is a disclosed constrained SciPy surrogate with a one-step trajectory block.
A passing holdout means its generated curves agree with the declared paper
tolerances; it does not claim recovery of unpublished author source code.

## 10. Inspect the BA-PCDTL Plan Without Training

Use a new empty directory. This command validates baseline authority, budgets,
seed separation, objective floors, preferences, gate grids, physics, profile,
ambiguity identities, and final source code. It does not train.

```powershell
python run_pareto_extension.py `
  --profile $profile `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --output runs\05_ba_pcdtl_primary `
  --research-seeds $researchSeeds `
  --validation-scene-seeds $validationScenes `
  --test-scene-seeds $testScenes `
  --episodes 1500 `
  --eval-episodes 5 `
  --checkpoint-every 1 `
  --comparison-level primary `
  --dry-run
```

The default nine-point preference grid includes both endpoints and the exact
balanced PNG point. The default residual scales are
`0,0.25,0.5,0.75,1`. Defaults are part of the immutable manifest.

## 11. Run the Primary Research Pipeline

Remove only `--dry-run` and keep every other argument identical.

```powershell
python run_pareto_extension.py `
  --profile $profile `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --output runs\05_ba_pcdtl_primary `
  --research-seeds $researchSeeds `
  --validation-scene-seeds $validationScenes `
  --test-scene-seeds $testScenes `
  --episodes 1500 `
  --eval-episodes 5 `
  --checkpoint-every 1 `
  --comparison-level primary
```

Primary mode trains:

- same-objective source/target joint-sum DTL;
- cross-objective scalar DTL, labeled diagnostic only;
- joint-objective DDPG from scratch;
- full BA-PCDTL;
- BA-PCDTL evaluation with and without its gate.

It freezes global Pareto anchors from scalar-comparator training data, evaluates
the preregistered gate grid only on validation scenes, writes
`gate_selection_lock.json`, and opens test scenes only afterward.
The run manifest also locks SHA-256 hashes for every accepted baseline source
and `DTL-pCSI` checkpoint and rechecks them on every restore.

This is a long multi-seed DRL experiment. Runtime depends on hardware,
environment projection frequency, SciPy evaluation, and checkpoint interval.

## 12. Run the Full Publication Comparison

Publication validation requires a separate `--comparison-level full` run in a
new directory. Full mode additionally trains:

- one fair fixed-preference vector policy per preference;
- unanchored preference-conditioned vector TD3;
- baseline-anchored BA-PCDTL with random trunks;
- the no-gate evaluation ablation.

```powershell
python run_pareto_extension.py `
  --profile $profile `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --output runs\06_ba_pcdtl_full `
  --research-seeds $researchSeeds `
  --validation-scene-seeds $validationScenes `
  --test-scene-seeds $testScenes `
  --episodes 1500 `
  --eval-episodes 5 `
  --checkpoint-every 1 `
  --comparison-level full `
  --dry-run
```

Rerun without `--dry-run` to execute. Full mode is substantially longer than
primary mode because it trains the complete independent-policy grid for every
research seed.

## 13. Validate the Research Result

Preliminary validation:

```powershell
python validate_pareto_extension.py `
  --summary runs\05_ba_pcdtl_primary\pareto_extension_summary.json `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --level preliminary `
  --output runs\05_ba_pcdtl_primary\pareto_validation.json
```

Publication validation:

```powershell
python validate_pareto_extension.py `
  --summary runs\06_ba_pcdtl_full\pareto_extension_summary.json `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --level publication `
  --output runs\06_ba_pcdtl_full\pareto_validation.json
```

Preliminary acceptance requires:

- all seed roles and raw test arrays complete and paired;
- no nonfinite, smoothed, or pointwise-envelope acceptance data;
- exact PNG objective identity;
- `100%` physical feasibility and every UE/target floor satisfied;
- unchanged checkpoint, baseline, anchor, and gate-lock provenance;
- a strictly positive lower bound of the paired 95% confidence interval for
  BA-PCDTL minus same-objective DTL joint rate;
- a strictly positive lower bound of the paired 95% confidence interval for
  BA-PCDTL minus the accepted paper-objective `DTL-pCSI`, reevaluated under the
  same PNG objective and pCSI test environment.

Publication acceptance additionally requires the complete fair independent
policy grid and a strictly positive paired 95% confidence-interval lower bound
for normalized Pareto hypervolume improvement.

## 14. Plot Hashed Evidence

```powershell
python plot_pareto_extension.py `
  --summary runs\06_ba_pcdtl_full\pareto_extension_summary.json `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --validation-level publication `
  --output runs\07_ba_pcdtl_figures
```

The command creates:

- `training_convergence.png`;
- `primary_paired_delta.png`;
- `pareto_front.png`;
- one raw JSON sidecar per image;
- `pareto_validation.json`;
- `result_manifest.json` with image and sidecar SHA-256 hashes.

Any causal smoothing is visualization-only. Validation uses unsmoothed arrays.
The Pareto plot draws each method separately and never constructs a pointwise
best-method envelope.

## 15. Resume After Power Failure

Rerun the exact same command with the same output directory. Do not add a fresh
flag, change a seed, change the episode target, or edit the manifest.

Each episode-boundary checkpoint stores:

- actor, target actor, every critic, and target critic;
- optimizer states;
- vector replay transitions and executed actions;
- frozen baseline actor and its digest;
- Pareto anchors and provenance;
- the canonical Pareto-anchor digest in both agent and run identity;
- Sobol sequence seed and position;
- residual/gate/algorithm contract;
- raw histories;
- environment state;
- Python, NumPy, Torch, and CUDA RNG states.

Atomic replacement keeps the previous complete checkpoint if a write is
interrupted. With `--checkpoint-every 1`, at most the current unfinished episode
is repeated.

The canonical reset control may be signal-infeasible and is recorded as such;
reset has no reward and enters no replay transition. Every actual transition
must pass all hard constraints.

## 16. Robustness Study

Copy `profiles/pareto_robustness_study.example.json` to a new design file and
replace every placeholder. Each scenario needs its own profile, accepted
baseline summary, accepted baseline validation, and output directory.

Create each omitted-parameter profile with a separate predeclared
`calibrate_reproduction.py --fixed-sensitivity KEY=VALUE` run. Then run and
validate `run_baseline.py` for that exact profile before putting it in the study
design. Do not reuse the primary baseline under a different profile digest.

Check all scenario manifests without training:

```powershell
python run_pareto_study.py `
  --design profiles\pareto_robustness_study.json `
  --dry-run
```

Run or resume sequentially:

```powershell
python run_pareto_study.py `
  --design profiles\pareto_robustness_study.json `
  --state runs\pareto_robustness_state.json
```

The bundled design contains executable examples for carrier frequency, element
spacing, source-task SCEE NMSE, target mobility, and the extension UE-rate
floor. Add other sensitivity-only profile keys only when they are preregistered.
The journal-reported transmit-power and HRIS-element sweeps belong to the paper
figure pipeline, including Figures 5-9, 13, and 14; they are not
calibration-profile overrides. BS antenna counts are fixed by Table III and
must not be relabeled as an omitted-value robustness profile.

The code uses explicit, disjoint source and target scene seeds because the paper
does not define executable transfer domains. Report that assumption. A future
transfer-domain study requires a separately specified protocol and code change;
the current Pareto design does not pretend that a profile can override those
seeds.
The orchestrator never adds a scenario because its observed reward is larger.

The journal also omits the CSI condition used by its standalone DDPG curve. The
primary run uses the fair pCSI comparator, while the iCSI comparator is a
non-selecting ambiguity sensitivity. To run the practical one-factor matrix:

```powershell
python run_ambiguity_sensitivity.py `
  --profile $profile `
  --output-root runs\ambiguity_sensitivity `
  --seeds $researchSeeds `
  --episodes 1500 `
  --eval-episodes 5 `
  --checkpoint-every 1 `
  --primary-baseline runs\03_baseline\baseline_summary.json `
  --primary-validation runs\03_baseline\baseline_validation.json
```

This base-reproduction ambiguity study executes eight isolated alternatives to
the accepted primary: noise mapping, Equation 30 update, transfer schedule, WFB
replacement, DDPG iCSI, LoS symbol-role mapping, independent slot fading, and
receiver-consistent cross-function interference. The report contains paired
deltas and no selected winner. The optional `--design full_factorial` executes
all 256 predeclared combinations and is substantially more expensive.

## 17. Legacy Sensitivity Modes

`run_extension.py` remains for explicit sensitivity studies:

- `png_joint_sum`: exact unweighted PNG objective, no `--retention` argument;
- `comm_retention_sensing`: sensing objective with an explicitly declared
  retention fraction;
- `normalized_pareto`: legacy fixed normalized weighting.

Example retention sensitivity:

```powershell
python run_extension.py `
  --profile $profile `
  --baseline-summary runs\03_baseline\baseline_summary.json `
  --baseline-validation runs\03_baseline\baseline_validation.json `
  --output runs\legacy_retention `
  --seeds $researchSeeds `
  --objective comm_retention_sensing `
  --retention 0.95
```

These modes are not the BA-PCDTL contribution. A reward-scaled weighted sweep
must be labeled historical and is not the strongest publication comparator.

## 18. Failure Interpretation

- **Baseline validation fails:** do not start or claim the extension.
- **Source-tree mismatch:** regenerate the baseline with this exact package.
- **Checkpoint hash or identity mismatch:** restore the original payload or use
  a new output directory; do not edit the checkpoint.
- **Same-channel recovery exhausted:** preserve proposal, previous-control, and
  deterministic-candidate diagnostics; do not resample the scene.
- **Gate-lock mismatch:** test evidence is invalid and must not be combined.
- **Rate-floor or feasibility failure:** the method fails the physical contract.
- **Joint confidence interval crosses zero:** improvement over same-objective DTL
  is not established.
- **Hypervolume confidence interval crosses zero:** publication-level Pareto
  improvement is not established.
- **Nonfinite critic or result:** preserve the last checkpoint and diagnostic;
  do not replace the value.

Failed results remain valid negative evidence. Do not change test seeds,
normalizers, smoothing, objectives, or acceptance thresholds after viewing the
failure.

## 19. Reporting

Report the exact profile digest, source-tree digest, baseline payload digest,
seed roles, episode budget, preference grid, objective floors, gate lock,
checkpoint hashes, raw component rates, paired intervals, feasibility, and all
failed checks.

Call BA-PCDTL superior only if the relevant validator reports
`overall_pass: true`. Until colleague-side full runs pass, this package is a
verified implementation and experiment protocol, not numerical proof of
reproduction or improvement.
