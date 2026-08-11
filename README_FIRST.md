# HRIS-UAV-ISAC Reproduction and BA-PCDTL Research Extension

Start with `REPRODUCTION_GUIDE.md`. This package has two sequential goals:

1. reproduce the journal's communication-sum-rate ISAC objective, with target
   sensing retained as a hard constraint;
2. evaluate the supplied joint communication-sensing PNG objective with the
   proposed Baseline-Anchored Preference-Conditioned Deep Transfer Learning
   (BA-PCDTL) method.

The supplied code already contained sensing rates and weighted sweeps. The
research contribution is therefore not "adding sensing." BA-PCDTL instead uses
a frozen accepted policy, a conditioned residual actor, vector twin critics,
training-only Pareto anchors, and a validation-locked conservative gate.

The exact PNG objective is:

```text
maximize sum_u R_u + sum_k R_k
```

subject to the printed per-UE, per-target, power, HRIS, and UAV constraints. No
reward penalty, artificial sensing multiplier, or communication-retention
percentage is part of this literal objective. The older
`comm_retention_sensing` mode remains only as an explicitly labeled sensitivity.

All training runners checkpoint complete mutable state and resume after an
interruption when the identical command is rerun. Do not edit checkpoints,
manifests, seed lists, gate locks, or result arrays.

Every executed transition is checked for signal, power, HRIS, and UAV
feasibility, including return reachability and exact final closure. Recovery is
deterministic on the same sampled channel; it never resamples a favorable
scene. Source-tree hashes make older results and checkpoints inadmissible after
code changes, so this final package requires a clean run.

Static source verification does not train DRL models:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall -q .
```

No numerical superiority is bundled or guaranteed. A preliminary claim needs
`validate_pareto_extension.py --level preliminary` to pass. A publication
claim additionally needs the complete independent-policy comparison and
`--level publication` to pass on untouched paired test scenes.
