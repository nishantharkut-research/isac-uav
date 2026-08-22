# Paper Reproduction Hardening Plan

## Scope

This plan covers code correctness for the Saikia et al. baseline and the
separately supplied communication-plus-sensing extension. It does not authorize
training, claim numerical curve agreement, or reconstruct unpublished author
code. Every irrecoverable choice is fixed before results are viewed and is bound
to manifests/checkpoints.

## Evidence Boundary

- Exact paper equations are primary when they are executable as printed.
- Missing mappings are represented as closed, report-all sensitivity axes.
- The SciPy AO implementation is an `AO_SURROGATE`, not Algorithms 2-3.
- Reported rewards are physical objective values and never include penalties.
- Fresh colleague-side runs and validators are required for numerical claims.

## Tasks

1. **Eq. (16c) contract** - completed
   - Sum per-element squared cascade norms as printed.
   - Enforce the omitted user index using a disclosed worst-user assumption.
   - Protect against coherent-cancellation regression.

2. **CSI quality versus state observability** - completed
   - Separate pCSI/iCSI estimation quality from statistical/instantaneous state.
   - Add a primary mapping and a report-all alternative.
   - Bind the identity to checkpoints, manifests, fingerprints, and plot evidence.

3. **DTL transfer semantics** - completed
   - Preserve the complete source mapping when target state appends CSI features.
   - Evaluate frozen source Q on the exact shared state prefix.
   - Fail closed when action semantics differ or target state omits source fields.
   - Represent literal Eq. (30) separately from researcher-defined Bellman readings.

4. **Training and resume determinism** - completed
   - Set exploration noise before the first action of every episode.
   - Declare its unreported floor and fingerprint all learning hyperparameters.
   - Keep episode-boundary atomic checkpoints and reject incompatible restores.

5. **AO surrogate integrity** - completed
   - Remove hidden module-global optimization context.
   - Keep geometry fixed for paper comparisons; AO iterations are not UAV slots.
   - Use terminal feasible iterates for comparisons; best iterates are diagnostic.
   - Remove exact WMMSE/SCA/Algorithms 2-3 claims and record solver assumptions.
   - Fail closed if exact author AO is requested.

6. **Figure and omission disclosure** - completed
   - Record one-stream and active/passive-split assumptions for every affected figure.
   - Correct LoS equation numbering and channel-period wording.
   - Update ambiguity scenario counts and running instructions.
   - Declare scalar-pCSI versus matrix-SCEE zero-error handling as a
     fingerprinted report-all ambiguity.

7. **Evaluation and attribution integrity** - completed
   - Store executed actions in replay and preserve proposals as diagnostics.
   - Use one allocator contract for train, validation, and test rollouts.
   - Derive extension references only from accepted baseline training tails.
   - Execute and preserve every configured evaluation episode.
   - Compare ungated Pareto methods and enforce projection-attribution limits.

8. **Physical aperture correction** - completed
   - Implement Eq. (9)'s `La=Lax*Lay` active rectangular steering exactly.
   - Place disjoint active/passive rectangular subarrays on one adjacent aperture.
   - Retain the former interleaved full-UPA reading only as a named sensitivity.
   - Bind aperture dimensions and partition identity to physical fingerprints.

9. **Diagnostic smoke protocol** - completed
   - Permit checkpointed 100/200-episode baseline diagnostics.
   - Mark smoke output ineligible for paper acceptance or numerical claims.
   - Keep 25-episode status, smoke checks, and pretraining response diagnostics
     advisory-only; none may terminate training for a weak curve.

10. **State, transition, and Pareto-unit correction** - completed
   - Expose primitive estimated CSI rather than phase-dependent composites.
   - Score the observed channel, then construct next channel/precoders atomically.
   - Keep the terminal state internally consistent without an unmatched target move.
   - Convert vector-critic discounted returns to rate-equivalent anchor units.
   - Lock hypervolume below normalized physical zero and reconstruct anchors/grid.

11. **Figure 11 aperture consistency** - completed
   - Use the primary Eq. (9) active-subarray coordinates.
   - Derive target signatures and diagonal loading from actual estimated two-hop
     channels and active noise, not synthetic angle-only data or arbitrary NMSE.

12. **Result-evidence authority** - completed
   - Pair all methods on one globally unique deterministic scene schedule.
   - Serialize the exact gate configuration used for every validation rollout.
   - Bind result histories to the internal identity and completion count in each
     checkpoint, in addition to file hashes and JSON identity sidecars.
   - Recompute serialized source-network digests and bind every DTL checkpoint
     to the exact accepted source checkpoint.
   - Apply calibrated-profile authority checks equally to dry runs and real runs.

13. **Verification and publication** - in progress
   - Run focused regressions, then the complete unit/static suite without training.
   - Perform independent final code/diff review against all rendered paper pages.
   - Commit and push only the reviewed branch; update draft PR #2 without CI/merge.
