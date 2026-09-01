# Evaluation status and claim boundary

This project is ready to share as a research prototype for collaboration. It is
not yet evidence for unattended proceedings ingestion.

## Results that can be stated

- The immutable first run on a ten-paper holdout recovered `5/10 = 0.5` selected
  targets. This is selected-target recall over one audited target per paper, not
  whole-paper recall.
- After a generic composition fix and bounded technical retry, the same ten
  inspected papers yielded `7/10 = 0.7`. This is explicitly post-hoc and must
  not be presented as a held-out result.
- The current open-development run completed all 380 selected legacy extraction
  blocks. Its bounded row plan accounted for all 334 planned rows: 315 received
  a typed disposition (`177` result, `94` not-result, `44` uncertain), 17
  remained unresolved, and two were explicitly unbatchable.
- The run reduced 1,297 proposals to 1,078 candidates by removing 219 duplicate
  proposals. Of the retained candidates, 1,055 require review and 23 are not
  eligible. No candidate established positive `PAPER_PRODUCED` origin, so zero
  canonical EEE records were emitted.
- Candidate-layer recall over the pre-existing open-development references was
  `106/109 = 0.972477` micro and `0.7` macro. The denominator is not a whole-paper
  gold standard, so this is not canonical-EEE recall, holdout evidence, or
  generalization evidence.
- The deterministic suite spans source freezing, extraction contracts, evidence
  checks, row planning, attribution, composition, replay, review cards, and
  privacy boundaries.

## Results that cannot yet be stated

- Precision is unmeasured. The previously reported `0.083333` was caused by an
  invalid scoring denominator and is withdrawn. The corrected `1/2` basis is
  too small to support a rate and must not be relabelled as `0.5` precision.
- The development run does not establish the correctness of the 315 row
  dispositions. That requires independent human annotation with genuine
  negative and mixed cases.
- Complete-tuple correctness, non-result-row specificity, false-positive rate,
  and current-version generalization have not been established through an
  independently annotated sample.
- The present deterministic attribution resolver can demote or abstain but
  cannot positively assert that a paper produced a result. Canonical EEE output
  can therefore be empty even when the review layer found useful candidates.

## Current public development artifact

The quote-free aggregate is
[`results/current-development-summary.json`](../results/current-development-summary.json).
Its technical status is `partial_failure` because 17 batchable rows remain
unresolved and two rows are typed as unbatchable, not because a legacy
extraction block failed. All planned rows are partitioned and accounted for.

The bounded resume that completed the two missing legacy blocks made six new
structured invocations, all on the open-development corpus, for a reported cost
of `$0.1855958`. All 378 already successful legacy blocks and all 120 row
batches were reused. The public JSON reports artifact-basis usage separately:
155 retained call records and a `$1.30342832` lower-bound cost, comprising the
six new legacy calls plus 149 restored row-call records. It is not a complete
cross-invocation billing ledger.

Regenerate the public projection offline with:

```bash
uv run ere build-public-development-summary runs/my-development-run \
  --output results/current-development-summary.json
```

The command is deliberately incompatible with the legacy run shape and refuses
holdout or unclassified corpus bindings, internally inconsistent row ledgers,
and canonical exports without positive `PAPER_PRODUCED` origin. A run with
typed unbatchable rows is reported as bounded partial completion; those rows
are not silently counted as resolved.

The next scientific measurement is an independently annotated, predeclared
sample with result rows, externally sourced results, setup and sample-count
rows, parameter rows, headings, and uncertain or mixed rows. The completed
single-annotator exploratory audit is not independent validation and is not
included in the public aggregate.
