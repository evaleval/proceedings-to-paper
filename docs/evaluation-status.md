# Evaluation status

ERE is an operational research prototype. Its implementation can extract candidates,
preserve source evidence, support review, and compose schema-valid EEE. The accuracy
of the current real-paper output has not been established by human annotation.

This page describes saved census artifacts dated **2026-09-05**. Preparing the
public repository did not rerun paid extraction. The code snapshot and those saved
artifacts must not be treated as an independently reproduced experiment.

## Latest saved counts

| Quantity | Count | Interpretation |
| --- | ---: | --- |
| Usable corpus papers | 3,251 | Available working corpus, not an evaluated gold standard |
| Papers processed | 105 | Run coverage |
| Papers with candidates | 99 | Candidate-bearing subset of processed papers |
| Candidate observations | 4,778 | Proposed results retained for checking and review |
| Candidates with textual support | 4,202 | Local evidence support, not semantic correctness |
| Schema-valid tiered EEE records | 317 | Output under an explicitly weaker export policy |
| Papers represented in tiered EEE | 56 | Output coverage, not paper completeness |
| Observations represented in tiered EEE | 1,192 | Several observations can share one EEE record |
| Canonical `positive_only` EEE records | 0 | No output passed that export policy in the saved census |

The [machine-readable summary](../results/census-summary-2026-09-05.json) is an
allowlisted aggregate. Real-paper source files, individual candidate data, evidence
quotations, and private annotations are not distributed with it.

The latest local work plan records 3,080 unstarted papers and a 60-paper evaluation
reserve. These planning counts are not a claim that all remaining papers have the
same availability or execution state.

## What the exported tiers mean

| Record-level tier or basis | Records |
| --- | ---: |
| `model_reviewed` tier | 313 |
| `deterministic` tier | 4 |
| `human_confirmed` tier | 0 |
| `model_reviewed_origin_quote` basis | 10 |

The origin-basis row overlaps the tier rows; it is not a fourth review tier.
A model-reviewed origin quote was checked against the page. It is not a human
attestation of producer origin. The remaining records carry weaker origin bases.

The census route uses a separate page-scoped model reviewer and offline
recomposition. It is distinct from the detailed tuple → independent verifier →
origin model chain available elsewhere in the code. The census model-reviewed path
can retain unresolved referential and field-provenance checks in exported metadata.
It therefore does not establish that every canonical non-origin gate passed.

All 105 saved paper runs requested `google/gemini-3.5-flash-lite` for extraction;
the separate reviewer requested the same model identifier. The tuple, independent
verifier, and origin-retrieval model stages were disabled in those paper runs.

`positive_only` remains the default canonical policy. The `tiered` policy makes
weaker evidence bases visible in every record and does not label them
`paper_produced`. Schema validity checks the output structure. It does not establish
that a tuple was interpreted correctly or that the current paper generated its score.

## Evidence that exists

- Saved run artifacts establish processed-paper, candidate, support, and export
  counts for the dated snapshot.
- Offline deterministic tests exercise contracts for source freezing, evidence
  binding, model-stage orchestration, conflicts, composition, review, and integrity.
- A synthetic offline demo exercises the sealed reviewed-export workflow and yields
  schema-valid EEE with a fixed synthetic attestation.
- A 150-item packet exists for human assessment of sampled candidate/model-review
  outcomes. Its labels are still unfilled in the saved state.

Test success and synthetic execution are implementation evidence. The model-review
outputs are model assessments. Neither supplies missing human quality labels.

## Claims that remain unavailable

No current result establishes:

- Precision of all extracted candidates or all tiered EEE observations.
- Complete-tuple correctness, including system, dataset, metric, value, and scope.
- The error rate of the current `model_reviewed` tier.
- Producer-origin accuracy or an automatic promotion rule suitable for canonical EEE.
- Whole-paper recall, including results outside selected pages and missed candidates.
- Independent current-version performance on unseen papers.
- Human inter-annotator agreement or an adjudicated reference standard for the census.

Earlier pilot measurements used different runs and limited target sets. They must
not be carried forward as current census accuracy or as whole-paper recall.

## Immediate measurement

Use the [annotation guide](annotation-guide.md) to review the prepared 150 items
against the source. Preserve the frozen candidate tuple and record `correct`,
`incorrect`, or `cannot_tell`, with concise reasons where needed.

Score accepted and non-accepted model-decision strata with their declared
denominators. An intentionally stratified sample is not automatically representative
of the whole corpus. Report unresolved labels explicitly. Inspect the actual errors
before interpreting a single overall rate.

These labels evaluate sampled candidates and reviewer behavior. They do not change
EEE records, establish whole-paper recall, or supply all evidence required by the
separate sealed human-reviewed export protocol.

The next independent evaluation should define the target population, sampling,
annotation instructions, error categories, and treatment of disagreement before
reserved papers are opened. After the current annotation results are understood,
choose whether to prioritize extraction errors, topic relevance, identity resolution,
producer origin, or further corpus processing.
