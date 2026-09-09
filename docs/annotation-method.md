# Annotation protocols and measurement boundaries

Human annotation is not required to share the code for collaboration. It is
required before making stronger accuracy, precision, or agreement claims.

For the immediate candidate-review sample, start with the
[annotation guide](annotation-guide.md). That workflow measures candidate tuple
correctness. The reviewed-export and mixed-development protocols below serve
different purposes and require different evidence. Their decisions are not
interchangeable. No independent human validation is claimed.

## Reviewed-export decisions

The reviewed-export protocol separates result and origin evidence. Each decision is
bound to the complete candidate payload and its exact
line in a sealed observation ledger. It records the result and origin judgments
separately:

- `result_evidence` retains the complete set of exact spans needed to support the
  reported tuple; at least one retained span must contain the raw value.
- A confirmed tuple adds one `field_attestations` entry for every populated tuple
  field. Each entry binds the unchanged field-value hash to one or more retained
  result-span IDs. Captions, headers, and wrapped names therefore remain available
  when they carry dataset, metric, setting, or system evidence.
- `origin_evidence` anchors who produced that result and may point to another page.
- Each anchor binds one frozen page, its exact UTF-8 substring, character and line
  spans, source and layout hashes, parser identity, original excerpt hash, and a
  stable exact-occurrence span ID. Evidence is not accepted through whitespace
  normalization, cross-page joins, or pooled extraction modes.
- The reviewer can confirm, reject, or leave the candidate tuple unresolved, but
  cannot edit it. Non-origin support, confidence, scope, conflict, cell, and schema
  gates are recomputed from the sealed run.
- Decisions declare their authority as single-expert, dual-consensus, or adjudicated.
  A pending decision is a locked abstention, not an implicit approval.

This contract is suitable for local reviewed export. It does not turn a
single-reviewer decision into independent validation, and the private packet must not
be committed because it contains exact source excerpts and reviewer identities.

## Mixed single-reviewer development package

The code includes a separate workflow for development review with one available
expert. It uses a configurable sampling frame and does not require two annotators.

The coordinator supplies a private selection plan covering all nine sampling intents:
paper-produced result candidates, external/copied result candidates, mixed or uncertain-origin
result candidates, and non-result headers, sample counts, setup values, parameters, thresholds,
and captions. The plan fixes configurable minimum counts and paper coverage. Package creation
fails unless every intent is represented. Per-item selection intent is projected away before
the reviewer sees `items.jsonl`.

The generated private bundle contains one blank `response.jsonl`, the item list, the protocol,
and hash-checked read-only copies of the frozen PDFs. Both the selection plan and package must
be outside the public repository. Creation refuses to overwrite an existing directory.

`PRIVATE_SELECTION.json` has the following strict shape. `source_run_id` must equal the
basename of `FROZEN_DEVELOPMENT_RUN`. Replace every excerpt with one exact, contiguous
source-native line from the declared page; the nine example items are placeholders, not
annotation data.

```json
{
  "schema_version": "mixed-development-annotation-selection/0.1",
  "source_scope": "development-and-error-analysis-only",
  "future_unseen_data_used": false,
  "source_run_id": "my-development-run",
  "minimum_result_candidates": 3,
  "minimum_result_papers": 1,
  "required_intents": [
    "result_paper_produced_candidate",
    "result_external_or_copied_candidate",
    "result_mixed_or_uncertain_origin_candidate",
    "non_result_header",
    "non_result_sample_count",
    "non_result_setup_value",
    "non_result_parameter",
    "non_result_threshold",
    "non_result_caption"
  ],
  "items": [
    {"paper_id": "example-paper-2026", "page": 1, "locator": "result-own", "selection_excerpt": "REPLACE WITH EXACT PAPER-PRODUCED RESULT LINE", "selection_intent": "result_paper_produced_candidate"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "result-external", "selection_excerpt": "REPLACE WITH EXACT EXTERNAL OR COPIED RESULT LINE", "selection_intent": "result_external_or_copied_candidate"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "result-uncertain", "selection_excerpt": "REPLACE WITH EXACT MIXED OR UNCERTAIN-ORIGIN RESULT LINE", "selection_intent": "result_mixed_or_uncertain_origin_candidate"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "header", "selection_excerpt": "REPLACE WITH EXACT HEADER LINE", "selection_intent": "non_result_header"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "sample-count", "selection_excerpt": "REPLACE WITH EXACT SAMPLE-COUNT LINE", "selection_intent": "non_result_sample_count"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "setup", "selection_excerpt": "REPLACE WITH EXACT SETUP-VALUE LINE", "selection_intent": "non_result_setup_value"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "parameter", "selection_excerpt": "REPLACE WITH EXACT PARAMETER LINE", "selection_intent": "non_result_parameter"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "threshold", "selection_excerpt": "REPLACE WITH EXACT THRESHOLD LINE", "selection_intent": "non_result_threshold"},
    {"paper_id": "example-paper-2026", "page": 1, "locator": "caption", "selection_excerpt": "REPLACE WITH EXACT CAPTION LINE", "selection_intent": "non_result_caption"}
  ]
}
```

`PRIVATE_SOURCE_PROJECT` is the project root against which each frozen source
manifest's `cache_relpath` resolves. It is normally the directory containing the
`data/sources/` cache created by the corpus run; it is not the run directory.

```bash
ere prepare-mixed-development-annotation PRIVATE_SELECTION.json \
  --run-root FROZEN_DEVELOPMENT_RUN \
  --source-project-root PRIVATE_SOURCE_PROJECT \
  --output PRIVATE_PACKAGE \
  --reviewer reviewer-primary
```

For every decision, `result_evidence` and `origin_evidence` are distinct anchors. Result
evidence must cite the selected item page; origin evidence may cite another page in the same
frozen PDF. Validation uses exact UTF-8 substring membership within one physical line of the
declared page. It does not normalize whitespace, pool extraction modes, join columns, or permit
cross-page excerpts. Positive `paper_produced` and `externally_sourced` decisions require an
origin anchor. A genuinely `uncertain` origin may leave that anchor null when the paper supplies
no page-local cue, rather than inventing irrelevant evidence.

```bash
ere validate-mixed-development-annotation-response PRIVATE_PACKAGE \
  --selection PRIVATE_SELECTION.json \
  --run-root FROZEN_DEVELOPMENT_RUN \
  --source-project-root PRIVATE_SOURCE_PROJECT \
  --package-manifest-sha256 SHA256_PRINTED_BY_PREPARE
```

The validator reports only completion count and a response hash on the command line. Source
text, decisions, notes, local paths, and per-item sampling intent remain private. A completed
single-reviewer package is development/error-analysis evidence only: it does not support
agreement, independent-validation, or unseen-generalization claims.

## Future evaluation packet

A future study should satisfy two design requirements:

1. Use result/origin evidence separation from the start. Freeze the protocol and
   preserve the original responses; changing the protocol requires a distinct study.
2. Include genuine negative cases. Sampling only numeric body
   rows cannot estimate rejection of headers, sample counts, method parameters,
   or other non-result rows. A future packet should stratify result rows and
   several kinds of non-result rows before annotation begins.

The sample and scoring plan should be fixed before annotation and should report at
least result-bearing classification, complete-tuple correctness, producer origin,
and final reviewed-export eligibility as separate outcomes. A second reviewer and
adjudication are needed only when making agreement or independently validated
performance claims; they are not a prerequisite for improving the pipeline now.
