# Architecture and invariants

## Pipeline

1. **Freeze sources.** Download or bind exact paper/supplement/repository bytes. Store final URL, UTC retrieval time, SHA-256, media type, source role, access state, and license disposition. Git sources require a commit.
2. **Parse and segment structure.** The MVP uses Poppler layout text split by page, then generic result signals to create bounded, hash-addressed blocks. Blocks retain page/line identity, horizontal table spacing, overlap, source-column bounds for separable side-by-side panels, and separate leading/header and trailing/caption context. A future Docling adapter can add cell bounding boxes behind the same fragment interface.
3. **Propose candidates.** A source-scoped LLM call sees one bounded result block and emits a strict candidate schema. An opt-in second stage enumerates dense table rows already exposed by those blocks in batches of at most four rows, 24 value tokens, and 4,000 characters. Every planned row receives `result`, `not_result`, or `uncertain`; invalid or missing rows get one bounded unresolved-only split level, while over-limit rows are retained as unbatchable review items and never sent. Both stages propose rather than decide.
4. **Verify support.** Deterministic logic requires the evidence quote and raw value to occur on the declared source page. Multiple matches are recorded.
5. **Resolve references and physical cells.** System role, dataset/scope, metric family/direction/scale, and value are checked independently. A small exact alias registry fills only unambiguous metric semantics. Table proposals that bind uniquely to the same source/page/table/row/value coordinates merge even when their model-written column descriptions differ. Repeated equal values that cannot be assigned to one cell abstain from structural merging; incompatible alternatives stay preserved and are forced to review.
6. **Resolve producer origin and apply export policy.** Producer origin is a checked
   state separate from model-reported `claim_type`. Only a positively established
   `paper_produced` result may enter canonical EEE. Externally sourced, unresolved,
   no-signal, non-primary, unsupported, low-confidence, role-unsafe, scope-unsafe,
   or evidence-sharing semantically conflicting items stay evidence-bearing in the
   candidate/review layer. `no_signal` is an abstention, never positive origin evidence.
   Result-rich blocks with zero proposals and papers with zero EEE receive explicit
   review states rather than silently appearing complete.
7. **Compose canonical EEE.** Paper-produced eligible observations are grouped by
   evaluated system and projected deterministically. EEE 0.2.2 cannot type the complete evidence and
   role model, so full quotes and typed role detail remain in the sidecar. Each
   numeric EEE result nevertheless retains a quote-free flattened anchor in
   `score_details.details`: paper/source ID and hash, page, structure kind,
   optional label/row/column, and quote hash.
8. **Validate and report.** JSON Schema validation includes an extra equality check between record `schema_version` and schema metadata. HTML displays all accepted and rejected candidates.
9. **Review and publish.** A deterministic risk-stratified sample supports a
   local, evidence-bearing analyst review with paper coverage, including
   explicit absence items for zero-candidate or zero-EEE papers. Only aggregate
   decisions and allowlisted derived artifacts can enter the public snapshot.
   Every paper receives a machine-readable and self-contained HTML **Extraction
   Review Card, not an Evaluation Card**, including papers with no exportable
   EEE. A corpus index, development-versus-holdout comparison, copied
   schema-revalidated EEE files, and `SHA256SUMS` bind the public bundle.

## Failure recovery

Extractor progress is checkpointed per result block under the private run
directory. The checkpoint contract binds the frozen source manifest, PDF parser
and version, segmentation settings, block IDs and text hashes, extractor
request contract, and code state. A rerun revalidates and reuses only successful
entries that match the complete contract; failed blocks are retried. Known
paper and corpus outputs are cleared before reconstruction so stale scores or
EEE files cannot be mixed into the new run.

A block failure produces a safe error code and does not discard successful
siblings. The paper is marked `partial_failure` until every selected block has a
successful or resumed result. Provider response bodies and exception text are
not persisted as recovery data.

The sealed holdout is extracted under the exact development-frozen semantic
contract. Its first completed run tree is checksummed before a human opens the
reference metrics. Only technically failed blocks may be retried under the same
contract; poor quality, zero candidates, zero EEE, and abstention are final
holdout outcomes rather than retry triggers.

## State dimensions

Reporting status, textual support, referential correctness, producer origin, and export status remain orthogonal. In particular, a quote can be textually supported but still have the wrong group, model role, or producer. `claim_type` does not prove origin, and `not_reported`, `unknown`, `not_applicable`, `unsupported`, `wrong_scope`, and `no_signal` are never collapsed.

## Evaluation-role example

In a synthetic moderation audit, Atlas Moderation API is an `evaluated_system`:
its agreement with human-labelled fixtures is measured. In a synthetic generation
study, the same API can instead be an `evaluation_instrument` that scores another
system's output. The current EEE schema has a dedicated LLM-judge shape but no
generic non-LLM evaluation-instrument role, so the sidecar preserves this
distinction and the review report exposes it for schema discussion.

## Trust boundary

LLM output is untrusted structured input. Pydantic rejects unknown fields; evidence must exist locally; metric semantics come from explicit registries; EEE composition and validation are deterministic. Provider responses cannot directly change source manifests, schemas, prompts, or code.

The optional second-model verifier is a separate, hash-bound boundary, not a
claim of independent human validation. It was disabled in the reported
ten-paper run because no separate verifier model had been calibrated for this
task. The reported manual risk check was a single-reviewer exploratory check
and is not independent human validation.

The public snapshot is field-allowlist-only and built atomically. It revalidates
EEE against the pinned schema and excludes source documents, evidence quotes,
candidate payloads, reviewer notes, provider traces, request IDs, cache paths,
and absolute local paths. `SHA256SUMS` binds the published tree.
