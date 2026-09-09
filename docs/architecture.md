# Architecture and invariants

## Pipeline

This section describes the available architecture, including optional model stages.
The stored census run used block extraction, optional rows, and a separate page-scoped
review. It did not validate the complete tuple/verifier/origin chain. See
[the project guide](project-guide.md) for the routes and
[the evaluation status](evaluation-status.md) for measured outcomes.

1. **Freeze sources.** Download and hash exact paper or supplement bytes. A primary
   paper uses exactly one `pdf_url` or `pdf_path`; local paths resolve against the
   corpus YAML before entering the same content-addressed cache and manifest contract.
   File sources retain their final location, UTC retrieval time, SHA-256, byte size,
   media type, source role, access state, and license disposition. A repository source
   retains its URL and a declared full commit only after `git ls-remote` advertises that
   commit. The workflow does not clone or hash the repository tree.
2. **Parse and segment structure.** The MVP uses Poppler layout text split by page, then generic result signals to create bounded, hash-addressed blocks. Blocks retain page/line identity, horizontal table spacing, overlap, source-column bounds for separable side-by-side panels, and separate leading/header and trailing/caption context. A future Docling adapter can add cell bounding boxes behind the same fragment interface.
3. **Extract block candidates (`block_candidate_extraction`).** A source-scoped LLM
   call sees one bounded result block and emits a strict candidate proposal. It does
   not decide support, tuple correctness, producer origin, or export eligibility.
4. **Optionally classify planned rows (`row_disposition`).** Dense table rows already
   exposed by selected blocks are frozen before calls in batches of at most four rows,
   24 value tokens, and 4,000 characters. Each row terminates as `result`,
   `not_result`, `uncertain`, `unresolved`, or `unsupported`; only one bounded
   unresolved-only split level is allowed. This stage may add proposals but has no
   export authority.
5. **Apply deterministic support and physical-cell checks.** Evidence and raw values
   must occur on the declared frozen page. System role, dataset/scope, metric
   direction/scale, value, and physical table coordinates are checked independently.
   Exact cell duplicates may merge; repeated equal values without a unique cell and
   incompatible alternatives remain review items.
6. **Require tuple concordance (`tuple_resolution`).** After deterministic validation
   and deduplication, the provider sees exact page-local result evidence with the
   candidate's prior tuple fields withheld. The locally verified proposal is compared
   with, but never copied onto, the immutable candidate. Any unsupported, failed, or
   non-concordant outcome can only demote the candidate to review.
7. **Run independent verification (`independent_verification`).** Only a tuple-passed,
   still-eligible primary-result candidate reaches the verifier with its frozen result
   block. A non-accept decision routes to review and cannot be repaired by a later
   model stage; a distinct authorized human re-attestation may still resolve it.
8. **Resolve producer origin (`origin_retrieval`).** Only a tuple-passed and
   verifier-accepted candidate reaches origin retrieval. Producer origin is separate
   from model-reported `claim_type`. Verified external evidence can demote; positive
   evidence and no-signal outcomes remain review-only. This stage has no automatic
   `paper_produced` or export route.
9. **Compose canonical EEE.** Paper-produced eligible observations are grouped by
   evaluated system and projected deterministically. Every emitted candidate needs a
   typed hash-bound authorization. Automatic production binds both a passing tuple and
   an accepting verifier gate. A tuple-only run is explicitly
   `tuple_gated_unverified`, remains review-only, and cannot emit canonical EEE;
   provider-disabled/manual output is explicitly marked `legacy_manual`. A composition
   run also records which producer-origin export policy it used. `positive_only` is the
   default and means canonical. `tiered`, reached only through `ere recompose-census`,
   exports candidates on a named weaker basis without ever writing `paper_produced`;
   the basis and the review tier travel in the record and inside the provenance hash.
   The separate `model_reviewed` census tier can admit candidates despite failed
   referential or field-provenance checks, retaining the earlier failure reason.
   It is not equivalent to passing the canonical chain. EEE 0.2.2
   cannot type the complete evidence and role model, so full
   quotes and typed role detail remain in the sidecar. Each
   numeric EEE result nevertheless retains a quote-free flattened anchor in
   `score_details.details`: paper/source ID and hash, page, structure kind,
   optional label/row/column, and quote hash.
10. **Validate, review, and publish.** JSON Schema validation includes an equality
   check between record `schema_version` and schema metadata. Extraction Review Cards
   expose candidates and
   abstentions, including papers with no candidates or no EEE. A separate
   reviewed-export bridge can bind private decisions to an exact sealed run,
   validate distinct result and producer-origin anchors, and deterministically
   recompose only candidates whose deterministic safety gates still pass. Full human
   tuple-and-evidence re-attestation may resolve an untrusted model tuple/verifier
   outcome in a distinct provenance mode. Only quote-free, allowlisted derived
   artifacts can be considered for publication. The cards are
   **Extraction Review Cards, not Evaluation Cards**.

## Five model boundaries and safe order

The five independently selectable model stages are
`block_candidate_extraction`, `row_disposition`, `tuple_resolution`,
`independent_verification`, and `origin_retrieval`. Block and row proposals converge
at deterministic validation and deduplication. The candidate-level safety chain is
then strictly:

```text
immutable candidate -> tuple gate -> verifier gate -> origin assessment -> composition
```

Configuration fails before source or provider work if a verifier is enabled without
a tuple model, or origin retrieval is enabled without a verifier. Passing one boundary
does not grant authority held by a later one, and none of the three downstream model
stages may mutate candidate tuple fields or automatically promote a candidate.

Provider/request/response failures are typed terminal attempts, checkpointed
immediately, and make the technical run incomplete. Semantic reject/review decisions
and locally unsupported evidence remain review outcomes rather than technical errors.

### Demotion-only tuple gate

Production tuple calls use the strict route contract with
`require_parameters=true`. A candidate must have exact page evidence and an exact
physical numeric-cell binding. The provider proposes system, dataset, metric,
direction, scale, value, uncertainty, unit, setting, and scope from that evidence;
local code verifies every field and compares it to the frozen candidate.

System version, dataset version, and setting may remain explicitly unresolved only
when the candidate made no corresponding claim. Scope is not relaxed: every claimed
split, subset, group, language, sample count, aggregation, or raw scope must match
exactly, and an absent scope claim cannot be invented. Every export-required semantic
and every optional semantic the candidate did claim must be verified and concordant.
The proposal remains a private sidecar; it never replaces candidate data. Passing only
permits the unchanged candidate to reach the verifier. Failure, rejection, unsupported
physical evidence, or a mismatch routes to review and can never promote.

### Exact checkpoint and gate chain

Each stage has its own resumable unit: result block, planned row batch, tuple candidate,
verifier candidate, or origin candidate. Its checkpoint contract and entry together
bind the frozen source manifest and layout, stage settings, prompt/schema/request
contract, code state, and exact candidate or batch input. Reuse occurs only after
complete revalidation; stale, malformed, or differently configured entries are
discarded.

The tuple gate hash binds the immutable candidate, tuple checkpoint contract and entry,
terminal status, semantic match, and pass decision. The verifier checkpoint entry binds
that exact tuple gate; the verifier gate binds the candidate, tuple gate, verifier
contract and entry, and verifier decision. The origin checkpoint contract binds the
tuple and verifier contracts, while each origin entry binds the exact tuple and
verifier gate hashes. Consequently, a downstream checkpoint cannot be replayed against
another candidate or an earlier upstream decision.

Exact excerpts and provider inputs remain private run material, including candidate
ledgers, checkpoints, and sidecars. Provider phases never receive human labels,
references, reviewer identities, or evaluation scores. Staged evaluations seal their
label-blind provider checkpoints before offline labels or references may be read.
Local user-facing reports are research artifacts, not publication outputs. Publishable
run projections are quote-free, hash-bound allowlists; raw provider envelopes and
source text are excluded.

Completed-call telemetry always records latency and attempt count. Provider-returned
model/provider, request ID, finish reason, token counts, and cost are nullable: they
are retained when returned and stay explicit `null` when unavailable, so aggregates
can distinguish zero usage from a metadata lower bound.

## Reviewed-export trust bridge

The implemented bridge connects a frozen model run to human-reviewed EEE without
letting review mutate or silently repair extraction:

```text
run -> immutable run seal -> private review packet -> locked decisions
    -> disjoint quote-free derived tree -> standalone verification
```

`prepare-export-review` first verifies `RUN-SEAL.json` and its complete file
inventory. It then creates a private packet containing read-only copies of each
paper's run manifest, observation ledger, source manifest, layout, and result blocks.
Each item binds the full candidate payload, ledger line, and structural identity by
hash. The packet also binds the parser, schema, source run, and any code-state hashes
recorded by that run. The editable decision template starts from the exact candidate
tuple.

Result evidence and producer-origin evidence are independent strict anchors. Each
anchor names one frozen source and layout page, exact UTF-8 excerpt, character and
line span, parser identity, original excerpt hash, stable exact-occurrence span ID,
and relevant hashes. A confirmed tuple must bind every populated field's unchanged
value hash to retained result spans; at least one result span must contain the raw
value. This keeps load-bearing captions, headers, and wrapped names in the evidence
bundle. The excerpts may come from different pages; whitespace normalization,
cross-page joins, and pooled extraction modes do not satisfy the review protocol. A
decisive `paper_produced` origin also has to name the evaluated system. Decisions
declare single-expert, dual-consensus, or adjudicated authority. Pending decisions
remain valid abstentions.

`validate-export-review` checks the complete private tree and locks the exact
decision bytes. Review can confirm, reject, or leave the existing tuple unresolved,
and can resolve producer origin. It cannot change the tuple, manufacture result
support, override low confidence or an unsafe role/scope, clear a conflict, merge a
duplicate cell, or bypass schema validation.

`compose-reviewed-eee` reopens the original seal, verifies every source artifact,
and recomputes deterministic non-origin safety checks before applying locked human
decisions. Its `tuple_audited_human_reviewed` provenance binds the source tuple and
verifier outcomes—even review, reject, failure, or not-run outcomes—without treating
a provider result as human authority. Exact field-to-span attestation may cure only a
field-provenance-only blocker; it cannot cure text, reference, confidence, semantic,
physical-cell, duplicate, or schema failures. Full human tuple-and-evidence
confirmation may resolve untrusted tuple/verifier outcomes; origin-only approval
cannot. Legacy source runs use the distinct `legacy_human_reviewed` mode.
Pending, external, unresolved, rejected, and unsafe items receive one typed outcome.
The composer writes atomically to a separate tree and checks the source seal again
afterward. That derived tree contains only EEE records, quote-free dual-evidence
provenance, per-item outcomes, manifest, checksums, and verification receipt.
`verify-run` can validate the derived tree without access to source excerpts or
reviewer identities.

This bridge is covered by synthetic local end-to-end and tamper tests. It establishes
artifact integrity and a usable reviewed workflow; it does not establish extraction
accuracy or solve automatic origin retrieval.

## Failure recovery

Extractor progress is checkpointed per result block and row progress per exact base
batch under the private run directory. Tuple, verifier, and origin use candidate-level
checkpoints connected by the gate hashes above. A rerun revalidates each complete
contract and reuses only exact terminal work. Failed block or row work follows its
bounded recovery policy; a downstream semantic failure is a review outcome, not a
retry or permission to bypass the chain. Known paper and corpus outputs are cleared
before reconstruction so stale scores or EEE files cannot be mixed into a new run.

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

The tuple resolver, second-model verifier, and origin resolver are separate,
hash-bound model boundaries, not independent human validation. Repair05 contract
smokes now cover all five stages. Matched development evidence supports measured
route selection for extraction, independent verification, and origin retrieval, while
row and tuple selection remain technical-only. The current ten-paper execution is a
page-scoped diagnostic run with all candidates still pending human review.

The public snapshot is field-allowlist-only and built atomically. It revalidates
EEE against the pinned schema and excludes source documents, evidence quotes,
candidate payloads, reviewer notes, provider traces, request IDs, cache paths,
and absolute local paths. The pre-human preview also verifies its sealed source,
prospective route, budget ledger, and corpus scope. After review, the reviewed bundle
copies only manifest-listed EEE from the contextually verified derived tree and
regenerates its canonical JSON and static HTML result tables from those EEE bytes.
Standalone verification rechecks the source binding, schema, inventories, rendering,
privacy boundary, and checksums. `SHA256SUMS` binds each published tree.
