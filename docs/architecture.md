# Architecture and invariants

## Pipeline

This section describes the available architecture, including optional model stages.
The stored census run used block extraction, optional rows, and a separate page-scoped
review. It did not validate the complete tuple/verifier/origin chain. See
[the project guide](project-guide.md) for the routes and
[the results page](results.md) for measured outcomes.

1. **Freeze sources.** Download and hash exact paper or supplement bytes. A primary
   paper uses exactly one `pdf_url` or `pdf_path`; local paths resolve against the
   corpus YAML before entering the same content-addressed cache and manifest contract.
   File sources retain their final location, UTC retrieval time, SHA-256, byte size,
   media type, source role, access state, and license disposition. A repository source
   retains its URL and a declared full commit only after `git ls-remote` advertises that
   commit. The workflow does not clone or hash the repository tree.
2. **Parse and segment structure.** The MVP uses Poppler layout text split by page, then generic result signals to create bounded, hash-addressed blocks. Blocks retain page/line identity, horizontal table spacing, overlap, source-column bounds for separable side-by-side panels, and separate leading/header and trailing/caption context. The deterministic region index attaches captions with one rule: a table takes the nearest unclaimed caption above or below it within `caption_distance_lines` (default 12) page lines, nearest pair first, each caption and table claimed once, never across a numbered or all-caps heading or another table. A caption changes column panel only when its text runs through the gutter or one side spans the page, and a figure caption attaches only from below. The block segmenter applies the same claim rule to add a caption that lies beyond the leading or trailing context window to the block's context, bounded by the caption distance on top of the window; it does not remove the caption from any window it already fell into, so a caption printed between two tables can appear in both blocks' contexts and a distant trailing caption still forms its own caption-only block. These caption and heading semantics changed under the unchanged `page-region-index/0.1` and `result-block/0.3` schema strings, whose field shapes did not change. A lone all-caps token such as `F1` or `ACM` is not a heading; an all-caps line needs two words and eight characters, while a single capitalized word such as `References` or `Abstract` still is. Bibliography headings are recognised in their English, French, Italian and German forms. A future Docling adapter can add cell bounding boxes behind the same fragment interface.
   Every page is read unless a corpus entry sets `max_result_pages` or `include_pages`,
   and every block on a selected page is kept unless `--max-blocks-per-page` caps it.
   Blocks whose lines lie inside the bibliography, as tracked by the region index across
   page breaks or by a page-local `References` heading, never reach a model; a block is
   dropped only when its own lines read like bibliography entries and none reports a
   value, so appendix results after the references and a result column beside a
   bibliography column survive. A table
   longer than `max_data_rows` (14), or whose rows would carry more than
   `max_numeric_cells` (24) numeric tokens into one call, is split into chunks that each
   repeat the caption and header context and record `chunk_index`, `chunk_count`, and
   `header_repeated`. The cell bound is what keeps the expected output per call small: on
   a wide table the extractor model returns one candidate and stops once a block holds
   roughly fifty numeric cells, so chunks are sized by cells, not rows. The same model
   occasionally returns almost nothing for a dense block of ordinary size, so a block
   with at least three data rows and nine decimal cells that yields at most one
   candidate is called once more and the attempt with more candidates is kept; both
   calls are recorded, the checkpoint names the kept one, and `run.json` counts
   `sparse_retries` and `sparse_retries_kept`.
   `run.json` and `preflight.json` record `page_count`, `pages_selected`,
   `blocks_dropped_in_references`, `tables_chunked`, and the number of extractor calls
   the provider cut off at `max_tokens`, so nothing is cut silently.
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
   must occur on the declared frozen page. A single-line quote must occur verbatim
   inside one layout line. A quote that itself spans lines is supported only when
   every newline-separated segment equals one whole page line, in order and on
   consecutive lines, after layout-whitespace normalization; fragments of
   neighbouring lines cannot be stitched together. The candidate records
   `support_match_mode` (`single_line` or `consecutive_lines`), and only
   single-line support can pass the deterministic export gate; consecutive-line
   support routes the candidate to review. `support_match_mode` is review state
   outside the candidate's stable id and never reaches records, but it is part of
   the full candidate dump that the tuple candidate binding and the verifier
   candidate hash cover, and the extractor checkpoint contract embeds
   `caption_distance_lines`, so gates and checkpoints written before these fields
   existed do not resume against the current code; every checkpoint contract also
   binds the code state, which already prevented that. System role, dataset/scope, metric
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
   It is not equivalent to passing the canonical chain. Within that tier the cue
   wins: a reviewer accept earns the `model_reviewed_origin_quote` basis only where
   the deterministic verdict recorded no attribution cue (a row whose label names the
   paper itself, such as "Ours", is the one exception: the table cues recorded on it
   describe the table and do not hold the row back), a reviewer that withheld
   origin grants nothing, and composition fails closed, naming the candidate, when an
   authorization disagrees with the verdict. EEE 0.3.0, the pinned default
   authority (0.2.2 stays vendored and selectable for older artifacts),
   cannot type the complete evidence and role model, so full
   quotes and typed role detail remain in the sidecar. Each
   numeric EEE result nevertheless retains a quote-free flattened anchor in
   `score_details.details`: paper/source ID and hash, page, structure kind,
   optional label/row/column, quote hash, and the ids of the attribution cues that
   fired (`attribution_cues`), never their matched text. The same map states how
   the metric identity was established (`metric_identity_source`: local registry,
   concordant short alias, vendored EEE registry snapshot, extractor only) and how
   its bounds were (`metric_bounds`: registry, inferred percent, or unresolved, in
   which case the record carries explicit null bounds rather than an invented
   scale). The snapshot vendors reviewed registry entries and confirmed aliases
   only, and leaves out placeholder entities such as `score` or `rank` whose name
   says what form a number has rather than what it measures. A signed difference
   or a value outside its metric's declared bounds never exports as a score.
   Because a model-proposed scale is withheld under every schema authority,
   recomposing the saved census under 0.2.2 no longer reproduces the historical
   317-record projection: candidates whose only bounds were model-proposed are
   skipped there, while 0.3.0 composes them with null bounds. Reporting paths that
   re-validate a finished run (the public development summary and
   completed-corpus validation) accept either vendored version and use the schema
   the run recorded.
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

## Model boundaries and safe order

The five independently selectable candidate-chain model stages are
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

### Dataset inventory, the sixth boundary

`dataset_inventory` is annotation-only and runs beside the chain, post-hoc over a
finished run (`ere inventory-datasets`). Per paper it sends the manifest title, every
caption from the region index on every page, and the layout text of the sections
whose heading matches
`data|dataset|corpus|task|benchmark|experimental setup|experiments|evaluation`,
bounded by `--max-section-characters` (the bound cuts section text only; captions
and the searchable text of later pages are unaffected), and receives a strict list
of entries (`name_raw`, `aliases_in_paper`, `kind`, `languages`, `splits`,
`introduced_here`, `evidence` as page plus verbatim quote). Every quote is re-found,
whitespace-normalised only (a line-end hyphen is not relaxed; the gate names the
normalisation), on the cited page of the frozen layout or in one of that page's
regions; an entry without a verified anchor is dropped and counted. The entry's
printed name must occur in the layout, whitespace-collapsed and ignoring case, or the
entry is dropped and counted as unnamed; an alias the layout does not print is
removed and counted; a kind outside the vocabulary is malformed. The gate
`<paper>/dataset-inventory.json` is quote-free (entry ids hash name and kind; anchors
are page, quote hash and verified flag; the prompt, schema and model identifiers, the
input hashes, the cost and the dropped counts are recorded) and
`<paper>/private/dataset-inventory.jsonl` keeps the proposal with its quotes. A gate
with the current prompt and schema hashes is not redone, and the call is charged to
`<run_root>/private/dataset-inventory-budget-ledger.jsonl`.

Binding is deterministic and lives in `ere recompose-census`, after validation stamps
observation ids and before attribution, only for papers that have an inventory. The
rules apply in order and record their id: `exact` (normalised raw string equals an
entry's name or alias), `caption` (the anchored table's caption or header row names
exactly one entry, and the raw string names nothing beyond it: a split phrase, a
generic word, a table reference, a language, or words that all occur in the entry's
name or aliases), `unique` (a single dataset or subtask entry and a raw string that
is a split word, a language, a generic word or a `Table N` reference), else `none`;
ambiguity at any rule is `none`, and so is a caption match that would narrow a raw
string naming more than the entry (`none_caption_overrides_named_raw`). The anchored
table is read from the index that located the quote: a full-width table on a
two-column page is located through the single-panel fallback, whose region ids
collide with the panelled index's, so the location says which index produced it
rather than the id being trusted alone. A split is derived only when the candidate states
none, from the anchor's column header first and the attached caption second. The
result is the `dataset_binding` sidecar on the replayed candidate, outside
`stable_id()`, outside `field_value_sha256s()` and outside the review gate payload
hash, so an unchanged candidate keeps its observation id and every gate that names
it; `scope.dataset_raw` is never rewritten and the run's `observations.jsonl` is
never touched. The export composition provenance binds the full candidate dump,
sidecar included, so a bound record's `export_candidate_sha256` differs from the
unbound one by design, while an absent sidecar is left out of that hash so a
candidate that was never bound keeps the export hash it had before the sidecar
existed.

Composition uses the bound name for `source_data.dataset_name` and `evaluation_name`
when the class is `exact`, `caption` or `unique`, and always keeps `dataset_raw`,
`dataset_binding_class` and `dataset_binding_rule` in `source_data.additional_details`,
with a derived `split` and `split_source` only when the split detail was empty and the
entry's `language` only when the candidate had none.
`source_metadata.additional_details.dataset_inventory_sha256` names the inventory the
record was composed with, as a hash of the gate's content without its telemetry, so
reruns that verify the same entries name the same inventory. The public snapshot
allowlists accept each of these keys and reject a binding class or split source
outside the composer's vocabulary.

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
