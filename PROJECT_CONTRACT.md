# Project contract

## Objective

Build a reproducible Proceedings -> EEE pipeline: extract evaluation results reported in scientific papers into evidence-bound candidates and deterministically compose eligible observations into EEE JSON. Experiment reruns and Evaluation Cards are out of scope.

## Scientific unit

One immutable Candidate Observation represents one reported value for one evaluated-system snapshot, one dataset and exact scope, one metric and scale, one setting, and one or more exact evidence anchors. An EEE record is a deterministic projection of eligible candidates, never the extraction object.

## Model-stage boundaries

The five model boundaries are `block_candidate_extraction`, `row_disposition`,
`tuple_resolution`, `independent_verification`, and `origin_retrieval`. Extractor and
row outputs are untrusted proposals that converge at deterministic evidence,
physical-cell, conflict, and deduplication checks. Candidate-level execution then has
one safe order: tuple gate -> verifier gate -> origin assessment. Configuration rejects
a verifier without a tuple model and origin retrieval without a verifier before source
or provider work.

The production tuple call uses `require_parameters=true`, with the candidate's prior
tuple fields withheld. Its proposal is locally evidence-checked and compared with the
immutable candidate; it is never copied onto that candidate. System version, dataset
version, and setting may remain unresolved only when the candidate did not claim them.
Every claimed scope dimension and every other required semantic must match exactly.
Tuple, verifier, and origin stages are demotion/review-only: none can mutate a tuple,
establish review authority, automatically promote `paper_produced`, or grant export.
The separate tiered census recomposition path does not change the canonical model
chain or write `paper_produced`. Its `model_reviewed` tier is weaker: it can admit
candidates despite failed referential or field-provenance checks. Its named review
tier, origin basis, and retained failure reason must not be read as canonical approval.

## Export policy

Export has two policies, and every composition run records which one it used.

**`positive_only`, the default, is canonical export.** Canonical means positively
established: a record composed under this policy asserts that the current paper produced
the number. Nothing about this policy has changed.

**`tiered` grants export without asserting paper-produced origin.** A candidate may
compose on a weaker basis, and the record must carry that basis: `review_tier` in
{`deterministic`, `model_reviewed`, `human_confirmed`} and `producer_origin_basis` in
{`human_confirmed`, `positive_structural`, `model_reviewed_origin_quote`,
`model_asserted_primary_no_external_cue`, `model_asserted_primary_unchecked`}, both in
`source_metadata.additional_details`, per result in `score_details.details`, and inside
the composition provenance hash. No record composed under `tiered` is ever labelled
`paper_produced`, and an externally sourced candidate never exports under any policy.
The `model_reviewed` path relies on a page-scoped reviewer and retains the earlier
deterministic failure reason. In particular, failed referential or field-provenance
checks need not block this tier. Its error rate is unmeasured. Candidate annotation
measures tuple correctness on its sampling frame; it does not automatically establish
the accuracy of the final grouped exports or producer-origin decisions.

Under `positive_only`, a candidate can enter canonical EEE only when it is positively
established as a result produced by the current paper, is a primary-result proposal, has an explicit
value/system/dataset/metric/scope, has a source quote containing the raw value, has
resolved metric direction and scale, passes the confidence threshold, and has no
unresolved semantic conflict. When the downstream model chain is enabled, the
unchanged candidate must also have an exact concordant tuple gate and verifier accept
before origin retrieval. `claim_type` is model self-report and is not evidence of
producer origin. Externally sourced, unresolved, and no-signal candidates remain
evidence-bearing in the candidate/review layer; `NO_SIGNAL` never means paper-produced.

Canonical composition always requires a typed, candidate-hash-bound provenance
authorization. Automatic production records the exact tuple sidecar/gate and exact
accepting verifier sidecar/gate; a missing or stale gate fails closed. Tuple-only
automatic processing is the review-only `tuple_gated_unverified` mode and cannot emit
canonical EEE. Provider-disabled/manual composition remains available only as the
visibly distinct `legacy_manual` mode. Locked human review is also distinct: it binds the
sealed source tuple and verifier outcomes, including failures or abstentions, but a
full human tuple-and-evidence confirmation supplies review authority. An origin-only
decision cannot bypass the non-origin review gates, and provider output never supplies
human authority or changes the candidate tuple.

## Reproducibility envelope

Every run records source hashes, parser/version, selected pages, requested provider and
model, settings, prompt and response hashes, attempts/retries, code state, EEE schema
version/hash, counts, and wall time. Returned model/provider, request ID, finish reason,
input/output/total token counts, and provider-reported cost are recorded when returned
and remain explicitly nullable when absent; latency is recorded for each completed
call. Aggregate usage is therefore a lower bound whenever provider telemetry is
missing.
Extractor blocks and row batches have exact resumable checkpoints; row records also
include the frozen plan, hard bounds, terminal disposition partition, and bounded
recovery. Tuple, verifier, and origin use candidate checkpoints. A tuple gate binds its
candidate plus tuple contract and entry; the verifier entry binds that gate; the
verifier gate binds the candidate, tuple gate, verifier contract/entry, and decision;
and each origin entry binds both upstream gate hashes. Only a completely matching
contract may resume. Provider secrets, raw response envelopes, and exception text are
excluded.

Response-validation and provider failures are terminal typed candidate attempts and
are checkpointed immediately, so an exact resume does not repeat a completed failed
call. Such failures make the paper/corpus run technically incomplete; semantic
rejects, review routes, and unsupported evidence remain non-technical review outcomes.

## Private and scoring boundary

Exact source excerpts, row text, provider proposals, and human decisions remain
private run material, including candidate ledgers, checkpoints, sidecars, and sealed
review artifacts. Providers receive only the source evidence required by their stage,
never human labels, references, reviewer identities, or evaluation scores. Evaluation
provider phases are label-blind and must be sealed before a separate offline scorer
can read private labels or references. Public results and publishable projections are
quote-free, hash-bound allowlists.

## Evaluation boundary

Development spot checks are stored separately and evaluated only after extraction;
they never appear in prompts. The code includes stage-specific evaluation harnesses,
but their existence and passing synthetic tests do not validate real-paper quality.
The stored census snapshot used block extraction, optional rows, and a separate
page-scoped review; it is not evidence for the complete optional model chain.
Matched row quality needs a balanced human-labelled frame. Generalization claims
require a frozen unseen sample and a predeclared scoring protocol. See
[the evaluation status](docs/evaluation-status.md) for the dated measurements.

## Non-goals

- Evaluation Card or Benchmark Card generation.
- Silent repair, guessed identities, or scope collapse.
- Figure digitization without raw data.
- Republishing papers, supplements, or raw model traces.
- Rerunning missing experiments or submitting data to an external EEE datastore.
