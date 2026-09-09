# Project guide

ERE asks whether evaluation results in papers can be converted into structured,
traceable data for Every Eval Ever (EEE). The current implementation can perform the
conversion and expose uncertainty. The open research question is how reliably its
candidates and exported records match the papers.

The latest public status is a quote-free summary of saved artifacts dated
2026-09-05. It is not a newly executed experiment. See
[evaluation status](evaluation-status.md) for counts and their limits.

## The scientific unit

The basic unit is a **candidate observation**: one numeric result together with its
proposed interpretation. A useful tuple answers these questions:

| Field | Question |
| --- | --- |
| Evaluated system | Which model, method, or configuration produced this score? |
| Dataset and scope | Which dataset, split, subset, language, or population was evaluated? |
| Metric | What was measured, in which unit and scale, and which direction is better? |
| Reported value | What exactly was printed? |
| Setting | Which additional conditions distinguish this evaluation? |
| Evidence | Which frozen source, page, and table/figure/prose location supports it? |
| Producer origin | Did this paper produce the result, or repeat a result from elsewhere? |

For example, a synthetic table might report `72.5` for `Example System` on the
`Example Dataset` test split using accuracy in percent. Extracting `72.5` is only a
small part of the task. Assigning that number to the wrong row, split, metric, or
system creates a wrong observation even when the number appears on the page.

The pipeline preserves raw names and values. Missing identities are not guessed.
Optional details remain nullable; unresolved facts needed by an export policy route
the item to review. Equal numeric values can describe different physical table
cells, so deduplication uses evidence and cell identity where available.

An EEE record groups eligible observations by evaluated system within a paper.
It is not the same unit as a candidate: the latest 317 tiered EEE records contain
1,192 observations from 56 papers.

## The stages

| Stage | What it does | Why it is separate |
| --- | --- | --- |
| Freeze sources | Store exact source bytes and their manifest/hash | Later checks must refer to the same input |
| Recover layout fragments | Preserve page boundaries and table-like text structure | A value requires a physical paper location |
| Propose candidates | Ask a model for typed, nullable result tuples | Discovery can be useful without being trusted |
| Resolve roles and scope | Check proposed systems, datasets, metrics, and settings | Names and context determine what the value means |
| Check evidence and references | Check support, field bindings, cell conflicts, and optional independent readings | Finding a number is insufficient to establish a correct tuple |
| Compose EEE | Apply the declared export policy in deterministic code | Models cannot grant themselves final export authority |
| Validate the schema | Check records against pinned EEE 0.2.2 | Output structure must be reproducible |
| Review | Preserve withheld items and gather separate human decisions | Uncertainty and failures must remain inspectable |

The current parser uses Poppler text with layout preservation from selected PDF
pages. It retains the surrounding table/header/caption context available in that
layout. It does not digitize curves or bars in figure images. Supplements can be
frozen and repository metadata recorded, but automated extraction adapters for
those source types are not the current primary-PDF path.

## What models and code each do

Models can propose candidate tuples, classify dense rows, independently read result
context, assess candidates, and propose separate origin evidence. Those outputs are
bounded by schemas and remain proposals or review assessments.

Deterministic code freezes inputs, selects and hashes work units, checks evidence
against the page, validates types and known metric semantics, handles duplicate
cells and conflicts, applies gates and policies, composes EEE, and validates the
result. A check can establish that a quote is on the page without establishing that
the model interpreted its meaning correctly.

The detailed tuple gate hides the extractor's tuple from a second reading. Code
compares that reading with the original candidate and does not rewrite the candidate
from it. Downstream verifier and origin artifacts bind the exact preceding gates.
A failed or missing gate cannot be reused as a pass for a different candidate.

Resumable checkpoints and budget limits make larger experiments operationally
possible. They are engineering properties, not evidence of extraction quality.

## Value evidence and producer origin

**Result support** asks whether the proposed value and its interpretation are
supported by the stated location. **Producer origin** asks who generated the
experiment represented by that value.

A paper may print a baseline score copied from an earlier publication. Its value
can be quoted exactly and its tuple can be correctly extracted, while the current
paper is still not its producer. Conversely, a paper may report a new experiment
whose producer evidence is in methods prose on another page.

A model's `primary_result` label is self-report. The absence of an external citation
is not positive origin evidence. Origin states therefore distinguish
`paper_produced`, `externally_sourced`, `unresolved`, and `no_signal`.
The detailed model origin stage cannot automatically promote a positive proposal
to canonical paper-produced authority. External evidence can demote a candidate.

## Execution paths

The repository contains several related paths. The existence of a stage in the code
does not mean that every saved experiment executed it.

| Path | Purpose | Main limitation |
| --- | --- | --- |
| Safeguarded corpus route | Extraction, optional rows, tuple concordance, independent verification, and origin retrieval when configured | The complete current route lacks independent end-to-end quality validation |
| Census extraction and page review | Process many papers resumably, then ask a page-scoped reviewer about frozen candidates | This is not the complete five-stage model chain |
| Offline census recomposition | Reapply deterministic processing and an explicit origin export policy to saved work | `tiered` records may retain unresolved checks and weaker origin bases |
| Sealed human-reviewed export | Bind explicit human tuple/evidence/origin decisions to an immutable source run | It needs actual human attestations; a label alone is not a full export authorization |
| Offline synthetic demo | Exercise source freezing, review, composition, and integrity locally | Synthetic success does not measure performance on papers |

The saved census uses extraction plus the separate census review/recomposition
workflow. In particular, a `model_reviewed` census export must not be described as
passing the detailed tuple → verifier → origin chain.

The saved 105 paper runs requested `google/gemini-3.5-flash-lite` for extraction.
Their tuple, independent-verifier, and origin-retrieval model stages were disabled.
The separate census reviewer requested the same model identifier. It is another
assessment of the candidate, not an independent human or a different model family.
These identifiers describe the historical run, not a recommendation of a current
provider model.

## Export policy, review tier, and origin basis

These are three different metadata dimensions.

**Export policy** chooses the acceptable producer-origin basis. `positive_only` is
the default canonical policy and requires positive origin establishment. `tiered`
permits named weaker bases. Tiered output does not assert `paper_produced`, and known
externally sourced candidates remain ineligible under either policy.

**Review tier** describes how the candidate was checked:

| Tier | Meaning |
| --- | --- |
| `deterministic` | Local code checks were used; semantic correctness is not established by the name |
| `model_reviewed` | A separate page-scoped model review supported export after local checks |
| `human_confirmed` | A human-confirmed authorization supports the result |

**Producer-origin basis** records the evidence actually available. It may be human
confirmation, positive structural evidence, a locally verified model-proposed origin
quote, or only a model primary-result assertion with no external cue or without a
completed cue check. The basis should be read alongside the tier, not inferred from
it. Multi-observation record summaries use the weakest included tier/basis; individual
observations retain their own metadata.

The census model-reviewed recomposition path can admit a candidate with unresolved
referential or field-provenance checks and carry those states in the record. This
is a material relaxation relative to the canonical route. Schema validation and the
review tier must not be presented as a replacement for those checks.

## How development reached the current state

Development began with a small paper pilot to test source freezing, candidate
extraction, evidence retention, and EEE composition. Selected targets and inspected
examples helped reveal failure modes, including duplicates, ambiguous scope, and
the distinction between a printed score and a paper-produced result.

Subsequent work added explicit model stage boundaries, candidate-bound evidence,
resumable checkpoints, bounded provider spending, and a sealed human-review/export
workflow. Offline synthetic tests exercise these contracts and their failure cases.

The later census expanded the working corpus to hate speech and toxicity research.
It added concurrent per-paper processing, aggregate reporting, a local atlas, a
separate page reviewer, and tiered recomposition. This made a larger set of results
inspectable while exposing how little producer origin could be established under
the strict policy. Tiered exports are therefore useful research artifacts with
explicit limitations, not evidence that those limitations have been solved.

Older pilot counts and selected-target recall figures belong to their original
experiments. They are not measurements of the present census or current public
source snapshot. The public overview omits those historical rates to keep the
current claim boundary clear.

## What the current evidence establishes

The saved census establishes that the workflow processed 105 papers, retained 4,778
candidates, found textual support for 4,202, and composed 317 schema-valid tiered
records. Those facts establish extraction volume and software behavior.

They do not establish candidate precision, complete-tuple correctness, producer-origin
accuracy, whole-paper recall, or performance on unseen papers. A supported quote is
not a semantic gold label. A schema-valid record can still contain a wrongly
interpreted result. Multiple models can agree on the same error.

The 150-item packet is a sample of existing candidates and model decisions. Human
labels can assess that sampled review behavior. They cannot measure the number of
missing results, and they do not automatically become locked authorizations for the
separate human-reviewed export path. Follow the [annotation guide](annotation-guide.md)
for the precise procedure and reporting denominator.

## What is built and what remains

Implemented capabilities include the staged candidate model, frozen source manifests,
text-layout extraction, evidence and conflict checks, pinned-schema composition,
local review reports, census processing, offline recomposition, and a sealed
human-reviewed export workflow. The repository provides synthetic examples that can
be executed without model credentials.

Open work includes human measurement of current candidates, reliable topic filtering
inside selected papers, dataset and system identity normalization, better origin
resolution, and independent evaluation on unseen papers. The local atlas can expose
the extracted material, but its contents are not yet a verified benchmark catalogue.
Evaluation Cards, figure digitization, and automatic harvesting of associated
repositories are outside the implemented primary workflow.

For a joint project review, read the status table, run the synthetic demo, and walk
through a few privately held annotation examples. Then decide which errors matter
most, what review tier is acceptable for the intended use, how to define the target
result population, and what independent test should follow. The saved local plan
records 3,080 unstarted papers and a 60-paper reserve; preserve the reserve until its
evaluation protocol is defined. Scaling the remaining corpus should follow an
explicit quality and cost decision.
