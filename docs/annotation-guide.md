# Annotating the census candidate-review sample

This guide covers `review-label-packet` and `score-review-labels`. One item is one
extracted result candidate checked against its supplied source page. The human judges
the candidate's correctness, then the scorer compares those judgments with the model's
decisions. The guide does not contain real annotation items or completed human labels.

This is a diagnostic review of existing candidates. It does not find results the
extractor missed, prove producer origin, or approve candidates for canonical EEE.
The richer, separately sealed reviewed-export protocol is described in
[annotation-method.md](annotation-method.md).

The code can be shared now. Completing 150 labels is not a prerequisite for a
methodology discussion or code collaboration. Agree on the labeling protocol first;
finish the human judgments before reporting stronger performance claims.

## What the 150 items represent

The default packet contains up to 100 candidates that the model accepted and up to 50
that it rejected or abstained on. These are sampling strata, not known answers:

| Sampling stratum | Model decisions included | What the human labels |
| --- | --- | --- |
| Accepted | `accept`, `accept_tuple_only` | Whether the candidate is correct as stated |
| Not accepted | `reject`, `abstain` | Exactly the same question |

`accept_tuple_only` means that the model accepted the result tuple without establishing
positive producer origin. An abstention supplies no negative truth about the result.
The 50 not-accepted items therefore are not a gold set of 50 incorrect results or genuine
non-result rows. Do not force a desired label balance.

For a new packet, the coordinator can run the following offline command once:

```bash
uv run ere review-label-packet PRIVATE_RUN_ROOT \
  --size 100 --negatives 50 --seed 20260905 \
  --output FROZEN_PACKET.json
```

An existing packet should be retained rather than regenerated after inspecting labels.
Record its file hash, source run, selection scope, and sample seed. Keep real packets,
responses, frozen PDFs, and notes outside the public checkout. Do not commit them.

## Prepare the reviewer copy before starting

The original packet exposes `model_decision` and `model_proposed_decision` and places
accepted items before nonaccepted items. Reading it directly can bias the annotation.
Keep that original unchanged for scoring. Prepare a shuffled copy that withholds the
model decisions, and annotate only that copy. Prior exposure cannot be undone; record
it if the reviewer has already inspected the model outcomes.

The command below writes a blank view and refuses to overwrite an existing response.
It does not label any item. Run it from the installed project environment, replacing
the two private filenames with the frozen packet and a new response destination:

```bash
uv run ere blind-review-label-packet FROZEN_PACKET.json \
  --output REVIEWER_LABELS.json --seed review-label-shuffle-v1
```

The view retains a content hash binding it to the original scoring packet. It hides
the model decisions and stratum order, but still shows the extracted candidate and its
evidence excerpt. Read the page before relying on that excerpt. This is decision
blinding, not a claim that the reviewer is independent of the project.

## Work through one item

1. Read `page_text`. Locate the result row or sentence and the relevant caption,
   column headings, footnotes, and surrounding context. The extractor's quoted excerpt
   is a locator, not proof that the extraction is correct.
2. Compare every stated component in `candidate`: evaluated system, dataset, metric,
   unit or scale, reported value, and scope. Scope can include the split, subset,
   language, group, and aggregation. Make sure the system is being evaluated, rather
   than merely providing labels or measuring another system.
3. Write one of the three exact labels below and a short `label_note`. Do not edit the
   proposed tuple, IDs, page text, or other fields to make the record correct.
4. Save regularly. Work in batches, for example 15 items at a time, while retaining
   the shuffled order. Leave unfinished labels `null`. A human must fill the real
   judgments; do not ask an LLM to complete or balance them.

| Label | Use when | Example note |
| --- | --- | --- |
| `correct` | Every stated part of the tuple is supported by the supplied page, with no material ambiguity. | "Table 2, Cedar row, test macro-F1 column; the value and percent scale match." |
| `incorrect` | The page establishes a concrete error in at least one material component. | "The value belongs to Pine, not Cedar." |
| `cannot_tell` | The supplied page does not let you decide, including ambiguous layout or missing context. | "The dataset is named only on another page; this page does not establish it." |

A missing optional field is not permission to invent one. If it prevents you from
identifying the actual result, use `cannot_tell`. If the page contradicts a stated
field, use `incorrect`. Preserve the printed scale: if the page reports `81%`, do not
accept a raw `reported_value` of `0.81` merely because it is numerically equivalent.
The candidate must retain the printed value and distinguish percent from proportion.

Use `correct` for a correct candidate even if the model rejected it. Use `incorrect`
for an incorrect candidate even if the model accepted it. You are not judging whether
the model's action seemed reasonable. Producer origin is a different question and is
not established by these three tuple labels.

## Locate the frozen PDF page

Every item already includes its page text. If table geometry is unclear, locate
`PRIVATE_RUN_ROOT/PAPER_ID/source-manifest.json`, match the item's `paper_id`, and
resolve the PDF source's `cache_relpath` relative to the private source project root.
Confirm its SHA-256 against the manifest before using it. The corresponding observation
in `observations.jsonl` supplies `evidence[].source_id` if several sources exist.

The packet's `page` is the one-based physical PDF page index. It is not necessarily
the printed page number in the article footer. Open that physical page in a PDF viewer;
do not substitute a newly downloaded version. If source identity or page mapping is
unclear, flag the item for the coordinator.

Fix the evidence scope before annotation. The default question is about the supplied
page; viewing that same PDF page can resolve layout. If a necessary fact exists only
elsewhere in the paper, use `cannot_tell` and note the missing context. A whole-paper
study needs a separately specified protocol and page-bound evidence; do not quietly
change the question midway through this sample. Record the evidence mode used in the
annotation session notes.

## Score progress and the completed sample

Score the filled reviewer copy against the unchanged original packet:

```bash
uv run ere score-review-labels FROZEN_PACKET.json REVIEWER_LABELS.json \
  --confidence 0.95 --output PRIVATE_SCORE.json
```

This is offline and does not call a provider or change EEE outputs. Scoring checks for
duplicate or unknown IDs, invalid labels, ambiguous IDs across papers, changed source
content or candidate fields in a full reviewer copy, and an incorrect original-packet
hash. Legacy minimal label files remain supported, but do not have the hash binding;
the summary explicitly reports whether the binding was verified.

Missing or `null` labels are allowed for progress checks. They produce
`completion_status: partial` and an explicit `unlabelled` count. A completed 150-item
sample must have `packet_entries: 150`, `labelled: 150`, `unlabelled: 0`, and
`completion_status: complete`. `cannot_tell` counts as a completed judgment, but is
reported separately and excluded from both rate denominators. Completion therefore
does not mean that all 150 candidates were decidable.

The two reported rates mean:

- `model_accepted_and_human_says_correct`: the fraction of decidable sampled accepted
  candidates labelled `correct`.
- `model_did_not_accept_and_human_says_incorrect`: the fraction of decidable sampled
  nonaccepted candidates labelled `incorrect`. This includes abstentions and is not
  specificity or recall.

Report each numerator, denominator, interval, unlabelled count, and `cannot_tell`
count. The intervals are Clopper-Pearson binomial intervals. They do not model the
dependence of several results from the same paper. Do not average the two strata into
whole-corpus accuracy: their sample sizes were chosen deliberately, not in proportion
to the population. The sample also does not directly estimate the precision of final
EEE exports, since export gates select a different subset of candidates.

A single reviewer provides a useful development diagnostic. These labels alone are
not inter-annotator agreement, adjudicated accuracy, producer-origin accuracy, or
generalization to unseen papers. Do not use inspected items to claim a new holdout.

## Try the synthetic example

The following files contain only invented source text, IDs, candidates, and example
answers. The answers demonstrate the format; they are not human research data:

```bash
uv run ere score-review-labels examples/annotation/packet.json \
  examples/annotation/labels.json
```

Expected: five completed labels, one `cannot_tell`, accepted correctness `1/2`, and
nonaccepted incorrectness `1/2`. The rejected or abstained stratum includes a correct
candidate, demonstrating why its human label must still be `correct`.

To see the blank reviewer workflow, create a new private output filename:

```bash
uv run ere blind-review-label-packet examples/annotation/packet.json \
  --output PRIVATE_SYNTHETIC_LABELS.json --seed review-label-shuffle-v1
uv run ere score-review-labels examples/annotation/packet.json PRIVATE_SYNTHETIC_LABELS.json
```

The untouched view reports five unlabelled items and a partial completion status.

Before discussing results, retain the frozen packet hash, annotated response hash,
reviewer count, protocol and evidence mode, any prior exposure to model judgments,
completion counts, both conditional rates, and a short taxonomy of errors. Those
records make a discussion about the next evaluation design concrete without treating
this small diagnostic sample as a finished benchmark.
