# Results of the two runs

The pipeline has been run on real papers in two versions. The September 2026 run
covered 105 of the 3,251 papers in the toxicity paper collection; the October 2026
run applied the current code to ten of those papers, so that the two versions can be
compared on identical input. The files behind every figure here are in the
[results folder](../results/README.md). Figures are either measured on saved
artifacts or judged by a model, and the text says which.

## The September run

The September run exported 317 EEE records carrying 1,192 values from 56 papers.
A model-assisted check of sixty sampled values found the numbers correct but the
labels unusable: only thirteen values named a recognisable dataset, eight values
that belonged to other teams in shared-task leaderboards had been exported as the
paper's own, and 69 percent of all values lacked a split. The records also failed
validation against the current EEE schema (0.3.0). In the export funnel, 579
reviewer-accepted candidates were dropped for a missing metric identity and 249
for a missing scale. The run had read at most eight pages per paper and six table
rows per model call, so results in appendices after the references were never read.

## What was changed

Schema 0.3.0 is the default authority; metric names are resolved through a curated
snapshot of the EEE metric registry; a metric without a known scale is exported
with explicit null bounds instead of being dropped. Captions are attached to
tables within twelve lines and shown to the extractor. All pages are read, with no
cap on blocks, and bibliography blocks are filtered out. Producer origin follows a
versioned cue lexicon in which ranked and official-results tables abstain, a
reviewer who accepts a value without naming its producer leaves it without an
origin basis, and a row whose label names the paper itself is the one row such
tables do not hold back. One extra model call per paper proposes a quote-anchored
inventory of the datasets the paper evaluates on, and candidates are bound to it by
deterministic rules that never rewrite the printed string.

## Measured effect

Replaying the new offline stages over the September extraction gives 324 records
with 1,227 values from 61 papers, all valid under schema 0.3.0. The values that a
reviewer had accepted without naming a producer, and the values from ranked
tables other than the paper's own row, are no longer exported; the number of
values with a recognisable dataset name rose from 838 to 904.

On the ten re-extracted papers the current code read 121 pages where September
had read 72, and proposed 1,331 candidates against 976 on the nine papers that ran
to completion; the tenth paper stopped at its per-paper ceiling during the row
stage and contributes no values. The run exported 452 values in 73 records, all of
which validate against schema 0.3.0, where the September export had 510 values on
the same ten papers. Of the 37 sampled September values that fall in these papers,
24 are exported again, ten are withheld, and three belong to the stopped paper. By
the September judgement, four of the withheld values are other teams' leaderboard
rows, two are the paper's own runs of a baseline, two are signed differences and
one could not be attributed; one is the paper's own result that the stricter rule
now holds back. Half of the exported values are bound to a verified dataset entry
(121 by exact name, 24 by the anchored table's caption, 81 as the sole dataset of
the paper), and none carries a split word or an empty string as its dataset name;
whether the bound names are correct awaits the human check.

The rerun also exposed a defect the September caps had hidden: once a block holds
roughly fifty or more numeric cells, the extractor model returns one candidate and
stops. Chunks are now bounded by 24 numeric cells, and the figures above are from
the run under that bound. The same model occasionally returns almost nothing for a
dense block of ordinary size, so the code now calls such a block once more and
keeps the attempt with more candidates; the run reported here predates that retry.

## What remains open

The share of values with a split is 30 percent, as in September, because the
papers state the split in prose or not at all. System identity across papers is
untouched by design, and the EEE registry holds almost none of the datasets in this
collection. Every quality figure on this page is model-judged until the sample
sheets are scored.
