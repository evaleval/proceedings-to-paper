# Results

This folder holds what the pipeline produced on real papers, in two versions, and a
comparison between them. Every table row and every record points back to the paper,
the page and the table or figure it came from, so a reader can open the PDF at the
cited page and check the value. No source text is distributed: the records carry
hashes of the quotations the pipeline verified, not the quotations themselves, and
the tables carry only printed names, values and table labels.

## Folders

`run-2026-09/` is the September 2026 run of the earlier code over 105 papers of the
toxicity paper collection: 317 EEE records with 1,192 values from 56 papers. The
folder is the hand-over made at the time, unchanged, with its aggregate summary.

`run-2026-10-subset/` is the October 2026 run of the current code over ten of those
papers (the four papers reviewed by hand in September and the six that contributed
most values to the September sample): 73 EEE records with 452 values from nine
papers; the tenth paper stopped at its per-paper ceiling before any value was
written, and its row in `papers.csv` says so.

`comparison/` matches the two runs on the same ten papers.

## Files in each run folder

`results.csv` has one row per exported value: paper, system, dataset, split, subset,
language, scope, metric, the value as printed and as a number, the unit, the page,
the table, the row and column labels, a link that opens the PDF on the cited page,
and the EEE record the value belongs to. The `ok` and `comment` columns are empty
and are meant for a reader's own verdicts.

`papers.csv` has one row per processed paper with its identifiers, title and PDF
link, the number of values proposed, the number of EEE records and exported values,
and, for a paper without records, a note saying why.

`eee/<paper>/` holds the EEE records as JSON, one file per record, as the run wrote
them. The September records follow schema 0.2.2, the October records schema 0.3.0.

`summary.json` is an aggregate of the run: paper and candidate counts, review
decisions, records by tier and by producer-origin basis, and the settings that
differ between the two versions.

## Files in the comparison folder

`papers.csv` has one row per paper with the values proposed and exported in each
run and the number of values kept, lost and gained. `values.csv` has one row per
value of either run, matched across runs by paper, page, printed value and metric,
with the system, dataset and split each run assigned to it and whether the value
was kept, appears only in the old run or only in the new one. A value that appears
only in the old run is not necessarily a loss: the stricter origin rule of the new
version withholds other teams' leaderboard rows and signed differences on purpose.
