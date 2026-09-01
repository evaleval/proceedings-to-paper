# Offline quickstart

The files in this directory are synthetic. They contain no paper text, provider
payload, private annotation, or measured accuracy claim.

## Prepare a corpus

`corpus-template.yaml` is a valid one-paper development corpus. Copy it,
replace the example metadata and PDF URL, and use the resulting file with
`ere run-corpus`. The main README documents the paid provider-backed command
and the development-split boundary.

## Inspect the review layer

`synthetic-extraction-review-card.html` and its machine-readable JSON show
the pipeline's current principal output. The example contains one withheld
candidate, explicit abstention reasons, and zero EEE records. It was generated from
an offline fixture and is not an extraction-quality result.

## Validate the packaged EEE schema

`synthetic-eee.json` is a hand-written EEE 0.2.2 record. It checks the
installed schema validator and was not extracted from a paper.

```bash
uv run ere validate-eee examples/quickstart/synthetic-eee.json
```

The expected result is schema `0.2.2` with an empty `issues` list. After
installing the wheel, the same command works from any directory when the record path
is absolute.
