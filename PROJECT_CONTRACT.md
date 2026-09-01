# Project contract

## Objective

Build a reproducible Proceedings -> EEE pipeline for NeurIPS Challenge Track A: parse evaluation results reported in papers into EEE JSON. The competition is a consumer and test setting, not an architectural dependency. Track B experiment reruns and Evaluation Cards are out of scope for this artifact.

## Scientific unit

One immutable Candidate Observation represents one reported value for one evaluated-system snapshot, one dataset and exact scope, one metric and scale, one setting, and one or more exact evidence anchors. An EEE record is a deterministic projection of eligible candidates, never the extraction object.

## Export policy

A candidate can enter canonical EEE only when it is positively established as a result produced by the current paper, is a primary-result proposal, has an explicit value/system/dataset/metric/scope, has a source quote containing the raw value, has resolved metric direction and scale, passes the confidence threshold, and has no unresolved semantic conflict. `claim_type` is model self-report and is not evidence of producer origin. Externally sourced, unresolved, and no-signal candidates remain evidence-bearing in the candidate/review layer; `NO_SIGNAL` never means paper-produced.

## Reproducibility envelope

Every run records source hashes, parser/version, selected pages, provider, requested and returned model, actual provider, request ID, settings, prompt and response hashes, token usage, cost, retries, code state, EEE schema version/hash, counts, and wall time. An enabled row stage additionally records its pre-call plan and hard bounds, independent prompt/schema hashes, per-row dispositions, invalid/unresolved/unbatchable rows, recovery attempts, and checkpoint state. Provider secrets and raw responses are excluded.

## Evaluation boundary

Development spot checks are stored separately and are evaluated only after extraction. They never appear in prompts. Claims about generalization require a future sealed test set; the ten-paper pilot is a demonstration and error study, not a universal benchmark.

## Non-goals

- Evaluation Card or Benchmark Card generation.
- Silent repair, guessed identities, or scope collapse.
- Figure digitization without raw data.
- Republishing papers, supplements, or raw model traces.
- Rerunning missing experiments or submitting data to an external EEE datastore.
