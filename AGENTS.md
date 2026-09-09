# Repository instructions

- Build Proceedings -> Every Eval Ever (EEE). Evaluation Cards are downstream and are not an output of this repository.
- Use English for code, prompts, tests, schemas, and technical documentation.
- Preserve the stage boundary: frozen sources -> layout fragments -> candidate observations -> role/scope resolution -> independent evidence and referential checks -> deterministic EEE composition -> pinned-schema validation -> review.
- Never guess missing values or identities. Preserve raw names and values and route uncertainty to review.
- Every numeric result must retain a paper/source ID plus page and table/figure/prose anchor. A URL alone is not evidence.
- LLMs may propose candidates. They may not write final EEE records directly.
- Do not commit PDFs, raw provider traces, credentials, private annotations, caches, absolute local paths, or copyrighted source snapshots.
- Keep private experiment logs and internal meeting or decision references out of public comments; describe regression cases with synthetic identifiers.
- Default tests are offline and deterministic. Network and provider runs must be explicit.
- Test distinct failure modes and observable behavior. Reuse common setup and parameterize variants; avoid literal-only assertions and tests that pin explanatory wording.
- Reuse from Auto-BenchmarkCards only through the documented narrow interfaces and provenance ledger; do not bulk-copy it.
- The only authorized push target for this project is `https://github.com/evaleval/proceedings-to-paper.git`.
