# Human annotation: current status and next design

Human annotation is not required to share the code for collaboration. It is
required before making stronger accuracy, precision, or agreement claims.

A first private reviewer response has been validated, but it remains outside
the public repository. A second reviewer and adjudication are deliberately
deferred, so there is no inter-annotator agreement or adjudicated performance
estimate to report.

The current packet also exposed two design issues that should be fixed before a
future study:

1. Result evidence and origin evidence need separate fields. A quantitative
   result may be on a table page while the sentence establishing who produced
   it is on a methods page. Each excerpt should have its own page binding and
   strict exact-substring check. Evidence should not pass through whitespace
   normalization or by pooling multiple extraction modes.
2. The sample must contain genuine negative cases. Sampling only numeric body
   rows cannot estimate rejection of headers, sample counts, method parameters,
   or other non-result rows. A future packet should stratify result rows and
   several kinds of non-result rows before annotation begins.

A backward-compatible future response can retain the current label fields and
add optional `result_evidence`, `result_evidence_page`, `origin_evidence`, and
`origin_evidence_page` fields. Existing private responses should not be mutated
retroactively; the improved schema should apply to a new packet.
