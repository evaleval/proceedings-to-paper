# Release contents and reproducibility

This release contains the implementation, offline tests, pinned schemas and resource
configuration, synthetic examples, and technical documentation. A dated aggregate
summarizes the existing development census without publishing its source material.

## Included

- Python pipeline and CLI, including extraction, validation, review, composition,
  census processing, and evaluation tools.
- Synthetic tests and examples that run without provider credentials.
- EEE 0.2.2 schema, license, and deterministic metric/origin configuration.
- A generated pipeline illustration and its editable source.
- Aggregate counts with hashes binding them to the locally retained run summaries.

## Kept outside the repository

Source PDFs, extracted source text, raw provider payloads, individual observations
from real papers, private corpus selections, human responses, account information,
credentials, local paths, personal notes, meeting records, and development diaries
are not release inputs. Historical experiment ledgers and paper-level preview
snapshots are also excluded. Generating a review report does not authorize publishing
its contents.

The real 150-item annotation packet remains private because it contains source-page
text. The public examples are synthetic. New annotation responses should be stored
outside the checkout or in an ignored local working directory.

## What can be reproduced

The test suite, installed-package checks, synthetic two-page demo, synthetic EEE
validation, and synthetic annotation scoring can be reproduced offline. A new real
paper run requires separately obtained source documents and explicit provider access.
The published aggregate is a projection of a stored development run, not a new result
obtained by rerunning this release. Its underlying private paper set is not bundled.

## Checking a release

The [CI workflow template](ci-workflow.yml) can be installed by a repository owner.
Until it is installed under `.github/workflows/`, checks are run locally.

Run `python scripts/check_public_release.py` against the Git index before committing,
then run the tests, lint, format, build, and installed-package checks listed in the
[CI workflow template](ci-workflow.yml). The checker enforces file and content
boundaries; it complements human inspection. It does not prove that every permissible
sentence is appropriate to share.

Review both the candidate tree and every reachable branch/tag history before sharing
a repository. Removing a file from the latest commit does not remove it from older
commits. Distribution archives need the same inspection because packaging rules can
include files that the source checkout otherwise ignores.
