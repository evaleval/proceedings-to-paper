from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.evaluation import unseen_annotation as unseen
from proceedings_to_eee.io import read_json, sha256_bytes, sha256_file, write_json, write_jsonl
from proceedings_to_eee.run_seal import seal_run_tree


def _digest(value: str) -> str:
    return sha256_bytes(value.encode())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def future_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("unseen-evaluation")
    public_repo = root / "public" / "proceedings-to-paper"
    private = root / "private"
    source_root = private / "sources"
    raw_run = private / "raw-run"
    public_repo.mkdir(parents=True)
    (source_root / "cache").mkdir(parents=True)
    raw_run.mkdir(parents=True)

    registry_path = private / "prior-paper-registry.json"
    write_json(
        registry_path,
        unseen.PriorPaperRegistry(
            snapshot_id="complete-development-registry",
            inspected_or_development_paper_ids=["old-development-paper"],
        ),
    )
    papers = [
        {
            "paper_id": f"future-paper-{index:02d}",
            "title": f"Synthetic metadata-only paper {index}",
            "year": 2026,
            "venue": "Synthetic Venue",
            "publisher": "Synthetic Publisher",
            "landing_url": f"https://example.test/papers/{index}",
            "layout_stratum": "two-column",
        }
        for index in range(10)
    ]
    stages = [
        {
            "stage": name,
            "model": f"frozen-model/{name}",
            "prompt_sha256": _digest(f"prompt:{name}"),
            "provider_schema_sha256": _digest(f"schema:{name}"),
            "request_contract_sha256": _digest(f"request:{name}"),
            "settings_sha256": _digest(f"settings:{name}"),
        }
        for name in (
            "block_extraction",
            "row_disposition",
            "tuple_resolution",
            "origin_retrieval",
            "verification",
        )
    ]
    freeze_payload: dict[str, Any] = {
        "schema_version": unseen.FREEZE_SCHEMA_VERSION,
        "status": "locked-before-metadata-selection",
        "freeze_id": "future-eval-freeze",
        "code_git_commit": "1" * 40,
        "code_source_tree_sha256": _digest("code-tree"),
        "dependency_lock_sha256": _digest("dependency-lock"),
        "thresholds_sha256": _digest("thresholds"),
        "attribution_policy_sha256": _digest("attribution"),
        "stages": stages,
    }
    freeze_payload["execution_contract_sha256"] = unseen.execution_contract_sha256(freeze_payload)
    freeze_path = private / "engineering-freeze.json"
    write_json(freeze_path, unseen.EngineeringFreezeManifest.model_validate(freeze_payload))
    selection_path = private / "metadata-selection.json"
    write_json(
        selection_path,
        unseen.MetadataSelectionManifest(
            engineering_freeze_sha256=sha256_file(freeze_path),
            prior_registry_sha256=sha256_file(registry_path),
            paper_count=len(papers),
            papers=papers,
        ),
    )
    sampling_path = private / "sampling-frame.json"
    write_json(
        sampling_path,
        unseen.SamplingFrame(
            engineering_freeze_sha256=sha256_file(freeze_path),
            metadata_selection_sha256=sha256_file(selection_path),
            randomization_seed="a" * 64,
            eligible_atomic_result_target=100,
            control_targets={
                unseen.PoolStratum.NON_RESULT_CANDIDATE: 20,
                unseen.PoolStratum.EXTERNAL_RESULT_CANDIDATE: 20,
                unseen.PoolStratum.SECONDARY_RESULT_CANDIDATE: 10,
            },
            minimum_sampled_papers=10,
        ),
    )
    retry_path = private / "retry-policy.json"
    write_json(
        retry_path,
        unseen.TransportRetryPolicy(
            engineering_freeze_sha256=sha256_file(freeze_path),
            metadata_selection_sha256=sha256_file(selection_path),
            maximum_attempts_per_request=2,
            allowed_retry_reasons=[
                unseen.TransportFailureReason.TIMEOUT,
                unseen.TransportFailureReason.CONNECTION_RESET,
            ],
        ),
    )
    corpus_path = raw_run / unseen.CORPUS_DEFINITION_PATH
    write_json(
        corpus_path,
        unseen.FrozenCorpusDefinition(
            engineering_freeze_sha256=sha256_file(freeze_path),
            metadata_selection_sha256=sha256_file(selection_path),
            paper_count=len(papers),
            paper_ids=[paper["paper_id"] for paper in papers],
        ),
    )

    pool_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for paper_index, paper_metadata in enumerate(papers):
        paper_id = paper_metadata["paper_id"]
        paper_run = raw_run / paper_id
        private_run = paper_run / "private"
        record_dir = private_run / "source-records"
        record_dir.mkdir(parents=True)
        source_id = f"source-{paper_index:02d}"
        pdf_bytes = b"%PDF-1.4\n" + f"synthetic {paper_id}\n%%EOF\n".encode()
        pdf_path = source_root / "cache" / f"{paper_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        source_manifest_path = paper_run / "source-manifest.json"
        write_json(
            source_manifest_path,
            {
                "paper_id": paper_id,
                "sources": [
                    {
                        "source_id": source_id,
                        "role": "paper",
                        "sha256": sha256_file(pdf_path),
                        "byte_size": len(pdf_bytes),
                        "cache_relpath": f"cache/{paper_id}.pdf",
                    }
                ],
            },
        )
        candidates = [
            *(
                (
                    unseen.PoolStratum.ELIGIBLE_ATOMIC_RESULT_CANDIDATE,
                    f"{paper_id} eligible result {index}: accuracy = 0.{paper_index}{index}",
                )
                for index in range(10)
            ),
            *(
                (
                    unseen.PoolStratum.NON_RESULT_CANDIDATE,
                    f"{paper_id} non-result sample count {index}: n = {100 + index}",
                )
                for index in range(2)
            ),
            *(
                (
                    unseen.PoolStratum.EXTERNAL_RESULT_CANDIDATE,
                    f"{paper_id} external result {index}: score = 0.{index}5",
                )
                for index in range(2)
            ),
            (
                unseen.PoolStratum.SECONDARY_RESULT_CANDIDATE,
                f"{paper_id} secondary result: robustness = 0.4",
            ),
        ]
        page_one = "\n".join(excerpt for _, excerpt in candidates)
        page_two = "\n".join(
            (
                f"{paper_id} authors ran all reported experiments.",
                f"{paper_id} external values were copied from cited work.",
            )
        )
        layout_path = private_run / "layout.json"
        write_json(
            layout_path,
            {
                "source_id": source_id,
                "pages": [
                    {"page": 1, "text": page_one, "text_sha256": _digest(page_one)},
                    {"page": 2, "text": page_two, "text_sha256": _digest(page_two)},
                ],
            },
        )
        run_path = paper_run / "run.json"
        write_json(run_path, {"paper_id": paper_id, "status": "completed-once"})
        source_manifest_sha = sha256_file(source_manifest_path)
        layout_sha = sha256_file(layout_path)
        for item_index, (stratum, excerpt) in enumerate(candidates):
            record_path = record_dir / f"record-{item_index:02d}.json"
            write_json(record_path, {"paper_id": paper_id, "locator": f"row-{item_index}"})
            relative_record = record_path.relative_to(raw_run).as_posix()
            payload = {
                "schema_version": unseen.POOL_SCHEMA_VERSION,
                "pipeline_stratum": stratum,
                "paper_id": paper_id,
                "source_id": source_id,
                "source_sha256": sha256_file(pdf_path),
                "source_manifest_sha256": source_manifest_sha,
                "layout_sha256": layout_sha,
                "page": 1,
                "page_text_sha256": _digest(page_one),
                "locator": f"row-{item_index:02d}",
                "source_excerpt": excerpt,
                "source_excerpt_sha256": _digest(excerpt),
                "source_record_path": relative_record,
                "source_record_sha256": sha256_file(record_path),
            }
            payload["unit_id"] = unseen.annotation_pool_unit_id(
                {
                    key: (value.value if isinstance(value, unseen.PoolStratum) else value)
                    for key, value in payload.items()
                    if key != "unit_id"
                }
            )
            pool_rows.append(
                unseen.AnnotationPoolRecord.model_validate(payload).model_dump(mode="json")
            )
        write_json(
            private_run / unseen.PAPER_BINDING_NAME,
            unseen.PaperFreezeBinding(
                paper_id=paper_id,
                freeze_manifest_sha256=sha256_file(freeze_path),
                execution_contract_sha256=freeze_payload["execution_contract_sha256"],
                run_manifest_sha256=sha256_file(run_path),
                source_manifest_sha256=source_manifest_sha,
                layout_sha256=layout_sha,
            ),
        )
        attempts.append(
            unseen.TransportAttempt(
                request_slot="document",
                request_fingerprint=_digest(f"request:{paper_id}"),
                call_id=f"call-{paper_index:02d}",
                paper_id=paper_id,
                stage="block_extraction",
                attempt_number=1,
                outcome=unseen.AttemptOutcome.SUCCESS,
            ).model_dump(mode="json")
        )

    pool_path = raw_run / unseen.POOL_PATH
    attempts_path = raw_run / unseen.ATTEMPTS_PATH
    write_jsonl(pool_path, pool_rows)
    write_jsonl(attempts_path, attempts)
    write_json(
        raw_run / unseen.RUN_RECEIPT_PATH,
        unseen.SealedRunReceipt(
            freeze_manifest_sha256=sha256_file(freeze_path),
            execution_contract_sha256=freeze_payload["execution_contract_sha256"],
            metadata_selection_sha256=sha256_file(selection_path),
            prior_registry_sha256=sha256_file(registry_path),
            sampling_frame_sha256=sha256_file(sampling_path),
            retry_policy_sha256=sha256_file(retry_path),
            corpus_definition_sha256=sha256_file(corpus_path),
            annotation_pool_sha256=sha256_file(pool_path),
            annotation_pool_records=len(pool_rows),
            transport_attempts_sha256=sha256_file(attempts_path),
            transport_attempt_records=len(attempts),
        ),
    )
    sealed_run = private / "sealed-run"
    seal_run_tree(raw_run, sealed_run)
    return {
        "root": root,
        "private": private,
        "public_repo_root": public_repo,
        "source_project_root": source_root,
        "raw_run_root": raw_run,
        "sealed_run_root": sealed_run,
        "metadata_selection_path": selection_path,
        "prior_registry_path": registry_path,
        "freeze_manifest_path": freeze_path,
        "sampling_frame_path": sampling_path,
        "retry_policy_path": retry_path,
    }


def _workspace_args(fixture: dict[str, Path], name: str) -> dict[str, Path]:
    return {
        "metadata_selection_path": fixture["metadata_selection_path"],
        "prior_registry_path": fixture["prior_registry_path"],
        "freeze_manifest_path": fixture["freeze_manifest_path"],
        "sampling_frame_path": fixture["sampling_frame_path"],
        "retry_policy_path": fixture["retry_policy_path"],
        "sealed_run_root": fixture["sealed_run_root"],
        "source_project_root": fixture["source_project_root"],
        "output_dir": fixture["private"] / name,
        "public_repo_root": fixture["public_repo_root"],
    }


def test_prepares_three_isolated_entirely_blank_packages(
    future_fixture: dict[str, Path],
) -> None:
    args = _workspace_args(future_fixture, "workspace-isolation")
    manifest, digest = unseen.prepare_unseen_annotation_workspace(**args)
    assert manifest.denominator == 150
    assert manifest.eligible_atomic_result_target == 100
    assert manifest.control_target == 50
    assert manifest.paper_count == 10
    assert digest == sha256_file(args["output_dir"] / "manifest.json")
    assert (
        unseen.validate_initial_unseen_annotation_workspace(
            workspace_dir=args["output_dir"],
            **{key: value for key, value in args.items() if key != "output_dir"},
        )
        == manifest
    )

    item_payloads: list[bytes] = []
    for package in manifest.packages:
        package_dir = args["output_dir"] / package.directory
        item_payload = (package_dir / "items.jsonl").read_bytes()
        item_payloads.append(item_payload)
        assert b"pipeline_stratum" not in item_payload
        assert b"unit_id" not in item_payload
        assert b"source_record" not in item_payload
        responses = _read_jsonl(package_dir / "response.jsonl")
        assert len(responses) == 150
        assert all(response["result_bearing"] is None for response in responses)
        assert len(list((package_dir / "pdfs").glob("*.pdf"))) == 10
    assert len(set(item_payloads)) == 1

    package_a = (args["output_dir"] / "package-a" / "manifest.json").read_text()
    package_b = (args["output_dir"] / "package-b" / "manifest.json").read_text()
    package_c = (args["output_dir"] / "package-c" / "manifest.json").read_text()
    assert "reviewer-b" not in package_a and "adjudicator-primary" not in package_a
    assert "reviewer-a" not in package_b and "adjudicator-primary" not in package_b
    assert "reviewer-a" not in package_c and "reviewer-b" not in package_c
    coordinator = (args["output_dir"] / "manifest.json").read_text()
    assert "source_excerpt" not in coordinator
    assert "reviewer-a" not in coordinator and "reviewer-b" not in coordinator


def test_completed_response_requires_strict_separate_evidence(
    future_fixture: dict[str, Path],
) -> None:
    args = _workspace_args(future_fixture, "workspace-completed")
    workspace, _ = unseen.prepare_unseen_annotation_workspace(**args)
    package_dir = args["output_dir"] / "package-a"
    package_receipt = workspace.packages[0].manifest_sha256
    items = [
        unseen.UnseenAnnotationItem.model_validate(row)
        for row in _read_jsonl(package_dir / "items.jsonl")
    ]
    completed: list[dict[str, Any]] = []
    used_uncertain_origin = False
    for item in items:
        result_evidence = unseen.EvidenceAnchor(
            source_id=item.source_id,
            page=item.page,
            exact_excerpt=item.source_excerpt,
        )
        if "non-result" in item.source_excerpt:
            response = unseen.UnseenAnnotationResponse(
                item_id=item.item_id,
                actor_id="reviewer-a",
                result_bearing=unseen.ResultBearing.NO,
                result_evidence=result_evidence,
                confidence=unseen.Confidence.HIGH,
            )
        else:
            is_external = "external result" in item.source_excerpt
            if not used_uncertain_origin and "eligible result" in item.source_excerpt:
                origin = unseen.ResultOrigin.UNCERTAIN
                origin_evidence = None
                used_uncertain_origin = True
            else:
                origin = (
                    unseen.ResultOrigin.EXTERNALLY_SOURCED
                    if is_external
                    else unseen.ResultOrigin.PAPER_PRODUCED
                )
                origin_evidence = unseen.EvidenceAnchor(
                    source_id=item.source_id,
                    page=2,
                    exact_excerpt=(
                        f"{item.paper_id} external values were copied from cited work."
                        if is_external
                        else f"{item.paper_id} authors ran all reported experiments."
                    ),
                )
            response = unseen.UnseenAnnotationResponse(
                item_id=item.item_id,
                actor_id="reviewer-a",
                result_bearing=unseen.ResultBearing.YES,
                result_evidence=result_evidence,
                origin=origin,
                origin_evidence=origin_evidence,
                claim_scope=(
                    unseen.ClaimScope.SECONDARY
                    if "secondary result" in item.source_excerpt
                    else unseen.ClaimScope.PRIMARY
                ),
                atomicity=unseen.Atomicity.ATOMIC,
                confidence=unseen.Confidence.HIGH,
            )
        completed.append(response.model_dump(mode="json"))
    assert used_uncertain_origin
    write_jsonl(package_dir / "response.jsonl", completed)
    validated = unseen.validate_completed_unseen_reviewer_response(
        package_dir=package_dir,
        sealed_run_root=future_fixture["sealed_run_root"],
        public_repo_root=future_fixture["public_repo_root"],
        expected_manifest_sha256=package_receipt,
    )
    assert validated.actor_id == "reviewer-a"
    assert validated.response_count == 150
    assert any(origin is None for _, origin in validated.evidence)
    assert any(origin is not None and origin.page == 2 for _, origin in validated.evidence)

    wrong_page = [dict(row) for row in completed]
    wrong_page[0] = dict(wrong_page[0])
    wrong_page[0]["result_evidence"] = dict(wrong_page[0]["result_evidence"])
    wrong_page[0]["result_evidence"]["page"] = 2
    write_jsonl(package_dir / "response.jsonl", wrong_page)
    with pytest.raises(ValueError, match="selected item page"):
        unseen.validate_completed_unseen_reviewer_response(
            package_dir=package_dir,
            sealed_run_root=future_fixture["sealed_run_root"],
            public_repo_root=future_fixture["public_repo_root"],
            expected_manifest_sha256=package_receipt,
        )

    wrong_whitespace = [dict(row) for row in completed]
    wrong_whitespace[0] = dict(wrong_whitespace[0])
    wrong_whitespace[0]["result_evidence"] = dict(wrong_whitespace[0]["result_evidence"])
    wrong_whitespace[0]["result_evidence"]["exact_excerpt"] += " "
    write_jsonl(package_dir / "response.jsonl", wrong_whitespace)
    with pytest.raises(ValueError, match="absent from one physical line"):
        unseen.validate_completed_unseen_reviewer_response(
            package_dir=package_dir,
            sealed_run_root=future_fixture["sealed_run_root"],
            public_repo_root=future_fixture["public_repo_root"],
            expected_manifest_sha256=package_receipt,
        )


def test_rejects_seen_papers_and_underpowered_frames(
    future_fixture: dict[str, Path],
) -> None:
    selection = read_json(future_fixture["metadata_selection_path"])
    selection["papers"] = selection["papers"][:9]
    selection["paper_count"] = 9
    with pytest.raises(ValidationError):
        unseen.MetadataSelectionManifest.model_validate(selection)
    sampling = read_json(future_fixture["sampling_frame_path"])
    sampling["eligible_atomic_result_target"] = 99
    with pytest.raises(ValidationError):
        unseen.SamplingFrame.model_validate(sampling)
    sampling = read_json(future_fixture["sampling_frame_path"])
    sampling["control_targets"][unseen.PoolStratum.NON_RESULT_CANDIDATE.value] = 19
    with pytest.raises(ValidationError):
        unseen.SamplingFrame.model_validate(sampling)

    overlap_registry = future_fixture["private"] / "overlap-registry.json"
    write_json(
        overlap_registry,
        unseen.PriorPaperRegistry(
            snapshot_id="overlap",
            inspected_or_development_paper_ids=["future-paper-00"],
        ),
    )
    overlap_selection = future_fixture["private"] / "overlap-selection.json"
    selected = read_json(future_fixture["metadata_selection_path"])
    selected["prior_registry_sha256"] = sha256_file(overlap_registry)
    write_json(overlap_selection, selected)
    args = _workspace_args(future_fixture, "workspace-overlap")
    args["metadata_selection_path"] = overlap_selection
    args["prior_registry_path"] = overlap_registry
    with pytest.raises(ValueError, match="inspected/development"):
        unseen.prepare_unseen_annotation_workspace(**args)


def test_rejects_unsealed_and_tampered_pipeline_trees(
    future_fixture: dict[str, Path],
) -> None:
    args = _workspace_args(future_fixture, "workspace-unsealed")
    args["sealed_run_root"] = future_fixture["raw_run_root"]
    with pytest.raises(ValueError):
        unseen.prepare_unseen_annotation_workspace(**args)

    tampered = future_fixture["private"] / "tampered-sealed-run"
    shutil.copytree(future_fixture["sealed_run_root"], tampered)
    layout = tampered / "future-paper-00" / "private" / "layout.json"
    layout.write_text(layout.read_text() + " ")
    args = _workspace_args(future_fixture, "workspace-tampered")
    args["sealed_run_root"] = tampered
    with pytest.raises(ValueError):
        unseen.prepare_unseen_annotation_workspace(**args)


def test_rejects_retry_outside_predeclared_transport_policy(
    future_fixture: dict[str, Path],
) -> None:
    retry_policy = unseen.TransportRetryPolicy.model_validate(
        read_json(future_fixture["retry_policy_path"])
    )
    changed_request = [
        unseen.TransportAttempt(
            request_slot="page-1",
            request_fingerprint=_digest("original-request"),
            call_id="changed-request-1",
            paper_id="future-paper-00",
            stage="block_extraction",
            attempt_number=1,
            outcome=unseen.AttemptOutcome.TRANSPORT_FAILURE,
            failure_reason=unseen.TransportFailureReason.TIMEOUT,
        ),
        unseen.TransportAttempt(
            request_slot="page-1",
            request_fingerprint=_digest("changed-request"),
            call_id="changed-request-2",
            paper_id="future-paper-00",
            stage="block_extraction",
            attempt_number=2,
            outcome=unseen.AttemptOutcome.SUCCESS,
            retry_reason=unseen.TransportFailureReason.TIMEOUT,
        ),
    ]
    with pytest.raises(ValueError, match="changed the frozen logical request"):
        unseen._validate_attempts(
            changed_request,
            policy=retry_policy,
            paper_ids={"future-paper-00"},
        )

    variant_raw = future_fixture["private"] / "invalid-retry-raw"
    shutil.copytree(future_fixture["raw_run_root"], variant_raw)
    attempts_path = variant_raw / unseen.ATTEMPTS_PATH
    attempts = [
        unseen.TransportAttempt(
            request_slot="document",
            request_fingerprint=_digest("invalid-retry-request"),
            call_id="invalid-call-1",
            paper_id="future-paper-00",
            stage="block_extraction",
            attempt_number=1,
            outcome=unseen.AttemptOutcome.SCHEMA_FAILURE,
        ).model_dump(mode="json"),
        unseen.TransportAttempt(
            request_slot="document",
            request_fingerprint=_digest("invalid-retry-request"),
            call_id="invalid-call-2",
            paper_id="future-paper-00",
            stage="block_extraction",
            attempt_number=2,
            outcome=unseen.AttemptOutcome.SUCCESS,
            retry_reason=unseen.TransportFailureReason.TIMEOUT,
        ).model_dump(mode="json"),
    ]
    write_jsonl(attempts_path, attempts)
    receipt_path = variant_raw / unseen.RUN_RECEIPT_PATH
    receipt = read_json(receipt_path)
    receipt["transport_attempts_sha256"] = sha256_file(attempts_path)
    receipt["transport_attempt_records"] = len(attempts)
    write_json(receipt_path, receipt)
    variant_sealed = future_fixture["private"] / "invalid-retry-sealed"
    seal_run_tree(variant_raw, variant_sealed)
    args = _workspace_args(future_fixture, "workspace-invalid-retry")
    args["sealed_run_root"] = variant_sealed
    with pytest.raises(ValueError, match="technical transport failure"):
        unseen.prepare_unseen_annotation_workspace(**args)


def test_rejects_prefilled_answers_extra_files_and_denominator_changes(
    future_fixture: dict[str, Path],
) -> None:
    args = _workspace_args(future_fixture, "workspace-adversarial")
    unseen.prepare_unseen_annotation_workspace(**args)
    package = args["output_dir"] / "package-a"
    responses = _read_jsonl(package / "response.jsonl")
    item = unseen.UnseenAnnotationItem.model_validate(_read_jsonl(package / "items.jsonl")[0])
    responses[0] = unseen.UnseenAnnotationResponse(
        item_id=item.item_id,
        actor_id="reviewer-a",
        result_bearing=unseen.ResultBearing.NO,
        result_evidence=unseen.EvidenceAnchor(
            source_id=item.source_id,
            page=item.page,
            exact_excerpt=item.source_excerpt,
        ),
        confidence=unseen.Confidence.HIGH,
    ).model_dump(mode="json")
    write_jsonl(package / "response.jsonl", responses)
    with pytest.raises(ValueError):
        unseen.validate_initial_unseen_annotation_workspace(
            workspace_dir=args["output_dir"],
            **{key: value for key, value in args.items() if key != "output_dir"},
        )

    extra_args = _workspace_args(future_fixture, "workspace-extra-file")
    unseen.prepare_unseen_annotation_workspace(**extra_args)
    (extra_args["output_dir"] / "package-b" / "peer-output.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected file contract"):
        unseen.validate_initial_unseen_annotation_workspace(
            workspace_dir=extra_args["output_dir"],
            **{key: value for key, value in extra_args.items() if key != "output_dir"},
        )

    changed_sampling = future_fixture["private"] / "changed-denominator.json"
    sampling = read_json(future_fixture["sampling_frame_path"])
    sampling["eligible_atomic_result_target"] = 101
    write_json(changed_sampling, sampling)
    denominator_args = _workspace_args(future_fixture, "workspace-denominator")
    denominator_args["sampling_frame_path"] = changed_sampling
    with pytest.raises(ValueError, match="run receipt"):
        unseen.prepare_unseen_annotation_workspace(**denominator_args)

    pool = tuple(
        unseen.AnnotationPoolRecord.model_validate(row)
        for row in _read_jsonl(future_fixture["sealed_run_root"] / unseen.POOL_PATH)
    )
    larger_frame = unseen.SamplingFrame.model_validate(sampling)
    with pytest.raises(ValueError, match="cannot satisfy"):
        unseen._sample_pool(pool, larger_frame)


def test_rejects_freeze_and_per_paper_binding_mismatches(
    future_fixture: dict[str, Path],
) -> None:
    changed_freeze_path = future_fixture["private"] / "changed-freeze.json"
    changed_freeze = read_json(future_fixture["freeze_manifest_path"])
    changed_freeze["thresholds_sha256"] = _digest("changed-thresholds")
    changed_freeze["execution_contract_sha256"] = unseen.execution_contract_sha256(changed_freeze)
    write_json(
        changed_freeze_path,
        unseen.EngineeringFreezeManifest.model_validate(changed_freeze),
    )
    args = _workspace_args(future_fixture, "workspace-changed-freeze")
    args["freeze_manifest_path"] = changed_freeze_path
    with pytest.raises(ValueError, match="another freeze"):
        unseen.prepare_unseen_annotation_workspace(**args)

    variant_raw = future_fixture["private"] / "bad-binding-raw"
    shutil.copytree(future_fixture["raw_run_root"], variant_raw)
    binding_path = variant_raw / "future-paper-00" / "private" / unseen.PAPER_BINDING_NAME
    binding = read_json(binding_path)
    binding["layout_sha256"] = "0" * 64
    write_json(binding_path, binding)
    variant_sealed = future_fixture["private"] / "bad-binding-sealed"
    seal_run_tree(variant_raw, variant_sealed)
    args = _workspace_args(future_fixture, "workspace-bad-binding")
    args["sealed_run_root"] = variant_sealed
    with pytest.raises(ValueError, match="per-paper freeze binding"):
        unseen.prepare_unseen_annotation_workspace(**args)
