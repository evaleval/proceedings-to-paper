"""The model reviewer and its local defence.

A reviewer that cannot show its evidence has told us nothing, not something negative, so
every failed quotation check abstains rather than rejects. Nothing here is a human label
and nothing here measures an error rate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.census_review import (
    ReviewItem,
    apply_local_defence,
    build_label_packet,
    candidate_payload_sha256,
    clopper_pearson,
    collect_review_items,
    load_gates,
    prepare_blinded_label_packet,
    review_prompt,
    review_response_schema,
    run_review,
    score_review_labels,
)

PAGE_TEXT = (
    "Results\n\n"
    "Our system PerspectiveAPI reaches 0.81 macro F1 on the HateCheck test split.\n"
    "We trained PerspectiveAPI ourselves on the released data.\n"
    "Prior work reported 0.7712 on the same split.\n"
)
RESULT_QUOTE = "PerspectiveAPI reaches 0.81 macro F1 on the HateCheck test split."
ORIGIN_QUOTE = "We trained PerspectiveAPI ourselves on the released data."


def _item(**overrides: Any) -> ReviewItem:
    payload: dict[str, Any] = {
        "paper_id": "review-fixture",
        "observation_id": "obs_fixture_1",
        "page": 3,
        "page_text": PAGE_TEXT,
        "raw_value": "0.81",
        "evidence_quote": RESULT_QUOTE,
        "evaluated_system": "PerspectiveAPI",
        "system_role": "evaluated_system",
        "dataset": "HateCheck",
        "metric": "macro F1",
        "unit": "proportion",
        "scope": "split=test",
        "candidate_payload_sha256": "0" * 64,
    }
    payload.update(overrides)
    return ReviewItem(**payload)


def _response(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "decision": "accept",
        "fields": {
            name: {"verdict": "correct", "page_quote": None}
            for name in (
                "value",
                "metric",
                "unit",
                "system_identity",
                "system_role",
                "dataset",
                "scope",
            )
        },
        "result_quote": RESULT_QUOTE,
        "origin_quote": ORIGIN_QUOTE,
        "rationale": "The page states it directly.",
    }
    payload.update(overrides)
    return payload


def test_an_accept_survives_when_both_quotations_hold() -> None:
    outcome = apply_local_defence(_item(), _response())

    assert outcome.decision == "accept"
    assert outcome.demotions == []


def test_an_accept_whose_origin_quote_is_not_on_the_page_keeps_only_the_tuple() -> None:
    outcome = apply_local_defence(
        _item(), _response(origin_quote="We built this system from scratch in 2019.")
    )

    assert outcome.decision == "accept_tuple_only"
    assert outcome.demotions == ["origin_quote_failed_local_defence"]


def test_an_accept_whose_result_quote_omits_the_value_abstains() -> None:
    outcome = apply_local_defence(
        _item(), _response(result_quote="Prior work reported 0.7712 on the same split.")
    )

    assert outcome.decision == "abstain"
    assert "result_quote_failed_local_defence" in outcome.demotions


def test_an_accept_whose_result_quote_omits_the_system_abstains() -> None:
    outcome = apply_local_defence(_item(evaluated_system="SomeOtherModel"), _response())

    assert outcome.decision == "abstain"


def test_a_reject_is_never_promoted_by_the_defence() -> None:
    outcome = apply_local_defence(_item(), _response(decision="reject", result_quote=None))

    assert outcome.decision == "reject"
    assert outcome.demotions == []


def test_a_value_inside_a_longer_number_does_not_count_as_the_value() -> None:
    """0.81 must not match inside 0.8123."""

    page = "Our system PerspectiveAPI reaches 0.8123 macro F1 here.\n"
    outcome = apply_local_defence(
        _item(page_text=page, raw_value="0.81"),
        _response(result_quote="Our system PerspectiveAPI reaches 0.8123 macro F1 here."),
    )

    assert outcome.decision == "abstain"


def test_the_prompt_never_carries_gate_outcomes_or_census_metadata() -> None:
    item = _item()
    rendered = review_prompt(item)

    for forbidden in (
        "claim_type",
        "primary_result",
        "export_status",
        "export_reason",
        "attribution",
        "needs_review",
        "referential_status",
        "field_provenance",
        "census_",
        "observation_id",
        "obs_fixture_1",
    ):
        assert forbidden not in rendered, forbidden
    # And what it must carry.
    assert item.page_text in rendered
    assert "0.81" in rendered
    assert "HateCheck" in rendered


def _paper(tmp_path: Path, *, slug: str = "review-fixture") -> Path:
    paper_dir = tmp_path / slug
    (paper_dir / "private").mkdir(parents=True, exist_ok=True)
    record = {
        "observation_id": "obs_fixture_1",
        "paper_id": slug,
        "claim_type": "primary_result",
        "text_support": "supported",
        "export_status": "needs_review",
        "export_reason": "referential_status=unresolved",
        "roles": [{"role": "evaluated_system", "raw_name": "PerspectiveAPI"}],
        "scope": {"dataset_raw": "HateCheck", "split": "test"},
        "metric": {"raw_name": "macro F1", "canonical_id": "macro_f1"},
        "value": {"raw": "0.81", "numeric": 0.81, "unit": "proportion"},
        "evidence": [
            {
                "source_id": "src_x",
                "page": 1,
                "kind": "prose",
                "quote": RESULT_QUOTE,
                "quote_sha256": hashlib.sha256(RESULT_QUOTE.encode()).hexdigest(),
            }
        ],
    }
    (paper_dir / "observations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps({"pages": [{"page": 1, "text": PAGE_TEXT}]}), encoding="utf-8"
    )
    return paper_dir


def test_a_completed_review_is_never_repeated(tmp_path: Path) -> None:
    from proceedings_to_eee.extraction.llm_schema import provider_json_schema
    from proceedings_to_eee.extraction.prompt import prompt_hash, row_prompt_hash

    extractor_before = (prompt_hash(), row_prompt_hash(), provider_json_schema())
    paper_dir = _paper(tmp_path)
    calls: list[str] = []

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(user)
        return _response(), {"cost_usd": 0.001}

    first = run_review(run_root=tmp_path, model="fixture/model", call=call)
    second = run_review(run_root=tmp_path, model="fixture/model", call=call)

    assert first["reviewed"] == 1
    assert second["reviewed"] == 0
    assert second["already_gated"] == 1
    assert len(calls) == 1
    gates = load_gates(paper_dir)
    assert gates["obs_fixture_1"]["decision"] == "accept"
    # The private record keeps the quotes; the gate file does not.
    assert "result_quote" not in gates["obs_fixture_1"]
    private = (paper_dir / "private" / "model-review.jsonl").read_text(encoding="utf-8")
    assert RESULT_QUOTE in private
    # Running the separate reviewer must leave the extractor's contract unchanged.
    assert (prompt_hash(), row_prompt_hash(), provider_json_schema()) == extractor_before
    assert review_response_schema() != provider_json_schema()


def test_the_cost_ceiling_stops_the_loop_without_losing_what_it_bought(
    tmp_path: Path,
) -> None:
    _paper(tmp_path, slug="paper-a")
    _paper(tmp_path, slug="paper-b")
    calls: list[str] = []
    budget_hit = {"value": False}

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(user)
        budget_hit["value"] = True
        return _response(), {"cost_usd": 9.0}

    summary = run_review(
        run_root=tmp_path,
        model="fixture/model",
        call=call,
        stop=lambda: budget_hit["value"],
    )

    assert len(calls) == 1
    assert summary["stopped_for_budget"] is True
    assert summary["reviewed"] == 1
    assert load_gates(tmp_path / "paper-a")


def test_the_selection_takes_supported_primary_candidates_that_do_not_already_export(
    tmp_path: Path,
) -> None:
    _paper(tmp_path)
    items = collect_review_items(tmp_path)

    assert len(items) == 1
    assert items[0].observation_id == "obs_fixture_1"
    assert items[0].evaluated_system == "PerspectiveAPI"
    assert items[0].candidate_payload_sha256 == candidate_payload_sha256(
        json.loads((tmp_path / "review-fixture" / "observations.jsonl").read_text())
    )


def test_the_label_packet_is_written_blank_and_scores_with_exact_bounds(
    tmp_path: Path,
) -> None:
    _paper(tmp_path)

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return _response(), {"cost_usd": 0.001}

    run_review(run_root=tmp_path, model="fixture/model", call=call)
    packet = build_label_packet(run_root=tmp_path, size=10, negatives=10, seed="20260905")

    assert len(packet["entries"]) == 1
    assert packet["entries"][0]["label"] is None
    assert packet["entries"][0]["page_text"] == PAGE_TEXT

    labels = {
        "entries": [{"observation_id": "obs_fixture_1", "label": "correct"}],
    }
    scored = score_review_labels(packet=packet, labels=labels)

    accepted = scored["model_accepted_and_human_says_correct"]
    assert (accepted["k"], accepted["n"]) == (1, 1)
    assert accepted["lower"] == pytest.approx(0.025, abs=1e-3)
    assert accepted["upper"] == 1.0


def _scoring_packet() -> dict[str, Any]:
    return {
        "seed": "synthetic-only",
        "entries": [
            {
                "observation_id": f"obs_synthetic_{index}",
                "paper_id": f"synthetic-paper-{index}",
                "page": 1,
                "page_text": "Synthetic model scores 0.81 F1 on the test split.",
                "candidate": {"reported_value": "0.81"},
                "model_decision": decision,
                "model_proposed_decision": decision,
                "label": None,
                "label_note": None,
            }
            for index, decision in enumerate(("accept", "accept_tuple_only", "reject", "abstain"))
        ],
    }


def test_labels_judge_the_candidate_not_agreement_with_the_model_decision() -> None:
    packet = _scoring_packet()
    labels = {
        "entries": [
            {"observation_id": entry["observation_id"], "label": label}
            for entry, label in zip(
                packet["entries"], ("correct", "incorrect", "incorrect", "correct"), strict=True
            )
        ]
    }
    scored = score_review_labels(packet=packet, labels=labels)
    assert scored["completion_status"] == "complete"
    assert scored["labelled"] == 4
    assert scored["unlabelled"] == 0
    assert scored["model_accepted_and_human_says_correct"]["rate"] == 0.5
    assert scored["model_did_not_accept_and_human_says_incorrect"]["rate"] == 0.5


def test_missing_and_undecidable_labels_are_reported_without_becoming_negative_labels() -> None:
    packet = _scoring_packet()
    labels = {
        "entries": [
            {"observation_id": "obs_synthetic_0", "label": "correct"},
            {"observation_id": "obs_synthetic_1", "label": "cannot_tell"},
            {"observation_id": "obs_synthetic_2", "label": None},
        ]
    }
    scored = score_review_labels(packet=packet, labels=labels)
    assert scored["completion_status"] == "partial"
    assert (scored["packet_entries"], scored["labelled"], scored["unlabelled"]) == (4, 2, 2)
    assert scored["undecidable_by_the_labeller"] == 1
    assert scored["model_accepted_and_human_says_correct"]["n"] == 1
    assert scored["model_did_not_accept_and_human_says_incorrect"]["rate"] is None


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([{"observation_id": "unknown", "label": "correct"}], "unknown"),
        ([{"observation_id": "obs_synthetic_0", "label": "yes"}], "invalid label"),
        ([{"observation_id": "obs_synthetic_0", "label": []}], "invalid label"),
        ([{"observation_id": 0, "label": "correct"}], "nonempty string"),
        (["correct"], "must be objects"),
        (
            [
                {"observation_id": "obs_synthetic_0", "label": "correct"},
                {"observation_id": "obs_synthetic_0", "label": "incorrect"},
            ],
            "duplicate",
        ),
        (
            [{"observation_id": "obs_synthetic_0", "paper_id": "wrong", "label": "correct"}],
            "differs",
        ),
    ],
)
def test_scoring_rejects_invalid_labels_instead_of_silently_changing_the_denominator(
    entries: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_review_labels(packet=_scoring_packet(), labels={"entries": entries})


def test_scoring_rejects_an_ambiguous_observation_id_even_across_different_papers() -> None:
    packet = _scoring_packet()
    packet["entries"][1]["observation_id"] = packet["entries"][0]["observation_id"]
    with pytest.raises(ValueError, match="ambiguous"):
        score_review_labels(packet=packet, labels={"entries": []})


def test_packet_builder_rejects_an_id_collision_instead_of_rebinding_another_paper(
    tmp_path: Path,
) -> None:
    _paper(tmp_path, slug="synthetic-a")
    _paper(tmp_path, slug="synthetic-b")
    with pytest.raises(ValueError, match="ambiguous"):
        build_label_packet(run_root=tmp_path, size=2, negatives=0, seed="synthetic")


def test_blinded_view_is_blank_decision_free_reproducible_and_bound_to_original() -> None:
    packet = _scoring_packet()
    snapshot = json.dumps(packet, sort_keys=True)
    blinded = prepare_blinded_label_packet(packet, seed="review-order")
    assert blinded == prepare_blinded_label_packet(packet, seed="review-order")
    assert "model_decision" not in json.dumps(blinded)
    assert "model_proposed_decision" not in json.dumps(blinded)
    assert all(entry["label"] is None for entry in blinded["entries"])
    assert {entry["observation_id"] for entry in blinded["entries"]} == {
        entry["observation_id"] for entry in packet["entries"]
    }
    blinded["entries"][0]["label"] = "correct"
    scored = score_review_labels(packet=packet, labels=blinded)
    assert scored["source_packet_binding_verified"] is True
    assert scored["completion_status"] == "partial"
    assert json.dumps(packet, sort_keys=True) == snapshot
    packet["seed"] = "changed"
    with pytest.raises(ValueError, match="different source packet"):
        score_review_labels(packet=packet, labels=blinded)


def test_blinding_cannot_erase_existing_labels_and_scoring_rejects_changed_candidate() -> None:
    packet = _scoring_packet()
    blinded = prepare_blinded_label_packet(packet, seed="review-order")
    blinded["entries"][0]["candidate"]["reported_value"] = "0.91"
    with pytest.raises(ValueError, match="differs"):
        score_review_labels(packet=packet, labels=blinded)
    assert all(entry["candidate"]["reported_value"] == "0.81" for entry in packet["entries"])
    packet["entries"][0]["label"] = "correct"
    with pytest.raises(ValueError, match="blank original"):
        prepare_blinded_label_packet(packet, seed="review-order")


def test_blind_packet_cli_roundtrip_preserves_original_and_reports_blank_bound_labels(
    tmp_path: Path,
) -> None:
    from proceedings_to_eee.cli import app

    original = tmp_path / "packet.json"
    reviewer = tmp_path / "reviewer.json"
    original.write_text(json.dumps(_scoring_packet()), encoding="utf-8")
    original_bytes = original.read_bytes()
    runner = CliRunner()
    prepared = runner.invoke(
        app,
        ["blind-review-label-packet", str(original), "--output", str(reviewer)],
    )
    assert prepared.exit_code == 0, prepared.output
    assert json.loads(prepared.output)["labels_filled"] == 0
    view = json.loads(reviewer.read_text(encoding="utf-8"))
    assert all(entry["label"] is None for entry in view["entries"])
    assert "model_decision" not in reviewer.read_text(encoding="utf-8")
    assert original.read_bytes() == original_bytes

    scored = runner.invoke(app, ["score-review-labels", str(original), str(reviewer)])
    assert scored.exit_code == 0, scored.output
    summary = json.loads(scored.output)
    assert summary["completion_status"] == "partial"
    assert summary["unlabelled"] == 4
    assert summary["source_packet_binding_verified"] is True


def test_blind_packet_cli_refuses_to_overwrite_an_existing_response(tmp_path: Path) -> None:
    from proceedings_to_eee.cli import app

    original = tmp_path / "packet.json"
    reviewer = tmp_path / "reviewer.json"
    original.write_text(json.dumps(_scoring_packet()), encoding="utf-8")
    existing = b'{"private_response_already_exists": true}\n'
    reviewer.write_bytes(existing)
    result = CliRunner().invoke(
        app,
        ["blind-review-label-packet", str(original), "--output", str(reviewer)],
    )
    assert result.exit_code == 2
    assert "output already exists" in result.output
    assert reviewer.read_bytes() == existing


def test_score_labels_cli_reports_invalid_labels_without_a_traceback_or_score_file(
    tmp_path: Path,
) -> None:
    from proceedings_to_eee.cli import app

    original = tmp_path / "packet.json"
    reviewer = tmp_path / "reviewer.json"
    output = tmp_path / "score.json"
    original.write_text(json.dumps(_scoring_packet()), encoding="utf-8")
    reviewer.write_text(
        json.dumps({"entries": [{"observation_id": "obs_synthetic_0", "label": "yes"}]}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["score-review-labels", str(original), str(reviewer), "--output", str(output)],
    )
    assert result.exit_code == 2
    assert "invalid label" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()


@pytest.mark.parametrize(
    ("successes", "trials", "lower", "upper"),
    [
        (0, 10, 0.0, 0.3085),
        (5, 10, 0.1871, 0.8129),
        (10, 10, 0.6915, 1.0),
        (95, 100, 0.8872, 0.9836),
    ],
)
def test_clopper_pearson_matches_the_published_exact_interval(
    successes: int, trials: int, lower: float, upper: float
) -> None:
    computed_lower, computed_upper = clopper_pearson(successes, trials)

    assert computed_lower == pytest.approx(lower, abs=5e-4)
    assert computed_upper == pytest.approx(upper, abs=5e-4)


def test_a_systematic_failure_stops_the_run_and_says_what_it_was(tmp_path: Path) -> None:
    """A rejected contract or a bad schema fails every call identically.

    The budget ledger fingerprints a request before dispatch, so omitted determinism
    fields can make every call fail. Report the reason and stop repeated failures.
    """

    from proceedings_to_eee.census_review import MAX_CONSECUTIVE_FAILURES

    for index in range(MAX_CONSECUTIVE_FAILURES + 10):
        _paper(tmp_path, slug=f"paper-{index:03d}")
    calls = {"count": 0}

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["count"] += 1
        raise ValueError("provider call telemetry is not request-bound")

    summary = run_review(run_root=tmp_path, model="fixture/model", call=call)

    assert summary["stopped_for_repeated_failures"] is True
    assert calls["count"] == MAX_CONSECUTIVE_FAILURES
    assert summary["failure_classes"] == {"ValueError": MAX_CONSECUTIVE_FAILURES}
    assert summary["reviewed"] == 0


def test_the_review_call_carries_the_determinism_fields_the_ledger_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from proceedings_to_eee import cli
    from proceedings_to_eee.census_review import (
        REVIEW_REASONING_EFFORT,
        REVIEW_SEED,
        REVIEW_TEMPERATURE,
    )

    _paper(tmp_path)
    requests: list[dict[str, Any]] = []

    class RecordingProvider:
        def structured_chat(self, **kwargs: Any) -> None:
            requests.append(kwargs)
            raise RuntimeError("Offline test stops after the budget ledger accepts the request")

    monkeypatch.setattr(cli, "runtime_key", lambda: "fixture-key")
    monkeypatch.setattr(cli, "OpenRouterClient", lambda **kwargs: RecordingProvider())
    result = CliRunner().invoke(
        cli.app, ["review-candidates", str(tmp_path), "--model", "fixture/model"]
    )

    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert requests[0]["temperature"] == REVIEW_TEMPERATURE
    assert requests[0]["reasoning_effort"] == REVIEW_REASONING_EFFORT
    assert requests[0]["seed"] == REVIEW_SEED
