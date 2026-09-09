"""Candidate auditor: page-scoped judging, quotation defence, sampling, and framing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from proceedings_to_eee.census_audit import (
    AUDITED_FIELDS,
    AuditItem,
    audit_prompt,
    audit_response_schema,
    collect_audit_items,
    quotation_is_on_page,
    run_audit,
    stratified_sample,
    summarize,
)

PAGE = """Table 3: Results on the HateCheck benchmark.

System            Accuracy   Macro F1
Perspective API      0.812      0.774
Our model            0.867      0.841

We report macro F1 across all functional tests.
"""


def _paper(
    root: Path,
    slug: str,
    *,
    candidates: list[dict[str, Any]],
    page_text: str = PAGE,
) -> None:
    directory = root / slug
    (directory / "private").mkdir(parents=True, exist_ok=True)
    (directory / "private" / "layout.json").write_text(
        json.dumps({"pages": [{"page": 7, "text": page_text}]}), encoding="utf-8"
    )
    (directory / "observations.jsonl").write_text(
        "\n".join(json.dumps(candidate) for candidate in candidates) + "\n", encoding="utf-8"
    )


def _candidate(
    observation_id: str,
    *,
    supported: str = "supported",
    value: str = "0.812",
    system: str = "Perspective API",
    page: int = 7,
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "text_support": supported,
        "claim_type": "primary_result",
        "value": {"raw": value, "unit": "proportion"},
        "roles": [{"role": "evaluated_system", "raw_name": system}],
        "scope": {"dataset_raw": "HateCheck"},
        "metric": {"raw_name": "Accuracy"},
        "evidence": [{"page": page, "quote": f"{system}      {value}"}],
    }


def _item(**overrides: Any) -> AuditItem:
    base: dict[str, Any] = {
        "paper_id": "paper-a",
        "observation_id": "obs-1",
        "page": 7,
        "page_text": PAGE,
        "raw_value": "0.812",
        "evidence_quote": "Perspective API      0.812",
        "system": "Perspective API",
        "dataset": "HateCheck",
        "metric": "Accuracy",
        "unit": "proportion",
        "claim_type": "primary_result",
        "stratum": ("2024", "acl", "builds_own_instrument"),
    }
    base.update(overrides)
    return AuditItem(**base)


def _answer(verdict: str, quote: str | None = None, reason: str | None = None) -> dict[str, Any]:
    return {"verdict": verdict, "reason": reason, "page_quote": quote}


def _all(verdict: str, quote: str | None = None) -> dict[str, Any]:
    return {field: _answer(verdict, quote) for field in AUDITED_FIELDS}


def test_schema_requires_every_audited_field() -> None:
    schema = audit_response_schema()
    assert set(schema["required"]) == set(AUDITED_FIELDS)
    assert schema["additionalProperties"] is False


def test_only_supported_candidates_with_a_real_page_are_collected(tmp_path: Path) -> None:
    _paper(
        tmp_path,
        "paper-a",
        candidates=[
            _candidate("obs-1"),
            _candidate("obs-2", supported="unsupported"),
            _candidate("obs-3", page=99),
        ],
    )

    items = collect_audit_items(tmp_path)

    assert [item.observation_id for item in items] == ["obs-1"]
    assert items[0].page_text.startswith("Table 3")


def test_the_prompt_carries_the_whole_page_and_no_gate_outcome(tmp_path: Path) -> None:
    prompt = audit_prompt(_item())

    # The page, not the extracted block, is what the auditor judges against.
    assert "Our model            0.867" in prompt
    assert "We report macro F1" in prompt
    for leaked in (
        "text_support",
        "supported",
        "export_status",
        "needs_review",
        "census_",
        "measurement_role",
        "reference",
    ):
        assert leaked not in prompt


def test_a_verdict_whose_quotation_is_absent_from_the_page_is_dropped(tmp_path: Path) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate("obs-1")])
    calls: list[str] = []

    def fabricating_call(system: str, user: str) -> dict[str, Any]:
        calls.append(user)
        return _all("incorrect", quote="a sentence that never appears on the page")

    summary = run_audit(
        run_root=tmp_path,
        output=tmp_path / "audit.json",
        model="test/model",
        sample=1,
        seed="s",
        call=fabricating_call,
    )

    assert summary["verdicts_kept"] == 0
    assert summary["verdicts_dropped_unquotable"] == len(AUDITED_FIELDS)
    assert len(calls) == 1


def test_a_correct_verdict_needs_no_quotation(tmp_path: Path) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate("obs-1")])

    summary = run_audit(
        run_root=tmp_path,
        output=tmp_path / "audit.json",
        model="test/model",
        sample=1,
        seed="s",
        call=lambda system, user: _all("correct"),
    )

    assert summary["verdicts_kept"] == len(AUDITED_FIELDS)
    assert summary["per_field"]["value"]["correct_rate_over_decided"] == 1.0


def test_a_quoted_defect_is_kept(tmp_path: Path) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate("obs-1")])

    def grounded(system: str, user: str) -> dict[str, Any]:
        answers = _all("correct")
        answers["dataset"] = _answer(
            "incorrect", quote="Results on the HateCheck benchmark", reason="different table"
        )
        return answers

    summary = run_audit(
        run_root=tmp_path,
        output=tmp_path / "audit.json",
        model="test/model",
        sample=1,
        seed="s",
        call=grounded,
    )

    assert summary["verdicts_dropped_unquotable"] == 0
    assert summary["per_field"]["dataset"]["counts"]["incorrect"] == 1
    assert summary["per_field"]["dataset"]["correct_rate_over_decided"] == 0.0


def test_cannot_tell_is_excluded_from_the_correct_rate_denominator() -> None:
    rows = [
        {
            "field": "value",
            "verdict": "correct",
            "stratum_role": "r",
            "paper_id": "p",
            "observation_id": "o1",
            "reason": None,
            "stratum_year": "2024",
            "stratum_venue": "acl",
        },
        {
            "field": "value",
            "verdict": "cannot_tell",
            "stratum_role": "r",
            "paper_id": "p",
            "observation_id": "o2",
            "reason": None,
            "stratum_year": "2024",
            "stratum_venue": "acl",
        },
    ]
    summary = summarize(rows, audited=2, dropped=0)

    field = summary["per_field"]["value"]
    assert field["denominator_all"] == 2
    assert field["denominator_decided"] == 1
    assert field["correct_rate_over_decided"] == 1.0
    assert field["cannot_tell_rate"] == 0.5


def test_sampling_is_deterministic_and_spread_across_strata() -> None:
    items = [
        _item(
            observation_id=f"obs-{index}",
            stratum=(str(2020 + index % 5), "acl", f"role-{index % 3}"),
        )
        for index in range(60)
    ]

    first = [item.observation_id for item in stratified_sample(items, seed="s", size=15)]
    again = [item.observation_id for item in stratified_sample(items, seed="s", size=15)]
    other = [item.observation_id for item in stratified_sample(items, seed="t", size=15)]

    assert first == again
    assert first != other
    roles = {item.stratum[2] for item in items if item.observation_id in set(first)}
    assert len(roles) == 3


def test_a_failing_audit_call_does_not_end_the_audit(tmp_path: Path) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate("obs-1"), _candidate("obs-2")])
    seen: list[int] = []

    def flaky(system: str, user: str) -> dict[str, Any]:
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("provider said no")
        return _all("correct")

    summary = run_audit(
        run_root=tmp_path,
        output=tmp_path / "audit.json",
        model="test/model",
        sample=2,
        seed="s",
        call=flaky,
    )

    assert summary["sampling_frame"]["audit_calls_failed"] == 1
    assert summary["candidates_audited"] == 1


def test_the_artifact_states_it_is_not_human_validation(tmp_path: Path) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate("obs-1")])

    run_audit(
        run_root=tmp_path,
        output=tmp_path / "audit.json",
        model="test/model",
        sample=1,
        seed="s",
        call=lambda system, user: _all("correct"),
    )
    written = json.loads((tmp_path / "audit.json").read_text())

    assert written["review_type"] == "ai_analyst_qa"
    assert written["independent_human_validation"] is False
    assert "not precision, recall" in written["claim_boundary"].lower()


def test_quotation_check_normalizes_whitespace_only() -> None:
    assert quotation_is_on_page("Perspective API      0.812", PAGE)
    assert quotation_is_on_page("Perspective API 0.812", PAGE)
    assert not quotation_is_on_page("Perspective API 0.813", PAGE)
    assert not quotation_is_on_page("", PAGE)
    assert not quotation_is_on_page(None, PAGE)


def test_the_audit_prompt_never_tells_the_auditor_what_the_extractor_concluded() -> None:
    """Keep the extractor's claim_type out of the audit prompt.

    It is the extractor's own label, and showing it to an auditor invites agreement
    rather than an independent judgement of the supplied evidence.
    """

    from proceedings_to_eee.census_audit import AuditItem, audit_prompt

    item = AuditItem(
        paper_id="p",
        observation_id="obs_1",
        page=3,
        page_text="Our system reaches 0.81 macro F1 on HateCheck.",
        raw_value="0.81",
        evidence_quote="Our system reaches 0.81 macro F1 on HateCheck.",
        system="OurSystem",
        dataset="HateCheck",
        metric="macro F1",
        unit="proportion",
        claim_type="primary_result",
        stratum=("2024", "acl", "applies_existing_instrument"),
    )

    rendered = audit_prompt(item)

    assert "primary_result" not in rendered
    assert "claim_type" not in rendered
    assert "claimed_result_type" not in rendered
    # What it must still carry.
    assert item.page_text in rendered
    assert "0.81" in rendered
    assert "HateCheck" in rendered
