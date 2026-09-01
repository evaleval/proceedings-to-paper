from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

import pytest

from proceedings_to_eee.extraction.llm import (
    enumerate_row_batch,
    enumerate_row_plan,
    extract_row_batch_candidates,
    extractor_request_contract,
    row_extractor_request_contract,
)
from proceedings_to_eee.extraction.llm_schema import (
    WireRowExtraction,
    row_provider_json_schema,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.prompt import (
    prompt_hash,
    row_batch_prompt,
    row_prompt_hash,
)
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.extraction.row_enumeration import (
    RowDisposition,
    RowEnumerationConfig,
    UnbatchableReason,
    build_row_enumeration_plan,
    make_row_batch,
)
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    StructuredResponse,
    structured_request_contract,
)

SYNTHETIC_PAGE = """Table 7. Synthetic evaluation scores for invented systems.

       Engine          X      Y     Z                         U     V    W
       Cedar      .12 .64 .20                              .71 .44 .54
       Juniper    .27 .58 .37                              .76 .31 .44
       Maple      .21 .74 .33                              .82 .61 .70
       Willow     .16 .69 .26                              .79 .52 .63
       Aspen      .25 .62 .36                              .77 .35 .48
"""


def _page(text: str = SYNTHETIC_PAGE, page: int = 7) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_src_paper_{page:04d}",
        source_id="src_paper",
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=0,
        result_signal_score=10.0,
    )


def _plan(
    text: str = SYNTHETIC_PAGE,
    config: RowEnumerationConfig | None = None,
):
    page = _page(text)
    layout = PdfLayout(
        source_id=page.source_id,
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[page],
    )
    blocks = segment_page_result_blocks(page)
    return build_row_enumeration_plan(layout, blocks, config)


def _scope() -> dict[str, Any]:
    return {
        "dataset_raw": "FIXTURE SET",
        "dataset_id": None,
        "dataset_url": None,
        "dataset_version": None,
        "split": None,
        "subset": None,
        "group": None,
        "language": None,
        "sample_count": None,
        "aggregation": None,
        "raw_scope": None,
    }


def _observation(*, row, system: str, value: str, column: str = "F1") -> dict[str, Any]:
    return {
        "claim_type": "primary_result",
        "roles": [
            {
                "role": "evaluated_system",
                "raw_name": system,
                "version": None,
                "provider": None,
                "confidence": 0.99,
            }
        ],
        "scope": _scope(),
        "metric": {
            "raw_name": column,
            "canonical_id": "f1" if column == "F1" else None,
            "kind": None,
            "unit": "proportion",
            "lower_is_better": False,
            "min_score": 0.0,
            "max_score": 1.0,
            "parameters": {},
        },
        "value": {
            "raw": value,
            "numeric": float(value),
            "unit": "proportion",
            "comparator": "exact",
            "uncertainty": None,
        },
        "evidence": [
            {
                "kind": "table",
                "label": "provider label is deterministically replaced",
                "row": "provider row is deterministically replaced",
                "column": column,
                "quote": row.raw_text,
            }
        ],
        "extraction_confidence": 0.99,
        "construct": None,
        "operationalization": None,
        "decision_rule": None,
        "evaluation_date": None,
        "notes": [],
    }


def _disposition(
    row,
    disposition: str,
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "disposition": disposition,
        "observations": observations or [],
        "note": None,
    }


class _ScriptedClient:
    def __init__(
        self,
        scripts: list[dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        self.scripts = list(scripts)
        self.requests: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.requests.append(kwargs)
        script = self.scripts.pop(0)
        payload = script(kwargs) if callable(script) else script
        contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs["seed"],
            require_parameters=kwargs["require_parameters"],
        )
        return StructuredResponse(
            payload=payload,
            call=ProviderCall(
                model_requested=kwargs["model"],
                model_returned=kwargs["model"],
                provider_returned="fixture",
                prompt_sha256=hashlib.sha256(kwargs["user"].encode("utf-8")).hexdigest(),
                response_sha256=hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                temperature=kwargs["temperature"],
                reasoning_effort=kwargs["reasoning_effort"],
                max_tokens=kwargs["max_tokens"],
                seed=kwargs["seed"],
                schema_name=kwargs["schema_name"],
                schema_sha256=contract["schema"]["schema_sha256"],
                latency_seconds=0.01,
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                cost_usd=0.001,
                attempts=1,
            ),
        )


def _all_not_result(kwargs: dict[str, Any]) -> dict[str, Any]:
    row_ids = re.findall(r'"row_id": "(trow_[0-9a-f]+)"', kwargs["user"])
    return {
        "dispositions": [
            {
                "row_id": row_id,
                "disposition": "not_result",
                "observations": [],
                "note": None,
            }
            for row_id in row_ids
        ],
        "warnings": [],
    }


def test_plan_assigns_stable_structural_ids_and_complete_context() -> None:
    first = _plan()
    second = _plan()

    assert [row.row_id for row in first.rows] == [row.row_id for row in second.rows]
    assert first.telemetry.dense_tables == 1
    assert [row.row_label for row in first.rows][:2] == ["Cedar", "Juniper"]
    juniper = next(row for row in first.rows if row.row_label == "Juniper")
    assert juniper.table_label == "Table 7"
    assert juniper.caption and "Synthetic evaluation" in juniper.caption
    assert juniper.headers
    assert [column.raw for column in juniper.headers[-1].columns] == [
        "Engine",
        "X",
        "Y",
        "Z",
        "U",
        "V",
        "W",
    ]
    assert juniper.raw_text.startswith("Juniper    .27 .58 .37")
    assert [value.raw for value in juniper.values] == [
        ".27",
        ".58",
        ".37",
        ".76",
        ".31",
        ".44",
    ]
    assert juniper.span.start_line == juniper.span.end_line == 5
    assert all(cell.span.column_start >= juniper.span.column_start for cell in juniper.raw_cells)
    rendered = row_batch_prompt(
        paper_title="Fixture Paper", paper_id="fixture-paper", batch=first.batches[0]
    )
    assert juniper.row_id in rendered
    assert "evidence_coordinates" in rendered
    assert "value_positions" in rendered


def test_plan_only_uses_shown_non_reference_dense_rows_and_hard_bounds() -> None:
    plan = _plan()
    assert len(plan.rows) == 5
    assert len(plan.batches) == 2
    assert all(len(batch.rows) <= 4 for batch in plan.batches)
    assert all(batch.value_token_count <= 24 for batch in plan.batches)
    assert all(batch.character_count <= 4_000 for batch in plan.batches)
    assert (
        build_row_enumeration_plan(
            PdfLayout(
                source_id="src_paper",
                parser="fixture",
                parser_version="fixture/1",
                page_count=1,
                pages=[_page()],
            ),
            [],
        ).rows
        == []
    )

    references = """References
Table 9. Reported scores.
     Model       F1
     Prior A    0.91
     Prior B    0.88
"""
    assert _plan(references).rows == []


def test_plan_intersects_selected_block_columns_on_parallel_table_pages() -> None:
    text = "\n".join(
        [
            "CohortX       Index      Locked Index"
            "                         Agent              SignalA    Signal-B",
            "TotalSet      0.68             ---"
            "                         Anchor A            0.73        0.69",
            "Batch A       0.72            0.67"
            "                         Tuned Unit          0.78        0.75",
            "Batch B       0.76            0.71"
            "                         Shifted X           0.81        0.83",
            "                  Table 3: Synthetic cohort indicators."
            "                    Table 4: Synthetic engine signals.",
        ]
    )
    page = _page(text, page=3)
    layout = PdfLayout(
        source_id=page.source_id,
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[page],
    )
    left, right = segment_page_result_blocks(page)

    left_plan = build_row_enumeration_plan(layout, [left])
    right_plan = build_row_enumeration_plan(layout, [right])

    assert [row.row_label for row in left_plan.rows] == [
        "TotalSet",
        "Batch A",
        "Batch B",
    ]
    assert [row.row_label for row in right_plan.rows] == [
        "Anchor A",
        "Tuned Unit",
        "Shifted X",
    ]
    assert {row.row_id for row in left_plan.rows}.isdisjoint(row.row_id for row in right_plan.rows)


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            RowEnumerationConfig(max_value_tokens_per_batch=5),
            UnbatchableReason.VALUE_TOKEN_LIMIT,
        ),
        (
            RowEnumerationConfig(max_characters_per_batch=128),
            UnbatchableReason.CHARACTER_LIMIT,
        ),
    ],
)
def test_single_rows_over_a_hard_limit_are_retained_but_never_sent(
    config: RowEnumerationConfig,
    reason: UnbatchableReason,
) -> None:
    plan = _plan(config=config)
    client = _ScriptedClient([])

    outcome = enumerate_row_plan(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        plan=plan,
    )

    assert plan.rows
    assert plan.batches == []
    assert plan.telemetry.unbatchable_rows == len(plan.rows)
    assert {item.row_id for item in plan.unbatchable_rows} == {row.row_id for row in plan.rows}
    assert all(reason in item.reasons for item in plan.unbatchable_rows)
    assert outcome.unbatchable_row_ids == [row.row_id for row in plan.unbatchable_rows]
    assert outcome.calls == outcome.attempts == []
    assert client.requests == []


def test_row_schema_and_prompt_are_independent_from_legacy_contract() -> None:
    schema = row_provider_json_schema()
    legacy_contract = extractor_request_contract()
    row_contract = row_extractor_request_contract()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"dispositions", "warnings"}
    assert WireRowExtraction.model_config["extra"] == "forbid"
    assert prompt_hash() == "ddbc62294a689f3637dcc2f87156e8b48dd32c9ee1833a5ca27dd033e5aff6de"
    assert legacy_contract["schema"]["schema_sha256"] == (
        "25f45c3acb53c5dbae4861e401a12256171a53fa568fcb826dbbc85deebcbcaf"
    )
    assert row_prompt_hash() != prompt_hash()
    assert row_contract["schema"]["schema_name"] == "paper_table_row_dispositions"
    assert row_contract["schema"]["schema_sha256"] != legacy_contract["schema"]["schema_sha256"]


def test_second_row_survives_with_not_result_sibling() -> None:
    rows = _plan().rows[:2]
    cedar, juniper = rows
    batch = make_row_batch(rows)
    client = _ScriptedClient(
        [
            {
                "dispositions": [
                    _disposition(cedar, "not_result"),
                    _disposition(
                        juniper,
                        "result",
                        [_observation(row=juniper, system="Juniper", value=".37")],
                    ),
                ],
                "warnings": [],
            }
        ]
    )

    outcome = enumerate_row_batch(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=batch,
    )

    assert outcome.unresolved_row_ids == []
    assert outcome.records[cedar.row_id].disposition is RowDisposition.NOT_RESULT
    assert outcome.records[cedar.row_id].candidates == []
    candidate = outcome.records[juniper.row_id].candidates[0]
    assert candidate.roles[0].raw_name == "Juniper"
    assert candidate.value and candidate.value.raw == ".37"
    assert candidate.evidence[0].label == "Table 7"
    assert candidate.evidence[0].row == "Juniper"


@pytest.mark.parametrize("disposition", ["not_result", "uncertain"])
def test_abstention_dispositions_legitimately_produce_zero_candidates(disposition: str) -> None:
    row = _plan().rows[1]
    batch = make_row_batch([row])
    client = _ScriptedClient([{"dispositions": [_disposition(row, disposition)], "warnings": []}])

    outcome = enumerate_row_batch(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=batch,
    )

    assert outcome.candidates == []
    assert outcome.unresolved_row_ids == []
    assert outcome.records[row.row_id].disposition.value == disposition
    assert len(client.requests) == 1


def test_missing_row_retries_only_that_row_and_retains_valid_sibling() -> None:
    cedar, juniper = _plan().rows[:2]
    batch = make_row_batch([cedar, juniper])
    client = _ScriptedClient(
        [
            {"dispositions": [_disposition(cedar, "not_result")], "warnings": []},
            {
                "dispositions": [
                    _disposition(
                        juniper,
                        "result",
                        [_observation(row=juniper, system="Juniper", value=".37")],
                    )
                ],
                "warnings": [],
            },
        ]
    )

    outcome = enumerate_row_batch(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=batch,
    )

    assert outcome.unresolved_row_ids == []
    assert len(client.requests) == 2
    assert juniper.row_id in client.requests[1]["user"]
    assert cedar.row_id not in client.requests[1]["user"]
    assert outcome.telemetry["recovery_calls"] == 1


def test_duplicate_and_unknown_are_detected_while_valid_row_is_retained() -> None:
    cedar, juniper = _plan().rows[:2]
    batch = make_row_batch([cedar, juniper])
    duplicate = _disposition(cedar, "not_result")
    client = _ScriptedClient(
        [
            {
                "dispositions": [
                    duplicate,
                    duplicate,
                    _disposition(juniper, "uncertain"),
                    {
                        "row_id": "trow_deadbeef",
                        "disposition": "not_result",
                        "observations": [],
                        "note": None,
                    },
                ],
                "warnings": [],
            },
            {"dispositions": [_disposition(cedar, "not_result")], "warnings": []},
        ]
    )

    outcome = enumerate_row_batch(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=batch,
    )

    assert outcome.unresolved_row_ids == []
    assert juniper.row_id in outcome.records
    assert outcome.unknown_row_ids == ["trow_deadbeef"]
    assert outcome.invalid_row_reasons[cedar.row_id] == "duplicate_disposition"
    assert len(client.requests) == 2


def test_out_of_row_evidence_is_invalid_and_can_abstain_on_recovery() -> None:
    cedar, juniper = _plan().rows[:2]
    wrong = _observation(row=juniper, system="Juniper", value=".37")
    wrong["evidence"][0]["quote"] = "Another system 0.37"
    client = _ScriptedClient(
        [
            {
                "dispositions": [
                    _disposition(cedar, "not_result"),
                    _disposition(juniper, "result", [wrong]),
                ],
                "warnings": [],
            },
            {"dispositions": [_disposition(juniper, "uncertain")], "warnings": []},
        ]
    )

    outcome = enumerate_row_batch(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=make_row_batch([cedar, juniper]),
    )

    assert outcome.unresolved_row_ids == []
    assert outcome.records[juniper.row_id].disposition is RowDisposition.UNCERTAIN
    assert outcome.invalid_row_reasons[juniper.row_id] == "observation_not_bound_to_row"


def test_row_value_binding_requires_an_exact_structural_token() -> None:
    juniper = _plan().rows[1]
    deceptive = _observation(row=juniper, system="Juniper", value=".3")
    result = extract_row_batch_candidates(
        client=_ScriptedClient(
            [
                {
                    "dispositions": [
                        _disposition(juniper, "result", [deceptive]),
                    ],
                    "warnings": [],
                }
            ]
        ),  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=make_row_batch([juniper]),
    )

    assert result.records == {}
    assert result.invalid_row_reasons == {juniper.row_id: "observation_not_bound_to_row"}


def test_single_row_failure_is_not_retried_identically() -> None:
    row = _plan().rows[1]
    client = _ScriptedClient([{"dispositions": [], "warnings": []}])

    outcome = enumerate_row_batch(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=make_row_batch([row]),
    )

    assert outcome.unresolved_row_ids == [row.row_id]
    assert len(client.requests) == 1


def test_every_base_batch_has_a_three_call_hard_bound() -> None:
    batch = make_row_batch(_plan().rows[:4])
    client = _ScriptedClient(
        [
            {"dispositions": [], "warnings": []},
            _all_not_result,
            _all_not_result,
        ]
    )

    outcome = enumerate_row_batch(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=batch,
    )

    assert outcome.unresolved_row_ids == []
    assert len(outcome.calls) == len(outcome.attempts) == 3
    assert outcome.telemetry["base_calls"] == 1
    assert outcome.telemetry["recovery_calls"] == 2


def test_result_without_candidates_and_abstention_with_candidates_are_invalid() -> None:
    cedar, juniper = _plan().rows[:2]
    payload = {
        "dispositions": [
            _disposition(cedar, "result"),
            _disposition(
                juniper,
                "not_result",
                [_observation(row=juniper, system="Juniper", value=".37")],
            ),
        ],
        "warnings": [],
    }
    result = extract_row_batch_candidates(
        client=_ScriptedClient([payload]),  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        batch=make_row_batch([cedar, juniper]),
    )

    assert result.records == {}
    assert result.invalid_row_reasons == {
        cedar.row_id: "result_without_observation",
        juniper.row_id: "abstention_with_observation",
    }
