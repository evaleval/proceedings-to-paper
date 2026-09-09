from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    structured_request_contract,
)
from proceedings_to_eee.resolution.origin_retrieval import (
    EvidenceRelation,
    MatchDimension,
    OriginRetrievalContract,
    OriginRoute,
    ProducerOriginProposal,
    ProducerOriginWireProposal,
    candidate_origin_binding_sha256,
    evidence_anchor_from_hit,
    layout_binding_sha256,
    materialize_producer_origin_wire_proposal,
    producer_origin_prompt,
    producer_origin_provider_json_schema,
    producer_origin_request_fingerprint,
    producer_origin_wire_response_sha256,
    propose_producer_origin,
    retrieve_origin_context,
    verify_producer_origin_proposal,
)

METHODS_TEXT = """2 Experimental Setup
We evaluate Paper System on Synthetic Benchmark using ROC-AUC.
All runs use the same fixed split and deterministic seed.
"""

GENERIC_METHODS_TEXT = """2 Experimental Setup
We evaluate all candidate systems on Synthetic Benchmark using ROC-AUC.
All runs use the same fixed split and deterministic seed.
"""

EXTERNAL_TEXT = """2 Experimental Setup
Scores for Paper System were taken from the public leaderboard.
The table below preserves the published values without rerunning the systems.
"""

RESULTS_TEXT = """3 Results
Table 2: Performance on Synthetic Benchmark.
System                         ROC-AUC
Paper System                   0.742
"""


def _page(text: str, page: int) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_fixture_{page:04d}",
        source_id="src_fixture",
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=1,
        result_signal_score=1.0,
    )


def _layout(first_page: str = METHODS_TEXT) -> PdfLayout:
    return PdfLayout(
        source_id="src_fixture",
        parser="poppler-pdftotext-layout",
        parser_version="fixture-1.0",
        page_count=2,
        pages=[_page(first_page, 1), _page(RESULTS_TEXT, 2)],
    )


def _candidate(*, attribution: AttributionVerdict | None = None) -> CandidateObservation:
    return CandidateObservation.model_validate(
        {
            "paper_id": "fixture-paper",
            "claim_type": "primary_result",
            "roles": [
                {
                    "role": "evaluated_system",
                    "raw_name": "Paper System",
                    "confidence": 0.95,
                }
            ],
            "scope": {"dataset_raw": "Synthetic Benchmark", "split": "test"},
            "metric": {
                "raw_name": "ROC-AUC",
                "canonical_id": "auroc",
                "unit": "proportion",
            },
            "value": {"raw": "0.742", "numeric": 0.742, "unit": "proportion"},
            "evidence": [
                {
                    "source_id": "src_fixture",
                    "page": 2,
                    "kind": "table",
                    "label": "Table 2",
                    "row": "Paper System",
                    "column": "ROC-AUC",
                    "quote": "Paper System                   0.742",
                }
            ],
            "extraction_method": "fixture",
            "extraction_confidence": 0.95,
            "attribution": attribution.model_dump(mode="json") if attribution else None,
        }
    )


def _hit(bundle: Any, *, page: int, result: bool = False):
    matches = [
        item
        for item in bundle.hits
        if item.page == page
        and ((MatchDimension.RESULT_ANCHOR in item.matched_dimensions) is result)
    ]
    assert matches
    return matches[0]


def _proposal(
    candidate: CandidateObservation,
    bundle: Any,
    *,
    state: AttributionState,
    origin_page: int | None,
    relation: EvidenceRelation,
) -> ProducerOriginProposal:
    result = evidence_anchor_from_hit(bundle, _hit(bundle, page=2, result=True).context_hit_id)
    origin = None
    if origin_page is not None:
        origin = evidence_anchor_from_hit(
            bundle,
            _hit(bundle, page=origin_page, result=False).context_hit_id,
        )
    return ProducerOriginProposal(
        candidate_binding_sha256=candidate_origin_binding_sha256(candidate),
        evaluated_system="Paper System",
        proposed_state=state,
        evidence_relation=relation,
        result_anchor=result,
        origin_anchor=origin,
        summary="The selected excerpt bears on the origin of this exact result.",
    )


def test_whole_paper_retrieval_finds_methods_and_result_pages_with_exact_spans() -> None:
    candidate = _candidate()
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)

    assert {1, 2}.issubset({hit.page for hit in bundle.hits})
    methods = _hit(bundle, page=1, result=False)
    result = _hit(bundle, page=2, result=True)
    assert {
        MatchDimension.EVALUATED_SYSTEM,
        MatchDimension.DATASET,
        MatchDimension.METRIC,
    }.issubset(set(methods.matched_dimensions))
    assert result.context_kind == "result"
    assert len(bundle.hits) <= bundle.retrieval_contract.max_hits
    for hit in bundle.hits:
        page = next(item for item in layout.pages if item.page == hit.page)
        assert page.text[hit.char_start : hit.char_end] == hit.exact_excerpt
        assert hashlib.sha256(hit.exact_excerpt.encode()).hexdigest() == hit.excerpt_sha256
        assert hit.page_text_sha256 == page.text_sha256


def test_retrieval_and_provider_request_hashes_are_deterministic_and_checkpoint_friendly() -> None:
    candidate = _candidate()
    layout = _layout()
    first = retrieve_origin_context(candidate, layout)
    second = retrieve_origin_context(candidate, layout)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert layout_binding_sha256(layout) == first.layout_sha256
    request = producer_origin_request_fingerprint(
        model="fixture/model",
        candidate=candidate,
        bundle=first,
    )
    assert request == producer_origin_request_fingerprint(
        model="fixture/model",
        candidate=candidate,
        bundle=second,
    )

    bounded = retrieve_origin_context(
        candidate,
        layout,
        contract=OriginRetrievalContract(max_hits=1),
    )
    assert len(bounded.hits) == 1
    assert bounded.checkpoint_sha256 != first.checkpoint_sha256


def test_layout_integrity_and_unique_declared_pages_are_required() -> None:
    candidate = _candidate()
    page = _page(METHODS_TEXT, 1)
    duplicate = PdfLayout(
        source_id="src_fixture",
        parser="poppler-pdftotext-layout",
        parser_version="fixture-1.0",
        page_count=2,
        pages=[page, page.model_copy(deep=True)],
    )
    with pytest.raises(ValueError, match="page numbers must be unique"):
        retrieve_origin_context(candidate, duplicate)

    corrupt_page = page.model_copy(update={"text_sha256": "0" * 64})
    corrupt = duplicate.model_copy(update={"page_count": 1, "pages": [corrupt_page]})
    with pytest.raises(ValueError, match="text_sha256"):
        layout_binding_sha256(corrupt)


def test_strict_provider_schema_selects_separate_frozen_result_and_origin_hits() -> None:
    schema = producer_origin_provider_json_schema()
    assert schema["additionalProperties"] is False
    assert "schema_version" not in schema["properties"]
    assert "schema_version" not in schema["required"]
    assert "schema_version" not in json.dumps(schema, sort_keys=True)
    assert {"result_context_hit_id", "origin_context_hit_id"}.issubset(schema["properties"])
    assert {"result_context_hit_id", "origin_context_hit_id"}.issubset(schema["required"])
    assert "result_anchor" not in schema["properties"]
    assert "exact_excerpt" not in json.dumps(schema)
    with pytest.raises(ValidationError):
        ProducerOriginWireProposal.model_validate(
            {
                "schema_version": "producer-origin-wire-proposal/0.2",
                "candidate_binding_sha256": "a" * 64,
                "evaluated_system": "Paper System",
                "proposed_state": "paper_produced",
                "evidence_relation": "direct",
                "result_context_hit_id": "origin_ctx_" + "b" * 24,
                "origin_context_hit_id": None,
                "summary": "unsupported",
                "unexpected": True,
            }
        )


def test_two_page_positive_origin_is_verified_but_never_automatically_promoted() -> None:
    candidate = _candidate()
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.PAPER_PRODUCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )

    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert assessment.result_anchor_verified
    assert assessment.origin_anchor_verified
    assert assessment.positive_evidence_verified
    assert assessment.proposed_state is AttributionState.PAPER_PRODUCED
    assert assessment.effective_state is AttributionState.UNRESOLVED
    assert assessment.route is OriginRoute.REVIEW
    assert "positive_origin_evidence_review_only" in assessment.reason_codes
    assert not assessment.allows_automatic_export


def test_generic_we_evaluate_without_the_system_is_a_weak_link() -> None:
    candidate = _candidate()
    layout = _layout(GENERIC_METHODS_TEXT)
    bundle = retrieve_origin_context(candidate, layout)
    generic = _hit(bundle, page=1, result=False)
    assert "We evaluate all candidate systems" in generic.exact_excerpt
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.PAPER_PRODUCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )

    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert assessment.origin_anchor_verified
    assert not assessment.positive_evidence_verified
    assert assessment.effective_state is AttributionState.UNRESOLVED
    assert "origin_link_weak" in assessment.reason_codes


@pytest.mark.parametrize("corruption", ["page", "excerpt", "hash"])
def test_wrong_page_excerpt_or_hash_is_rejected_locally(corruption: str) -> None:
    candidate = _candidate()
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.PAPER_PRODUCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )
    assert proposal.origin_anchor is not None
    if corruption == "page":
        bad_anchor = proposal.origin_anchor.model_copy(update={"page": 99})
    elif corruption == "excerpt":
        text = proposal.origin_anchor.exact_excerpt + "invented"
        bad_anchor = proposal.origin_anchor.model_copy(
            update={
                "exact_excerpt": text,
                "excerpt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    else:
        bad_anchor = proposal.origin_anchor.model_copy(update={"excerpt_sha256": "0" * 64})
    bad = proposal.model_copy(update={"origin_anchor": bad_anchor})

    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=bad,
    )

    assert not assessment.origin_anchor_verified
    assert assessment.effective_state is AttributionState.UNRESOLVED
    assert "origin_proposal_invalid" in assessment.reason_codes
    assert not assessment.allows_automatic_export


def test_candidate_specific_external_evidence_demotes() -> None:
    candidate = _candidate()
    layout = _layout(EXTERNAL_TEXT)
    bundle = retrieve_origin_context(candidate, layout)
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.EXTERNALLY_SOURCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )

    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert assessment.origin_anchor_verified
    assert assessment.effective_state is AttributionState.EXTERNALLY_SOURCED
    assert assessment.route is OriginRoute.DEMOTE
    assert "origin_external_evidence" in assessment.reason_codes
    assert not assessment.allows_automatic_export


def test_external_evidence_contradicting_a_positive_proposal_still_demotes() -> None:
    candidate = _candidate()
    layout = _layout(EXTERNAL_TEXT)
    bundle = retrieve_origin_context(candidate, layout)
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.PAPER_PRODUCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )

    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert assessment.effective_state is AttributionState.EXTERNALLY_SOURCED
    assert assessment.route is OriginRoute.DEMOTE
    assert "origin_proposal_contradicted_by_external_evidence" in assessment.reason_codes


def test_no_signal_is_preserved_as_no_signal_not_positive_origin() -> None:
    candidate = _candidate()
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.NO_SIGNAL,
        origin_page=None,
        relation=EvidenceRelation.NONE,
    )

    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert assessment.result_anchor_verified
    assert not assessment.origin_anchor_verified
    assert assessment.effective_state is AttributionState.NO_SIGNAL
    assert assessment.route is OriginRoute.REVIEW
    assert "origin_no_signal" in assessment.reason_codes


def test_existing_unresolved_is_not_downgraded_to_no_signal() -> None:
    candidate = _candidate(
        attribution=AttributionVerdict(
            state=AttributionState.UNRESOLVED,
            rule_id="weak_cue_recorded",
        )
    )
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.NO_SIGNAL,
        origin_page=None,
        relation=EvidenceRelation.NONE,
    )
    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert assessment.effective_state is AttributionState.UNRESOLVED
    assert "existing_unresolved_preserved" in assessment.reason_codes


def test_deterministic_external_cue_cannot_be_overridden_by_positive_model_proposal() -> None:
    candidate = _candidate(
        attribution=AttributionVerdict(
            state=AttributionState.EXTERNALLY_SOURCED,
            rule_id="row_scoped_foreign_cue",
        )
    )
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)
    proposal = _proposal(
        candidate,
        bundle,
        state=AttributionState.PAPER_PRODUCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )
    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert assessment.deterministic_external_preserved
    assert assessment.effective_state is AttributionState.EXTERNALLY_SOURCED
    assert assessment.route is OriginRoute.DEMOTE
    assert "deterministic_external_preserved" in assessment.reason_codes
    assert not assessment.allows_automatic_export


class _FixtureClient:
    def __init__(
        self,
        proposal: ProducerOriginProposal,
        *,
        wire_overrides: dict[str, Any] | None = None,
        wrong_response_hash: bool = False,
    ) -> None:
        self.proposal = proposal
        self.wire_overrides = wire_overrides or {}
        self.wrong_response_hash = wrong_response_hash
        self.kwargs: dict[str, Any] | None = None
        self.call: ProviderCall | None = None

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.kwargs = kwargs
        contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs["seed"],
            require_parameters=kwargs["require_parameters"],
        )
        schema_contract = contract["schema"]
        payload = {
            "candidate_binding_sha256": self.proposal.candidate_binding_sha256,
            "evaluated_system": self.proposal.evaluated_system,
            "proposed_state": self.proposal.proposed_state.value,
            "evidence_relation": self.proposal.evidence_relation.value,
            "result_context_hit_id": self.proposal.result_anchor.context_hit_id,
            "origin_context_hit_id": (
                self.proposal.origin_anchor.context_hit_id
                if self.proposal.origin_anchor is not None
                else None
            ),
            "summary": self.proposal.summary,
        }
        payload.update(self.wire_overrides)
        response_bytes = json.dumps(payload, sort_keys=True).encode()
        response_sha256 = hashlib.sha256(response_bytes).hexdigest()
        if self.wrong_response_hash:
            response_sha256 = "0" * 64
        self.call = ProviderCall(
            model_requested=kwargs["model"],
            model_returned="fixture/model",
            provider_returned="fixture-provider",
            prompt_sha256="a" * 64,
            response_sha256=response_sha256,
            temperature=kwargs["temperature"],
            reasoning_effort=kwargs["reasoning_effort"],
            max_tokens=kwargs["max_tokens"],
            completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
            seed=contract["seed"],
            response_format=schema_contract["response_format"],
            schema_name=schema_contract["schema_name"],
            schema_sha256=schema_contract["schema_sha256"],
            schema_strict=schema_contract["schema_strict"],
            latency_seconds=0.01,
            attempts=1,
        )
        return StructuredResponse(payload=payload, call=self.call)


def test_provider_function_uses_shared_structured_chat_without_network() -> None:
    candidate = _candidate()
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)
    expected = _proposal(
        candidate,
        bundle,
        state=AttributionState.PAPER_PRODUCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )
    client = _FixtureClient(expected)

    actual, call = propose_producer_origin(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        candidate=candidate,
        bundle=bundle,
    )

    assert actual == expected
    assert call is client.call
    assert client.kwargs is not None
    assert client.kwargs["temperature"] is None
    assert client.kwargs["seed"] is None
    assert client.kwargs["require_parameters"] is True
    assert "<ORIGIN_INPUT>" in client.kwargs["user"]
    assert "Omit schema_version from the response" in client.kwargs["user"]
    assert producer_origin_prompt(candidate, bundle) == client.kwargs["user"]
    assert call.response_sha256 == producer_origin_wire_response_sha256(actual)

    provider_payload = {
        "candidate_binding_sha256": expected.candidate_binding_sha256,
        "evaluated_system": expected.evaluated_system,
        "proposed_state": expected.proposed_state.value,
        "evidence_relation": expected.evidence_relation.value,
        "result_context_hit_id": expected.result_anchor.context_hit_id,
        "origin_context_hit_id": (
            expected.origin_anchor.context_hit_id if expected.origin_anchor is not None else None
        ),
        "summary": expected.summary,
    }
    wire = materialize_producer_origin_wire_proposal(provider_payload)
    assert wire.schema_version == "producer-origin-wire-proposal/0.2"
    assert (
        producer_origin_wire_response_sha256(wire)
        == hashlib.sha256(
            json.dumps(provider_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="must omit schema_version"):
        materialize_producer_origin_wire_proposal(
            provider_payload | {"schema_version": "provider-owned-version"}
        )

    with pytest.raises(ProviderResponseValidationError, match="WireExtraction validation"):
        propose_producer_origin(
            client=_FixtureClient(expected, wrong_response_hash=True),  # type: ignore[arg-type]
            model="fixture/model",
            candidate=candidate,
            bundle=bundle,
        )


def test_provider_hit_selection_is_materialized_locally_and_unknown_hits_fail_closed() -> None:
    candidate = _candidate()
    layout = _layout()
    bundle = retrieve_origin_context(candidate, layout)
    expected = _proposal(
        candidate,
        bundle,
        state=AttributionState.PAPER_PRODUCED,
        origin_page=1,
        relation=EvidenceRelation.DIRECT,
    )
    client = _FixtureClient(
        expected,
        wire_overrides={"origin_context_hit_id": "origin_ctx_" + "f" * 24},
    )

    with pytest.raises(ProviderResponseValidationError) as captured:
        propose_producer_origin(
            client=client,  # type: ignore[arg-type]
            model="fixture/model",
            candidate=candidate,
            bundle=bundle,
        )

    assert captured.value.code == "wire_validation"


@pytest.mark.parametrize(
    "state",
    [
        AttributionState.PAPER_PRODUCED,
        AttributionState.EXTERNALLY_SOURCED,
        AttributionState.UNRESOLVED,
        AttributionState.NO_SIGNAL,
    ],
)
def test_no_provider_proposal_state_has_an_automatic_export_path(
    state: AttributionState,
) -> None:
    candidate = _candidate()
    first_page = EXTERNAL_TEXT if state is AttributionState.EXTERNALLY_SOURCED else METHODS_TEXT
    layout = _layout(first_page)
    bundle = retrieve_origin_context(candidate, layout)
    origin_page = None if state is AttributionState.NO_SIGNAL else 1
    relation = EvidenceRelation.NONE if origin_page is None else EvidenceRelation.DIRECT
    proposal = _proposal(
        candidate,
        bundle,
        state=state,
        origin_page=origin_page,
        relation=relation,
    )
    assessment = verify_producer_origin_proposal(
        candidate=candidate,
        layout=layout,
        bundle=bundle,
        proposal=proposal,
    )

    assert not assessment.allows_automatic_export
    assert assessment.effective_state is not AttributionState.PAPER_PRODUCED
    assert assessment.route in {OriginRoute.REVIEW, OriginRoute.DEMOTE}
