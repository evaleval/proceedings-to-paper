"""Bulk freeze: retry, pacing, atomic cache writes, resume, and per-paper failures."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest

from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.pipeline import PipelineSettings, freeze_corpus
from proceedings_to_eee.sources.manifest import (
    HostRateLimiter,
    SourceRole,
    download_and_freeze_source,
)

PDF_BYTES = b"%PDF-1.4\nbulk freeze fixture\n%%EOF\n"


class _Recorder:
    """A fake transport that scripts one response sequence per URL."""

    def __init__(self, script: dict[str, list[httpx.Response]]) -> None:
        self._script = {url: list(responses) for url, responses in script.items()}
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        with self._lock:
            self.calls.append(url)
            queued = self._script.get(url)
            if not queued:
                return httpx.Response(404, request=request)
            response = queued.pop(0) if len(queued) > 1 else queued[0]
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers=response.headers,
            request=request,
        )


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):
    def install(script: dict[str, list[httpx.Response]]) -> _Recorder:
        recorder = _Recorder(script)
        original = httpx.Client.__init__

        def patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["transport"] = httpx.MockTransport(recorder.handler)
            original(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched)
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        return recorder

    return install


def _ok(content: bytes = PDF_BYTES) -> httpx.Response:
    return httpx.Response(200, content=content, headers={"content-type": "application/pdf"})


def _settings(tmp_path: Path) -> PipelineSettings:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    return PipelineSettings(
        project_root=tmp_path,
        schema_path=schema,
        schema_sha256="0" * 64,
        output_root=tmp_path / "runs" / "census",
        model="not-used",
    )


def _corpus(count: int, *, base: str = "https://example.org") -> CorpusSpec:
    return CorpusSpec(
        corpus_id="bulk",
        evaluation_split="development",
        description="d",
        papers=[
            PaperSpec(
                paper_id=f"paper-{index}",
                title=f"Paper {index}",
                year=2024,
                venue="acl",
                pdf_url=f"{base}/{index}.pdf",
                perspective_role="Primary paper.",
            )
            for index in range(count)
        ],
    )


def test_throttled_download_is_retried_then_succeeds(tmp_path: Path, transport) -> None:
    url = "https://example.org/0.pdf"
    recorder = transport({url: [httpx.Response(429, headers={"retry-after": "0"}), _ok(), _ok()]})

    source = download_and_freeze_source(
        paper_id="paper-0",
        role=SourceRole.PAPER,
        url=url,
        cache_root=tmp_path / "cache",
    )

    assert source.byte_size == len(PDF_BYTES)
    assert len(recorder.calls) == 2


def test_transient_server_errors_are_retried_up_to_the_limit(tmp_path: Path, transport) -> None:
    url = "https://example.org/0.pdf"
    recorder = transport({url: [httpx.Response(503)]})

    with pytest.raises(httpx.HTTPStatusError):
        download_and_freeze_source(
            paper_id="paper-0",
            role=SourceRole.PAPER,
            url=url,
            cache_root=tmp_path / "cache",
            max_attempts=3,
        )

    assert len(recorder.calls) == 3


def test_permanent_status_is_not_retried(tmp_path: Path, transport) -> None:
    url = "https://example.org/missing.pdf"
    recorder = transport({url: [httpx.Response(404)]})

    with pytest.raises(httpx.HTTPStatusError):
        download_and_freeze_source(
            paper_id="paper-0",
            role=SourceRole.PAPER,
            url=url,
            cache_root=tmp_path / "cache",
            max_attempts=4,
        )

    assert len(recorder.calls) == 1


def test_before_request_hook_runs_once_per_attempt(tmp_path: Path, transport) -> None:
    url = "https://example.org/0.pdf"
    transport({url: [httpx.Response(429, headers={"retry-after": "0"}), _ok(), _ok()]})
    ticks: list[int] = []

    download_and_freeze_source(
        paper_id="paper-0",
        role=SourceRole.PAPER,
        url=url,
        cache_root=tmp_path / "cache",
        before_request=lambda: ticks.append(1),
    )

    assert len(ticks) == 2


def test_identical_bytes_from_two_papers_share_one_cache_entry(tmp_path: Path, transport) -> None:
    transport({f"https://example.org/{index}.pdf": [_ok()] for index in range(6)})
    settings = _settings(tmp_path)

    summary = freeze_corpus(_corpus(6), settings, workers=4, requests_per_second=0.0)

    assert summary["papers_succeeded"] == 6
    cached = list((tmp_path / "data" / "sources").rglob("*.pdf"))
    assert len(cached) == 1
    assert cached[0].read_bytes() == PDF_BYTES


def test_failures_are_per_paper_and_typed(tmp_path: Path, transport) -> None:
    script = {f"https://example.org/{index}.pdf": [_ok()] for index in range(5)}
    script["https://example.org/2.pdf"] = [httpx.Response(404)]
    transport(script)
    settings = _settings(tmp_path)

    summary = freeze_corpus(_corpus(5), settings, workers=3, requests_per_second=0.0)

    assert summary["status"] == "partial_failure"
    assert summary["papers_succeeded"] == 4
    assert summary["papers_failed"] == 1
    assert summary["failure_reasons"] == {"HTTPStatusError": 1}
    failed = [item for item in summary["results"] if item["status"] == "error"]
    assert [item["paper_id"] for item in failed] == ["paper-2"]


def test_results_keep_corpus_order_under_concurrency(tmp_path: Path, transport) -> None:
    transport(
        {f"https://example.org/{index}.pdf": [_ok(f"pdf-{index}".encode())] for index in range(12)}
    )
    settings = _settings(tmp_path)

    summary = freeze_corpus(_corpus(12), settings, workers=6, requests_per_second=0.0)

    assert [item["paper_id"] for item in summary["results"]] == [
        f"paper-{index}" for index in range(12)
    ]


def test_a_reused_paper_still_reports_success_so_preflight_can_plan_it(
    tmp_path: Path, transport
) -> None:
    """Regression: preflight refuses to plan any paper whose freeze status is not
    "success", so reuse must be reported beside the status, never instead of it.

    A previous version returned "reused" here, which made preflight report
    source_freeze_failed for every already-frozen paper in the corpus."""

    transport(
        {f"https://example.org/{index}.pdf": [_ok(f"pdf-{index}".encode())] for index in range(3)}
    )
    settings = _settings(tmp_path)

    first = freeze_corpus(_corpus(3), settings, workers=1, requests_per_second=0.0)
    second = freeze_corpus(_corpus(3), settings, workers=1, requests_per_second=0.0)

    assert {item["status"] for item in first["results"]} == {"success"}
    assert {item["status"] for item in second["results"]} == {"success"}
    assert [item["reused"] for item in first["results"]] == [False, False, False]
    assert [item["reused"] for item in second["results"]] == [True, True, True]


def test_second_freeze_reuses_manifests_and_downloads_nothing(tmp_path: Path, transport) -> None:
    recorder = transport(
        {f"https://example.org/{index}.pdf": [_ok(f"pdf-{index}".encode())] for index in range(5)}
    )
    settings = _settings(tmp_path)

    first = freeze_corpus(_corpus(5), settings, workers=2, requests_per_second=0.0)
    downloads_after_first = len(recorder.calls)
    second = freeze_corpus(_corpus(5), settings, workers=2, requests_per_second=0.0)

    assert first["papers_downloaded"] == 5
    assert first["papers_reused"] == 0
    assert second["papers_downloaded"] == 0
    assert second["papers_reused"] == 5
    assert len(recorder.calls) == downloads_after_first


def test_rate_limiter_paces_concurrent_callers() -> None:
    limiter = HostRateLimiter(50.0)
    started = time.monotonic()

    def worker() -> None:
        for _ in range(5):
            limiter.acquire()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # 20 acquisitions at 50/s cannot finish faster than ~0.38 s.
    assert time.monotonic() - started >= 0.30


def test_rate_limiter_disabled_when_rate_is_zero() -> None:
    limiter = HostRateLimiter(0.0)
    started = time.monotonic()
    for _ in range(100):
        limiter.acquire()
    assert time.monotonic() - started < 0.1
