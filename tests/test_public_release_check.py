from __future__ import annotations

import json
import runpy
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNER = runpy.run_path(str(ROOT / "scripts/check_public_release.py"))
IndexedFile = SCANNER["IndexedFile"]
check_files = SCANNER["check_files"]
read_index = SCANNER["read_index"]
main = SCANNER["main"]
SUMMARY_PATH = "results/census-summary-2026-09-05.json"


def entry(path: str, text: str | bytes = "safe\n", mode: str = "100644"):
    return IndexedFile(path, mode, text.encode() if isinstance(text, str) else text)


def categories(entries, **kwargs) -> set[str]:
    return {finding.category for finding in check_files(entries, **kwargs)}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(*extra_chunks: bytes) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + b"".join(extra_chunks)
        + png_chunk(b"IDAT", zlib.compress(b"\0\0\0\0"))
        + png_chunk(b"IEND", b"")
    )


def test_safe_candidate_tree_and_empty_environment_example() -> None:
    assert (
        check_files(
            [
                entry("README.md", "[Architecture](docs/architecture.md)\n"),
                entry("docs/architecture.md", "[Back](../README.md#overview)\n"),
                entry(".env.example", "# Configure locally.\nOPENROUTER_API_KEY=\n"),
                entry("src/proceedings_to_eee/example.py"),
                entry(SUMMARY_PATH, (ROOT / SUMMARY_PATH).read_text(encoding="utf-8")),
                entry("assets/proceedings-to-eee-pipeline.png", png()),
            ]
        )
        == []
    )


@pytest.mark.parametrize("location", ["root", "sources", "population", "eee"])
def test_summary_cannot_hide_private_fields_in_approved_filename(location: str) -> None:
    summary = json.loads((ROOT / SUMMARY_PATH).read_text(encoding="utf-8"))
    target = summary if location == "root" else summary[location]
    target["source_quote"] = "private source text must not enter the public artifact"
    assert "public result does not match the approved aggregate schema" in categories(
        [entry(SUMMARY_PATH, json.dumps(summary))]
    )


def test_summary_rejects_private_metadata_invalid_types_and_unapproved_text() -> None:
    original = json.loads((ROOT / SUMMARY_PATH).read_text(encoding="utf-8"))
    for path, replacement in (
        (("sources", "producer"), "private metadata"),
        (("population", "papers_total"), True),
        (("sources", "atlas_summary_sha256"), "not a hash"),
        (("classification",), "human_validated"),
        (("limitations",), ["unapproved source text"]),
        (("eee", "records"), -1),
        (("candidates", "text_supported"), 1_000_000),
    ):
        data = json.loads(json.dumps(original))
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        assert "public result does not match the approved aggregate schema" in categories(
            [entry(SUMMARY_PATH, json.dumps(data))]
        )
    duplicate = json.dumps(original).replace(
        '"snapshot_date": "2026-09-05"',
        '"snapshot_date": "private text", "snapshot_date": "2026-09-05"',
    )
    assert "public result does not match the approved aggregate schema" in categories(
        [entry(SUMMARY_PATH, duplicate)]
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("notes.md", "not in the public top-level allowlist"),
        ("vault/status.md", "private or generated artifact directory"),
        ("src/private/annotations.json", "private or generated artifact directory"),
        ("docs/meeting-notes.md", "private working-note filename"),
        ("results/another-summary.json", "result artifact is not explicitly approved"),
        ("assets/diagram.PDF", "disallowed artifact extension"),
        ("examples/private.csv", "disallowed artifact extension"),
        ("examples/trace.jsonl", "disallowed artifact extension"),
        ("docs/.env.backup", "environment file is private"),
    ],
)
def test_private_or_unapproved_paths_fail(path: str, expected: str) -> None:
    assert expected in categories([entry(path)])


@pytest.mark.parametrize("mode", ["120000", "160000", "unmerged"])
def test_non_regular_index_entries_fail(mode: str) -> None:
    assert "symlink, submodule, or unresolved index entry" in categories(
        [entry("README.md", mode=mode)]
    )


def test_binary_pdf_magic_metadata_and_trailing_bytes_fail() -> None:
    image_path = "assets/proceedings-to-eee-pipeline.png"
    assert "binary content is not explicitly approved" in categories(
        [entry("assets/unapproved.png", png())]
    )
    assert "PDF content is not a public repository artifact" in categories(
        [entry("docs/disguised.txt", b"%PDF-1.7\n")]
    )
    for unsafe in (
        png(png_chunk(b"tEXt", b"Comment\0private metadata")),
        png() + b"trailing bytes",
        png()[:-1],
        png().replace(b"IDAT", b"IDBT"),
    ):
        assert "unsafe or metadata-bearing PNG" in categories([entry(image_path, unsafe)])


def test_secret_diagnostics_do_not_echo_values() -> None:
    token = "sk-" + "or-v1-" + "a" * 64
    findings = check_files([entry("README.md", "value=" + token)])
    assert len(findings) == 1
    assert findings[0].category == "secret-shaped content"
    assert findings[0].line == 1
    assert token not in str(findings[0])
    assert "environment example must leave values empty" in categories(
        [entry(".env.example", "OPENROUTER_API_KEY=placeholder\n")]
    )


def test_home_paths_have_narrow_synthetic_test_exception() -> None:
    real_home = "/home/" + "someone" + "/research/source.txt"
    fake_home = "/Users/example/private/source.pdf"
    assert "local home path" in categories([entry("README.md", real_home)])
    assert "local home path" in categories([entry("tests/test_example.py", real_home)])
    assert check_files([entry("tests/test_example.py", fake_home)]) == []
    assert "local home path" in categories([entry("README.md", fake_home)])
    windows_home = "C:" + "\\Users\\" + "someone" + "\\notes.txt"
    assert "local home path" in categories([entry("README.md", windows_home)])


def test_local_document_links_use_candidate_inventory() -> None:
    text = (
        "[Missing](docs/missing.md)\n"
        "[Escape](../outside.md)\n"
        "[Remote](https://example.org/page)\n"
        "[Section](#overview)\n"
        "```markdown\n[Example](does-not-exist.md)\n```\n"
    )
    findings = check_files([entry("README.md", text)])
    assert [(item.line, item.category) for item in findings] == [
        (1, "documentation link is absent from the index"),
        (2, "documentation link escapes the release tree"),
    ]
    assert check_files([entry("README.md", text)], check_links=False) == []


def test_empty_and_oversized_candidates_fail() -> None:
    assert check_files([])[0].path == "<index>"
    assert "file exceeds the release size limit" in categories(
        [entry("README.md", "too long")], max_file_bytes=4
    )


def test_reads_new_repository_index_not_unstaged_worktree(tmp_path: Path, capsys) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    readme = tmp_path / "README.md"
    readme.write_text("# Public prototype\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    private_local_path = "/home/" + "someone" + "/notes.txt"
    readme.write_text(private_local_path, encoding="utf-8")
    assert read_index(tmp_path)[0].content == b"# Public prototype\n"
    assert main(["--root", str(tmp_path)]) == 0
    nested = tmp_path / "src"
    nested.mkdir()
    assert read_index(nested) == read_index(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    assert main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert "local home path" in output.err
    assert private_local_path not in output.err
