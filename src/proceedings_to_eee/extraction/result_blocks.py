"""Paper-agnostic segmentation of layout text into result-rich blocks.

The segmenter deliberately uses only local typography and lexical signals. It has no
knowledge of paper titles, expected systems, metrics, or values. LLM extraction can
therefore operate on bounded blocks while retaining source line ranges and repeated
table/header context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from pydantic import Field, model_validator

from proceedings_to_eee.domain.observation import StrictModel
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout

# A five-line block reaches one-line leaves within three changed-input split levels.
# Keep this shared with execution and offline authorization planning so the latter
# cannot understate the provider calls that the former is allowed to make.
LEGACY_RECOVERY_MAX_DEPTH = 3


class ResultBlock(StrictModel):
    """One bounded, evidence-addressable unit for candidate extraction.

    ``body_text`` is always a contiguous page range. ``context_text`` is a preceding
    heading/caption/header range and ``trailing_context_text`` is commonly a caption
    printed below a table. Both may be repeated across chunks. Keeping them separate
    avoids pretending that repeated context was adjacent to every later row in the
    source.
    """

    schema_version: str = "result-block/0.3"
    block_id: str
    source_id: str
    page: int = Field(ge=1)
    page_ordinal: int = Field(ge=1)
    source_column_start: int | None = Field(default=None, ge=1)
    source_column_end: int | None = Field(default=None, ge=1)
    context_start_line: int | None = Field(default=None, ge=1)
    context_end_line: int | None = Field(default=None, ge=1)
    body_start_line: int = Field(ge=1)
    body_end_line: int = Field(ge=1)
    trailing_context_start_line: int | None = Field(default=None, ge=1)
    trailing_context_end_line: int | None = Field(default=None, ge=1)
    context_text: str = ""
    body_text: str = Field(min_length=1)
    trailing_context_text: str = ""
    text_sha256: str
    character_count: int = Field(ge=1)
    line_count: int = Field(ge=1)
    numeric_token_count: int = Field(ge=0)
    numeric_density_per_kchar: float = Field(ge=0.0)
    result_signal_score: float = Field(ge=0.0)
    signal_kinds: list[str]
    data_row_count: int = Field(ge=0)
    overlap_with_previous_lines: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_ranges_and_digest(self) -> ResultBlock:
        has_column_range = (
            self.source_column_start is not None or self.source_column_end is not None
        )
        if has_column_range:
            if self.source_column_start is None or self.source_column_end is None:
                raise ValueError("source column range must provide both endpoints")
            if self.source_column_end < self.source_column_start:
                raise ValueError("source_column_end must be at or after source_column_start")
        if self.body_end_line < self.body_start_line:
            raise ValueError("body_end_line must be at or after body_start_line")
        has_context_range = self.context_start_line is not None or self.context_end_line is not None
        if has_context_range:
            if self.context_start_line is None or self.context_end_line is None:
                raise ValueError("context line range must provide both endpoints")
            if self.context_end_line < self.context_start_line:
                raise ValueError("context_end_line must be at or after context_start_line")
            if not self.context_text:
                raise ValueError("context line range requires context_text")
        elif self.context_text:
            raise ValueError("context_text requires a context line range")
        has_trailing_range = (
            self.trailing_context_start_line is not None
            or self.trailing_context_end_line is not None
        )
        if has_trailing_range:
            if self.trailing_context_start_line is None or self.trailing_context_end_line is None:
                raise ValueError("trailing context line range must provide both endpoints")
            if self.trailing_context_end_line < self.trailing_context_start_line:
                raise ValueError("trailing_context_end_line must be at or after its start line")
            if not self.trailing_context_text:
                raise ValueError("trailing context line range requires text")
        elif self.trailing_context_text:
            raise ValueError("trailing_context_text requires a line range")
        expected = _text_digest(self.context_text, self.body_text, self.trailing_context_text)
        if self.text_sha256 != expected:
            raise ValueError("text_sha256 does not match the block text spans")
        return self

    def prompt_text(self) -> str:
        """Render context and body with explicit, non-source boundary labels."""

        column_label = (
            f", columns {self.source_column_start}-{self.source_column_end}"
            if self.source_column_start is not None
            else ""
        )
        parts: list[str] = []
        if self.context_text:
            parts.append(
                f"LEADING CONTEXT (page {self.page}, lines {self.context_start_line}-"
                f"{self.context_end_line}{column_label}):\n{self.context_text}"
            )
        parts.append(
            f"RESULT BLOCK (page {self.page}, lines {self.body_start_line}-"
            f"{self.body_end_line}{column_label}):\n{self.body_text}"
        )
        if self.trailing_context_text:
            parts.append(
                f"TRAILING CONTEXT (page {self.page}, lines "
                f"{self.trailing_context_start_line}-{self.trailing_context_end_line}"
                f"{column_label}):\n"
                f"{self.trailing_context_text}"
            )
        return "".join(parts)


@dataclass(frozen=True, slots=True)
class ResultBlockConfig:
    """Deterministic segmentation and prompt-budget limits."""

    max_lines: int = 40
    max_characters: int = 6_000
    context_lines: int = 8
    trailing_context_lines: int = 2
    overlap_lines: int = 3
    signal_gap_lines: int = 3
    max_blank_gap: int = 1
    max_data_rows: int | None = 6
    min_signal_score: float = 1.5
    max_blocks_per_page: int | None = None
    detect_parallel_columns: bool = True
    min_column_gutter_width: int = 3
    min_parallel_lines: int = 4
    max_column_analysis_width: int = 240

    def __post_init__(self) -> None:
        if self.max_lines < 2:
            raise ValueError("max_lines must be at least 2")
        if self.max_characters < 128:
            raise ValueError("max_characters must be at least 128")
        if not 0 <= self.context_lines < self.max_lines:
            raise ValueError("context_lines must be non-negative and smaller than max_lines")
        if self.trailing_context_lines < 0:
            raise ValueError("trailing_context_lines must be non-negative")
        if not 0 <= self.overlap_lines < self.max_lines:
            raise ValueError("overlap_lines must be non-negative and smaller than max_lines")
        if self.signal_gap_lines < 0 or self.max_blank_gap < 0:
            raise ValueError("signal and blank gap limits must be non-negative")
        if self.min_signal_score <= 0:
            raise ValueError("min_signal_score must be positive")
        if self.max_data_rows is not None and self.max_data_rows < 1:
            raise ValueError("max_data_rows must be positive when provided")
        if self.max_blocks_per_page is not None and self.max_blocks_per_page < 1:
            raise ValueError("max_blocks_per_page must be positive when provided")
        if self.min_column_gutter_width < 2:
            raise ValueError("min_column_gutter_width must be at least 2")
        if self.min_parallel_lines < 2:
            raise ValueError("min_parallel_lines must be at least 2")
        if self.max_column_analysis_width < 80:
            raise ValueError("max_column_analysis_width must be at least 80")


_NUMERIC = re.compile(r"(?<![\w@])[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*%?(?!\w)")
_RESULT_TERMS = re.compile(
    r"\b(?:results?|accuracy|auc|auroc|f1|precision|recall|error|rate|score|"
    r"performance|evaluation|evaluated|mean|median|significant|correlation|"
    r"agreement|specificity|sensitivity|loss|perplexity)\b",
    re.IGNORECASE,
)
_CAPTION_LABEL = (
    r"(?:table|figure|fig\.?|chart|plot)\s+"
    r"(?:[A-Z]?\d+(?:[.\-]\d+)*|[IVXLCDM]+)\s*(?::|[.]|[-–—])"
)
_CAPTION = re.compile(rf"^\s*{_CAPTION_LABEL}", re.IGNORECASE)
_CAPTION_ANY = re.compile(
    rf"(?:^|\s{{2,}}){_CAPTION_LABEL}",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(r"^\s*(?:(?:\d+(?:\.\d+)*)|(?:[A-Z]))(?:[.)])?\s+[A-Za-z]")
_STATISTIC = re.compile(
    r"(?:\b(?:p|r|rho|f|t|z)\s*[=<>]|\b(?:ci|confidence interval)\b|[±])",
    re.IGNORECASE,
)
_REFERENCE_LIKE = re.compile(r"(?:https?://|\bdoi\b|\bpp?\.\s*\d+[-–]\d+)", re.IGNORECASE)
_DECIMAL_OR_PERCENT = re.compile(
    r"(?:[<>≤≥≈~]\s*)?(?:\d+[.,]\d+|[.,]\d+|\d+\s*%)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _LineFeatures:
    index: int
    text: str
    blank: bool
    numeric_count: int
    numeric_token_ratio: float
    caption: bool
    heading: bool
    result_heading: bool
    aligned_columns: bool
    score: float
    signal_kinds: tuple[str, ...]

    @property
    def is_signal(self) -> bool:
        return self.score > 0.0

    @property
    def is_data_signal(self) -> bool:
        return any(
            kind
            in {
                "aligned_numeric_row",
                "numeric_dense",
                "metric_value",
                "statistic",
                "wrapped_metric_value",
            }
            for kind in self.signal_kinds
        )

    @property
    def is_tabular_data(self) -> bool:
        return any(kind in {"aligned_numeric_row", "numeric_dense"} for kind in self.signal_kinds)


@dataclass(frozen=True, slots=True)
class _Cluster:
    signal_indices: tuple[int, ...]

    @property
    def start(self) -> int:
        return self.signal_indices[0]

    @property
    def end(self) -> int:
        return self.signal_indices[-1]


@dataclass(frozen=True, slots=True)
class _BlockDraft:
    context_start: int | None
    context_end: int | None
    body_start: int
    body_end: int
    trailing_context_start: int | None
    trailing_context_end: int | None
    overlap: int
    score: float
    signal_kinds: tuple[str, ...]
    data_row_count: int


@dataclass(frozen=True, slots=True)
class _PagePanel:
    """A deterministic vertical slice whose lines retain original page ordinals."""

    lines: tuple[str, ...]
    source_column_start: int | None = None
    source_column_end: int | None = None


def _is_heading(text: str, *, numeric_count: int, aligned_columns: bool) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120 or len(stripped.split()) > 16:
        return False
    if aligned_columns or numeric_count > 1:
        return False
    letters = [character for character in stripped if character.isalpha()]
    uppercase = bool(letters) and all(character.isupper() for character in letters)
    title_like = len(stripped.split()) >= 2 and stripped == stripped.title()
    return bool(
        _NUMBERED_HEADING.match(stripped) or (numeric_count == 0 and (uppercase or title_like))
    )


def _line_features(index: int, text: str, minimum_score: float) -> _LineFeatures:
    stripped = text.strip()
    blank = not stripped
    numbers = _NUMERIC.findall(text)
    numeric_count = len(numbers)
    token_count = len(re.findall(r"\S+", stripped))
    numeric_token_ratio = numeric_count / max(token_count, 1)
    columns = [part for part in re.split(r"\s{2,}", stripped) if part]
    aligned_columns = len(columns) >= 2
    caption_at_start = bool(_CAPTION.match(stripped))
    caption = bool(_CAPTION_ANY.search(stripped))
    heading = _is_heading(stripped, numeric_count=numeric_count, aligned_columns=aligned_columns)
    result_term = bool(_RESULT_TERMS.search(stripped))
    result_heading = heading and result_term
    kinds: list[str] = []
    score = 0.0

    if caption:
        kinds.append("caption")
        score += 3.0
    if result_heading:
        kinds.append("result_heading")
        score += 2.5

    # A caption or numbered heading contains a label number, not a reported value.
    if not caption_at_start and not heading:
        if numeric_count >= 2 and numeric_token_ratio >= 0.25:
            kinds.append("numeric_dense")
            score += min(3.0, 1.5 + numeric_token_ratio * 2.0)
        if aligned_columns and numeric_count and numeric_token_ratio >= 0.18:
            kinds.append("aligned_numeric_row")
            score += 1.75
        if result_term and numeric_count:
            kinds.append("metric_value")
            score += 1.75
        if numeric_count and _STATISTIC.search(stripped):
            kinds.append("statistic")
            score += 1.5

    if _REFERENCE_LIKE.search(stripped) and not result_term and not caption:
        score = max(0.0, score - 2.5)
        if score == 0.0:
            kinds.clear()
    if blank or score < minimum_score:
        score = 0.0
        kinds.clear()

    return _LineFeatures(
        index=index,
        text=text,
        blank=blank,
        numeric_count=numeric_count,
        numeric_token_ratio=round(numeric_token_ratio, 6),
        caption=caption,
        heading=heading,
        result_heading=result_heading,
        aligned_columns=aligned_columns,
        score=round(score, 6),
        signal_kinds=tuple(dict.fromkeys(kinds)),
    )


def _page_line_features(lines: list[str], minimum_score: float) -> list[_LineFeatures]:
    """Score lines and recover values whose metric is on an adjacent wrapped line.

    Poppler frequently places a sparse table cell or the final part of a prose result
    on a line by itself. A lone decimal is deliberately not a result signal unless a
    nearby, uninterrupted line names a result metric. The LLM still decides whether
    the resulting block contains a reportable observation.
    """

    features = [_line_features(index, line, minimum_score) for index, line in enumerate(lines)]
    promoted: list[_LineFeatures] = []
    for feature in features:
        if (
            feature.is_signal
            or feature.blank
            or feature.heading
            or feature.caption
            or feature.numeric_count == 0
            or not _DECIMAL_OR_PERCENT.search(feature.text)
            or _REFERENCE_LIKE.search(feature.text)
        ):
            promoted.append(feature)
            continue
        window: list[_LineFeatures] = []
        for direction in (-1, 1):
            for distance in (1, 2):
                neighbor_index = feature.index + direction * distance
                if neighbor_index < 0 or neighbor_index >= len(features):
                    break
                neighbor = features[neighbor_index]
                if neighbor.blank or neighbor.caption:
                    break
                window.append(neighbor)
                if neighbor.heading:
                    break
        if not any(_RESULT_TERMS.search(neighbor.text) for neighbor in window):
            promoted.append(feature)
            continue
        promoted.append(
            replace(
                feature,
                score=max(feature.score, 1.75),
                signal_kinds=tuple((*feature.signal_kinds, "wrapped_metric_value")),
            )
        )
    return promoted


def _supports_parallel_panel(text: str) -> bool:
    """Require lexical content on each side so a wide single table is not split."""

    words = re.findall(r"[A-Za-z]{2,}", text)
    return bool(words) and (len(words) >= 2 or bool(_NUMERIC.search(text)))


def _column_gutter(lines: list[str], config: ResultBlockConfig) -> tuple[int, int] | None:
    """Find a stable central gutter shared by two independently meaningful panels.

    The returned endpoints are zero-based and inclusive. Very wide vector-graphic
    text is excluded from this heuristic; it is kept intact for the visual/caption
    path instead of being mistaken for ordinary two-column body text.
    """

    if not config.detect_parallel_columns:
        return None
    raw_width = max((len(line.rstrip()) for line in lines), default=0)
    if raw_width > config.max_column_analysis_width * 2:
        return None
    analysis_lines = [
        line.rstrip()
        for line in lines
        if line.strip() and len(line.rstrip()) <= config.max_column_analysis_width
    ]
    if len(analysis_lines) < config.min_parallel_lines:
        return None
    widths = sorted(len(line) for line in analysis_lines)
    width = widths[min(len(widths) - 1, (len(widths) * 9) // 10)]
    if width < 80:
        return None
    lower = max(25, int(width * 0.35))
    upper = min(width - 25, int(width * 0.65))
    occupancy_limit = max(1, len(analysis_lines) // 20)
    sparse_columns = [
        column
        for column in range(lower, upper + 1)
        if sum(column < len(line) and not line[column].isspace() for line in analysis_lines)
        <= occupancy_limit
    ]
    if not sparse_columns:
        return None
    runs: list[tuple[int, int]] = []
    run_start = sparse_columns[0]
    run_end = run_start
    for column in sparse_columns[1:]:
        if column == run_end + 1:
            run_end = column
            continue
        runs.append((run_start, run_end))
        run_start = run_end = column
    runs.append((run_start, run_end))

    candidates: list[tuple[int, int, int, int]] = []
    for start, end in runs:
        if end - start + 1 < config.min_column_gutter_width:
            continue
        shared_lines = 0
        for line in analysis_lines:
            if line[start : end + 1].strip():
                continue
            left = line[:start].strip()
            right = line[end + 1 :].strip()
            if _supports_parallel_panel(left) and _supports_parallel_panel(right):
                shared_lines += 1
        if shared_lines < config.min_parallel_lines:
            continue
        midpoint = (start + end) // 2
        balance = -abs(midpoint - width // 2)
        candidates.append((shared_lines, end - start + 1, balance, start))
    if not candidates:
        return None
    _, gutter_width, _, start = max(candidates)
    return start, start + gutter_width - 1


def _full_width_runs(
    lines: list[str], gutter: tuple[int, int], config: ResultBlockConfig
) -> list[tuple[int, int]]:
    """Line runs that belong to one table spanning the full page width.

    A full-width table on an otherwise two-column page is severed by the gutter: half of
    every row lands in each panel and the row exists in neither. It is recognised by a
    stable grid of numeric column offsets straddling the gutter. A lone number in
    left-column prose beside a right-column table also straddles, so at least two numeric
    offsets are required on each side and at least two of the left offsets must recur
    across the run. In development diagnostics, true rows shared a stable offset grid,
    while incidental prose-table straddles carried a single shifting offset.
    """

    gutter_start, gutter_end = gutter
    grids: list[frozenset[int]] = []
    for line in lines:
        offsets = [match.start() for match in _NUMERIC.finditer(line)]
        left = frozenset(offset for offset in offsets if offset < gutter_start)
        right = [offset for offset in offsets if offset > gutter_end]
        grids.append(left if len(left) >= 2 and len(right) >= 2 else frozenset())

    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(grids):
        if not grids[index]:
            index += 1
            continue
        end = index
        shared = grids[index]
        while end + 1 < len(grids) and grids[end + 1] and len(shared & grids[end + 1]) >= 2:
            shared &= grids[end + 1]
            end += 1
        if end - index + 1 >= 3:
            runs.append(_extend_full_width_run(lines, index, end, config))
        index = end + 1
    return runs


def _extend_full_width_run(
    lines: list[str], start: int, end: int, config: ResultBlockConfig
) -> tuple[int, int]:
    """Pull the table's own header and caption lines into the full-width run.

    Extension only crosses lines that read as an aligned header or a caption. Two-column
    body prose carries long text on both sides of the gutter and is excluded by the same
    header test the region index uses, which keeps a neighbouring paragraph from being
    removed from both column panels.
    """

    def attached(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and (_is_table_header_line(line) or bool(_CAPTION.match(stripped)))

    lower = start
    blanks = 0
    for index in range(start - 1, max(-1, start - config.context_lines - 1), -1):
        if not lines[index].strip():
            blanks += 1
            if blanks > config.max_blank_gap:
                break
            continue
        if not attached(lines[index]):
            break
        blanks = 0
        lower = index

    upper = end
    blanks = 0
    for index in range(end + 1, min(len(lines), end + config.trailing_context_lines + 2)):
        if not lines[index].strip():
            blanks += 1
            if blanks > config.max_blank_gap:
                break
            continue
        if not attached(lines[index]):
            break
        blanks = 0
        upper = index
    return lower, upper


def _is_table_header_line(text: str) -> bool:
    """A column-aligned, value-free line: a header row rather than a sentence."""

    stripped = text.strip()
    if not stripped or _CAPTION.match(stripped):
        return False
    cells = [cell for cell in re.split(r"\s{2,}", stripped) if cell]
    if len(cells) < 2:
        return False
    return all(len(cell.split()) <= 6 for cell in cells)


def _page_panels(lines: list[str], config: ResultBlockConfig) -> list[_PagePanel]:
    gutter = _column_gutter(lines, config)
    if gutter is None:
        return [_PagePanel(lines=tuple(lines))]
    gutter_start, gutter_end = gutter
    source_width = max(len(line.rstrip()) for line in lines)

    full_width = _full_width_runs(lines, gutter, config)
    spanning = {index for start, end in full_width for index in range(start, end + 1)}

    left_lines = tuple(
        "" if index in spanning else line[:gutter_start].rstrip()
        for index, line in enumerate(lines)
    )
    right_lines = tuple(
        "" if index in spanning else line[gutter_end + 1 :].rstrip()
        for index, line in enumerate(lines)
    )
    panels = [
        _PagePanel(
            lines=left_lines,
            source_column_start=1,
            source_column_end=gutter_start,
        ),
        _PagePanel(
            lines=right_lines,
            source_column_start=gutter_end + 2,
            source_column_end=source_width,
        ),
    ]
    if spanning:
        panels.insert(
            0,
            _PagePanel(
                lines=tuple(
                    line.rstrip() if index in spanning else "" for index, line in enumerate(lines)
                ),
                source_column_start=1,
                source_column_end=source_width,
            ),
        )
    return panels


def _cluster_signals(features: list[_LineFeatures], config: ResultBlockConfig) -> list[_Cluster]:
    signal_indices = [feature.index for feature in features if feature.is_signal]
    if not signal_indices:
        return []

    clusters: list[list[int]] = [[signal_indices[0]]]
    for current in signal_indices[1:]:
        previous = clusters[-1][-1]
        between = features[previous + 1 : current]
        previous_feature = features[previous]
        current_feature = features[current]
        cluster_features = [features[index] for index in clusters[-1]]
        tabular_run = previous_feature.is_tabular_data and current_feature.is_tabular_data
        last_caption = next(
            (feature.index for feature in reversed(cluster_features) if feature.caption),
            None,
        )
        last_data = next(
            (feature.index for feature in reversed(cluster_features) if feature.is_data_signal),
            None,
        )
        caption_closed_table = (
            current_feature.is_data_signal
            and last_caption is not None
            and last_data is not None
            and last_caption > last_data
        )
        separated = (
            current - previous - 1 > config.signal_gap_lines
            or sum(feature.blank for feature in between) > config.max_blank_gap
            or any(feature.caption for feature in between)
            or (any(feature.heading for feature in between) and not tabular_run)
            or (
                current_feature.heading and any(not feature.heading for feature in cluster_features)
            )
            or caption_closed_table
        )
        if separated:
            clusters.append([current])
        else:
            clusters[-1].append(current)
    clustered = [_Cluster(signal_indices=tuple(indices)) for indices in clusters]
    return [cluster for cluster in clustered if not _is_isolated_edge_artifact(cluster, features)]


def _is_isolated_edge_artifact(cluster: _Cluster, features: list[_LineFeatures]) -> bool:
    """Reject a lone running header/footer without suppressing an edge table."""

    if len(cluster.signal_indices) != 1:
        return False
    feature = features[cluster.start]
    at_page_edge = feature.index <= 1 or feature.index >= len(features) - 2
    substantive = (
        feature.caption
        or feature.result_heading
        or bool({"metric_value", "statistic"} & set(feature.signal_kinds))
    )
    return at_page_edge and not substantive


def _expanded_start(
    features: list[_LineFeatures], lower_bound: int, start: int, context_lines: int
) -> int:
    if features[start].caption:
        header_start = _caption_only_header_start(features, lower_bound, start, context_lines)
        if header_start is not None:
            return header_start
        return start
    # A detected heading is already the strongest available local context. Pulling
    # an earlier paragraph into it can cross from methods into results.
    if features[start].heading:
        return start
    candidate = max(lower_bound, start - context_lines)
    blank_run = 0
    for index in range(start - 1, candidate - 1, -1):
        feature = features[index]
        if feature.blank:
            blank_run += 1
            if blank_run >= 2:
                candidate = index + blank_run
                break
            continue
        blank_run = 0
        if feature.heading or feature.caption:
            candidate = index
            break
    while candidate < start and features[candidate].blank:
        candidate += 1
    return candidate


def _caption_only_header_start(
    features: list[_LineFeatures], lower_bound: int, caption: int, context_lines: int
) -> int | None:
    """Find a multi-line aligned header above a caption-only/graphical table."""

    window_start = max(lower_bound, caption - context_lines)
    aligned = [
        index
        for index in range(window_start, caption)
        if features[index].aligned_columns and features[index].numeric_count == 0
    ]
    paired = [
        (left, right)
        for left, right in zip(aligned, aligned[1:], strict=False)
        if right - left <= 2
    ]
    if paired:
        start = paired[-1][0]
    else:
        # A graphically rendered table may expose only one aligned header row to
        # pdftotext. Keep it when it explicitly names a result metric; arbitrary
        # two-column prose is not enough.
        metric_headers = [index for index in aligned if _RESULT_TERMS.search(features[index].text)]
        if not metric_headers:
            return None
        start = metric_headers[-1]
    while start > window_start and (
        features[start - 1].heading or features[start - 1].aligned_columns
    ):
        start -= 1
    return start


def _expanded_end(
    features: list[_LineFeatures], upper_bound: int, end: int, trailing_lines: int
) -> int:
    candidate = min(upper_bound, end + trailing_lines)
    blank_run = 0
    for index in range(end + 1, candidate + 1):
        feature = features[index]
        if feature.heading or feature.caption:
            return index - 1
        if feature.blank:
            blank_run += 1
            if blank_run >= 2:
                return index - blank_run
        else:
            blank_run = 0
    while candidate > end and features[candidate].blank:
        candidate -= 1
    return candidate


def _trim_context_to_limits(
    lines: list[str], start: int, end: int, config: ResultBlockConfig
) -> tuple[int, int] | None:
    while start <= end and not lines[start].strip():
        start += 1
    while end >= start and not lines[end].strip():
        end -= 1
    if start > end:
        return None
    start = max(start, end - config.context_lines + 1)
    # Leave room for body text and a possible caption below the table.
    context_budget = max(1, config.max_characters // 3)
    while start < end and len(_join_lines(lines, start, end)) > context_budget:
        start += 1
    return start, end


def _trailing_caption_range(
    lines: list[str],
    features: list[_LineFeatures],
    last_data: int | None,
    expanded_end: int,
    config: ResultBlockConfig,
) -> tuple[int, int] | None:
    if last_data is None:
        return None
    caption_start = next(
        (index for index in range(last_data + 1, expanded_end + 1) if features[index].caption),
        None,
    )
    if caption_start is None:
        return None
    end = min(expanded_end, caption_start + config.context_lines - 1)
    while end > caption_start and not lines[end].strip():
        end -= 1
    context_budget = max(1, config.max_characters // 3)
    while end > caption_start and len(_join_lines(lines, caption_start, end)) > context_budget:
        end -= 1
    return caption_start, end


def _join_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start : end + 1]).rstrip() + "\n"


def _chunk_end(
    lines: list[str],
    features: list[_LineFeatures],
    start: int,
    final_end: int,
    max_lines: int,
    max_characters: int,
    max_data_rows: int | None,
) -> int:
    end = start
    characters = len(lines[start]) + 1
    data_rows = int(features[start].is_tabular_data)
    while end < final_end:
        next_characters = len(lines[end + 1]) + 1
        next_is_data = int(features[end + 1].is_tabular_data)
        exceeds_data_rows = max_data_rows is not None and data_rows + next_is_data > max_data_rows
        if (
            end - start + 2 > max_lines
            or characters + next_characters > max_characters
            or exceeds_data_rows
        ):
            break
        end += 1
        characters += next_characters
        data_rows += next_is_data

    # Prefer a nearby blank boundary for prose, but do not sacrifice more than 25%
    # of the available chunk. Dense tables normally contain no such boundary.
    if end < final_end:
        earliest_break = start + max(1, (end - start + 1) * 3 // 4)
        for candidate in range(end, earliest_break - 1, -1):
            if not lines[candidate].strip() and candidate > start:
                return candidate - 1
    return end


def _drafts_for_cluster(
    lines: list[str],
    features: list[_LineFeatures],
    clusters: list[_Cluster],
    cluster_index: int,
    config: ResultBlockConfig,
) -> list[_BlockDraft]:
    cluster = clusters[cluster_index]
    lower_bound = clusters[cluster_index - 1].end + 1 if cluster_index else 0
    upper_bound = (
        clusters[cluster_index + 1].start - 1
        if cluster_index + 1 < len(clusters)
        else len(lines) - 1
    )
    expanded_start = _expanded_start(features, lower_bound, cluster.start, config.context_lines)
    expanded_end = _expanded_end(features, upper_bound, cluster.end, config.trailing_context_lines)

    first_data = next(
        (
            index
            for index in range(expanded_start, expanded_end + 1)
            if features[index].is_data_signal
        ),
        None,
    )
    last_data = next(
        (
            index
            for index in range(expanded_end, expanded_start - 1, -1)
            if features[index].is_data_signal
        ),
        None,
    )
    context_range: tuple[int, int] | None = None
    body_start = expanded_start
    if first_data is not None and first_data > expanded_start:
        context_range = _trim_context_to_limits(lines, expanded_start, first_data - 1, config)
        body_start = first_data
    elif features[cluster.start].caption and cluster.start > expanded_start:
        context_range = _trim_context_to_limits(lines, expanded_start, cluster.start - 1, config)
        body_start = cluster.start
    trailing_context_range = _trailing_caption_range(
        lines, features, last_data, expanded_end, config
    )
    body_end = trailing_context_range[0] - 1 if trailing_context_range is not None else expanded_end
    while body_start < body_end and not lines[body_start].strip():
        body_start += 1
    while body_end > body_start and not lines[body_end].strip():
        body_end -= 1

    context_line_count = context_range[1] - context_range[0] + 1 if context_range is not None else 0
    context_text = (
        _join_lines(lines, context_range[0], context_range[1]) if context_range is not None else ""
    )
    trailing_context_line_count = (
        trailing_context_range[1] - trailing_context_range[0] + 1
        if trailing_context_range is not None
        else 0
    )
    trailing_context_text = (
        _join_lines(lines, trailing_context_range[0], trailing_context_range[1])
        if trailing_context_range is not None
        else ""
    )
    body_line_limit = max(
        1,
        config.max_lines - context_line_count - trailing_context_line_count,
    )
    body_character_limit = max(
        1,
        config.max_characters - len(context_text) - len(trailing_context_text),
    )

    drafts: list[_BlockDraft] = []
    cursor = body_start
    previous_end: int | None = None
    while cursor <= body_end:
        end = _chunk_end(
            lines,
            features,
            cursor,
            body_end,
            body_line_limit,
            body_character_limit,
            config.max_data_rows,
        )
        overlap = 0 if previous_end is None else max(0, previous_end - cursor + 1)
        feature_slice = features[cursor : end + 1]
        if context_range is not None:
            feature_slice = features[context_range[0] : context_range[1] + 1] + feature_slice
        if trailing_context_range is not None:
            feature_slice += features[trailing_context_range[0] : trailing_context_range[1] + 1]
        kinds = tuple(
            dict.fromkeys(kind for feature in feature_slice for kind in feature.signal_kinds)
        )
        score = round(sum(feature.score for feature in feature_slice), 6)
        drafts.append(
            _BlockDraft(
                context_start=context_range[0] if context_range else None,
                context_end=context_range[1] if context_range else None,
                body_start=cursor,
                body_end=end,
                trailing_context_start=(
                    trailing_context_range[0] if trailing_context_range else None
                ),
                trailing_context_end=(
                    trailing_context_range[1] if trailing_context_range else None
                ),
                overlap=overlap,
                score=score,
                signal_kinds=kinds,
                data_row_count=sum(
                    feature.is_tabular_data for feature in features[cursor : end + 1]
                ),
            )
        )
        if end >= body_end:
            break
        previous_end = end
        next_cursor = max(cursor + 1, end - config.overlap_lines + 1)
        cursor = next_cursor
    return drafts


def _text_digest(context_text: str, body_text: str, trailing_context_text: str) -> str:
    return hashlib.sha256(
        context_text.encode("utf-8")
        + b"\0"
        + body_text.encode("utf-8")
        + b"\0"
        + trailing_context_text.encode("utf-8")
    ).hexdigest()


def _materialize_block(
    *,
    page: PageFragment,
    lines: list[str],
    draft: _BlockDraft,
    page_ordinal: int,
    source_column_start: int | None,
    source_column_end: int | None,
) -> ResultBlock:
    context_text = (
        _join_lines(lines, draft.context_start, draft.context_end)
        if draft.context_start is not None and draft.context_end is not None
        else ""
    )
    body_text = _join_lines(lines, draft.body_start, draft.body_end)
    trailing_context_text = (
        _join_lines(
            lines,
            draft.trailing_context_start,
            draft.trailing_context_end,
        )
        if draft.trailing_context_start is not None and draft.trailing_context_end is not None
        else ""
    )
    text_digest = _text_digest(context_text, body_text, trailing_context_text)
    identity_parts = [
        page.source_id,
        str(page.page),
        str(draft.context_start),
        str(draft.context_end),
        str(draft.body_start),
        str(draft.body_end),
        str(draft.trailing_context_start),
        str(draft.trailing_context_end),
        text_digest,
    ]
    # Preserve v0.2 identities for unsplit blocks. Only genuinely new panel
    # slices need column coordinates in their content address.
    if source_column_start is not None and source_column_end is not None:
        identity_parts.extend((str(source_column_start), str(source_column_end)))
    identity = "\0".join(identity_parts)
    block_id = "rblk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    combined_text = context_text + body_text + trailing_context_text
    numeric_count = len(_NUMERIC.findall(combined_text))
    character_count = len(combined_text)
    line_count = (
        draft.body_end
        - draft.body_start
        + 1
        + (
            draft.context_end - draft.context_start + 1
            if draft.context_start is not None and draft.context_end is not None
            else 0
        )
        + (
            draft.trailing_context_end - draft.trailing_context_start + 1
            if draft.trailing_context_start is not None and draft.trailing_context_end is not None
            else 0
        )
    )
    return ResultBlock(
        block_id=block_id,
        source_id=page.source_id,
        page=page.page,
        page_ordinal=page_ordinal,
        source_column_start=source_column_start,
        source_column_end=source_column_end,
        context_start_line=draft.context_start + 1 if draft.context_start is not None else None,
        context_end_line=draft.context_end + 1 if draft.context_end is not None else None,
        body_start_line=draft.body_start + 1,
        body_end_line=draft.body_end + 1,
        trailing_context_start_line=(
            draft.trailing_context_start + 1 if draft.trailing_context_start is not None else None
        ),
        trailing_context_end_line=(
            draft.trailing_context_end + 1 if draft.trailing_context_end is not None else None
        ),
        context_text=context_text,
        body_text=body_text,
        trailing_context_text=trailing_context_text,
        text_sha256=text_digest,
        character_count=character_count,
        line_count=line_count,
        numeric_token_count=numeric_count,
        numeric_density_per_kchar=round(numeric_count / max(character_count / 1_000.0, 1.0), 6),
        result_signal_score=draft.score,
        signal_kinds=list(draft.signal_kinds),
        data_row_count=draft.data_row_count,
        overlap_with_previous_lines=draft.overlap,
    )


def segment_page_result_blocks(
    page: PageFragment, config: ResultBlockConfig | None = None
) -> list[ResultBlock]:
    """Segment one layout page into bounded result-rich blocks in source order.

    Limits are hard for ordinary source lines. If one individual Poppler line exceeds
    ``max_characters``, it is preserved intact rather than silently truncated.
    """

    config = config or ResultBlockConfig()
    lines = page.text.rstrip("\n").split("\n")
    if not lines or not any(line.strip() for line in lines):
        return []
    panel_drafts: list[tuple[_PagePanel, _BlockDraft]] = []
    for panel in _page_panels(lines, config):
        panel_lines = list(panel.lines)
        features = _page_line_features(panel_lines, config.min_signal_score)
        clusters = _cluster_signals(features, config)
        panel_drafts.extend(
            (panel, draft)
            for cluster_index in range(len(clusters))
            for draft in _drafts_for_cluster(
                panel_lines,
                features,
                clusters,
                cluster_index,
                config,
            )
        )

    if config.max_blocks_per_page is not None and len(panel_drafts) > config.max_blocks_per_page:
        ranked = sorted(
            panel_drafts,
            key=lambda item: (
                item[1].score,
                item[1].body_end - item[1].body_start,
                -item[1].body_start,
                -(item[0].source_column_start or 1),
            ),
            reverse=True,
        )[: config.max_blocks_per_page]
        panel_drafts = sorted(
            ranked,
            key=lambda item: (
                item[0].source_column_start or 1,
                item[1].body_start,
                item[1].body_end,
            ),
        )

    return [
        _materialize_block(
            page=page,
            lines=list(panel.lines),
            draft=draft,
            page_ordinal=ordinal,
            source_column_start=panel.source_column_start,
            source_column_end=panel.source_column_end,
        )
        for ordinal, (panel, draft) in enumerate(panel_drafts, start=1)
    ]


def segment_result_blocks(
    layout: PdfLayout, config: ResultBlockConfig | None = None
) -> list[ResultBlock]:
    """Segment all pages and return stable document order."""

    config = config or ResultBlockConfig()
    pages = sorted(layout.pages, key=lambda page: (page.page, page.fragment_id))
    return [block for page in pages for block in segment_page_result_blocks(page, config=config)]


class PagePanel(StrictModel):
    """One deterministic vertical slice of a page in original line ordinals.

    ``lines`` is index-aligned with the page's own line list: a line that carries no
    text inside this panel's column window is present as an empty string. Column
    endpoints are one-based and inclusive so they compose with ``ResultBlock``.
    """

    lines: list[str]
    source_column_start: int | None = None
    source_column_end: int | None = None


def page_panels(page: PageFragment, config: ResultBlockConfig | None = None) -> list[PagePanel]:
    """Split one page into column panels using the segmenter's own gutter detection.

    Exposed so that any consumer needing page geometry, such as the region index,
    reuses this exact gutter rule instead of growing a second one that can disagree.
    """

    config = config or ResultBlockConfig()
    lines = page.text.splitlines()
    return [
        PagePanel(
            lines=list(panel.lines),
            source_column_start=panel.source_column_start,
            source_column_end=panel.source_column_end,
        )
        for panel in _page_panels(lines, config)
    ]


def split_result_block(block: ResultBlock) -> list[ResultBlock]:
    """Halve a block's body into two smaller blocks that keep the same context.

    Retrying a temperature-zero, fixed-seed request with identical input cannot change
    its result, so a block whose response failed content validation can only be
    recovered by changing the input. Halving the body is the smallest such change that
    preserves every line range and every anchor the original block would have produced.

    Leading and trailing context are repeated on both halves. The block contract already
    treats context as repeatable across chunks, and the extraction prompt forbids
    emitting repeated context as a result.
    """

    body_lines = block.body_text.splitlines()
    if len(body_lines) < 2:
        return []
    midpoint = len(body_lines) // 2
    spans = (
        (0, midpoint, block.body_start_line, block.body_start_line + midpoint - 1),
        (midpoint, len(body_lines), block.body_start_line + midpoint, block.body_end_line),
    )
    halves: list[ResultBlock] = []
    for ordinal, (start, stop, start_line, end_line) in enumerate(spans, start=1):
        body_text = "".join(f"{line}\n" for line in body_lines[start:stop])
        features = _page_line_features(body_lines[start:stop], ResultBlockConfig().min_signal_score)
        halves.append(
            block.model_copy(
                update={
                    "block_id": f"{block.block_id}_s{ordinal}",
                    "body_start_line": start_line,
                    "body_end_line": end_line,
                    "body_text": body_text,
                    "text_sha256": _text_digest(
                        block.context_text, body_text, block.trailing_context_text
                    ),
                    "character_count": len(block.context_text)
                    + len(body_text)
                    + len(block.trailing_context_text),
                    "line_count": stop - start,
                    "numeric_token_count": sum(feature.numeric_count for feature in features),
                    "data_row_count": sum(feature.is_tabular_data for feature in features),
                    "overlap_with_previous_lines": 0,
                }
            )
        )
    return halves


def maximum_legacy_block_invocations(
    block: ResultBlock,
    *,
    max_recovery_depth: int = LEGACY_RECOVERY_MAX_DEPTH,
) -> int:
    """Return the exact hard invocation bound for one uncached legacy block.

    The initial full-block call counts once. Every completed response-validation
    failure may then split into two changed-input child calls until the fixed depth or
    one-line leaves stop recursion. Request rejections and transport failures stop
    earlier, so the all-validation-failure tree is the hard maximum.
    """

    if max_recovery_depth < 0:
        raise ValueError("max recovery depth must be non-negative")

    def count(node: ResultBlock, remaining_depth: int) -> int:
        invocations = 1
        if remaining_depth == 0:
            return invocations
        return invocations + sum(
            count(child, remaining_depth - 1) for child in split_result_block(node)
        )

    return count(block, max_recovery_depth)
