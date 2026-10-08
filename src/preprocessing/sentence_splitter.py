"""Document-level sentence splitting for Module 2.

This stage adds sentence-level location and classification metadata only.  It
does not clean pages, tokenize sentences, or create final SentenceRecords.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from .cleaning import find_references_start
from .cleaning import clean_pages
from .tokenizer import normalize_tokens, tokenize

if TYPE_CHECKING:
    from src.utils.schemas import PageRecord
    from src.utils.schemas import SentenceRecord


_CAPTION_LINE_RE = re.compile(
    r"(?i)^[ \t]*(?:figure|fig\.|table)[ \t]+"
    r"\d+(?:\.\d+)*[a-z]?[ \t]*(?:[.:](?=\s|$)|$)"
)
_PANEL_LABEL_LINE_RE = re.compile(r"(?i)^(?:\([a-z]\)[ \t]*)+$")


def _value(page: Any, key: str) -> Any:
    if isinstance(page, Mapping):
        return page.get(key)
    return getattr(page, key, None)


def _validate_pages(pages: Sequence[Any]) -> None:
    if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence):
        raise ValueError("pages must be a sequence of PageRecord objects")
    required = ("document_id", "filename", "page_number", "text")
    for index, page in enumerate(pages):
        if not isinstance(page, Mapping) and not hasattr(page, "text"):
            raise ValueError(f"malformed page at index {index}")
        if any(_value(page, key) is None for key in required):
            raise ValueError(f"malformed page at index {index}")
        if not isinstance(_value(page, "page_number"), int):
            raise ValueError(f"page number at index {index} must be an integer")
        if not isinstance(_value(page, "text"), str):
            raise ValueError(f"page text at index {index} must be a string")


def _config_value(cfg: dict, key: str, default: Any) -> Any:
    """Read processing settings, with a root-level compatibility fallback."""

    processing = cfg.get("processing", {})
    if isinstance(processing, Mapping) and key in processing:
        return processing[key]
    return cfg[key] if key in cfg else default


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return sentence spans from NLTK's Punkt tokenizer.

    Importing lazily keeps this module importable for static inspection when
    the optional runtime dependency is not installed.
    """

    try:
        from nltk.tokenize.punkt import PunktTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "sentence_splitter requires NLTK; install the project's declared "
            "dependencies before running sentence splitting"
        ) from exc

    try:
        return list(PunktTokenizer().span_tokenize(text))
    except LookupError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "NLTK Punkt data is unavailable; download punkt and punkt_tab "
            "using the project's setup instructions"
        ) from exc


def _document_pages(pages: list[PageRecord]) -> list[list[PageRecord]]:
    """Group pages by document in first-seen document order and page order."""

    grouped: dict[Any, list[tuple[int, PageRecord]]] = {}
    for index, page in enumerate(pages):
        grouped.setdefault(_value(page, "document_id"), []).append((index, page))
    return [
        [page for _, page in sorted(items, key=lambda item: (_value(item[1], "page_number"), item[0]))]
        for items in grouped.values()
    ]


def _build_stream(
    pages: list[PageRecord],
) -> tuple[str, list[tuple[int, int, int]]]:
    """Join pages and return ``(stream, start, end, page_number)`` spans."""

    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    offset = 0
    for page_index, page in enumerate(pages):
        if page_index:
            # Keep a line boundary so the existing References heading detector
            # can recognize a heading at the start of any page.
            parts.append("\n")
            offset += 1
        text = _value(page, "text")
        start = offset
        parts.append(text)
        offset += len(text)
        page_spans.append((start, offset, _value(page, "page_number")))
    return "".join(parts), page_spans


def _page_for_position(
    position: int,
    page_spans: list[tuple[int, int, int]],
) -> int:
    for start, end, page_number in page_spans:
        if start <= position < end:
            return page_number
        if position < start:
            # The document stream inserts one newline between page texts. If
            # Punkt includes that separator at the start of a span, associate
            # it with the following page rather than the document's last page.
            return page_number
    return page_spans[-1][2]


def _is_caption(text: str) -> bool:
    """Return whether text begins with a numbered figure/table caption label."""

    return bool(_CAPTION_LINE_RE.match(text))


def _is_panel_label_line(text: str) -> bool:
    """Recognize a line made only of figure-panel labels, not prose parentheses."""

    return bool(_PANEL_LABEL_LINE_RE.fullmatch(text.strip()))


def _caption_line_is_unfinished(text: str) -> bool:
    """Return whether layout text strongly indicates the caption wraps onward."""

    stripped = text.rstrip()
    if re.search(r"\([a-z]\)\s*$", stripped, re.IGNORECASE):
        return True
    if stripped.endswith((",", ":", ";", "-", "–")):
        return True
    if stripped.count("(") > stripped.count(")") or stripped.count("[") > stripped.count("]"):
        return True
    return bool(re.search(r"(?i)\b(?:and|or|of|the|a|an|to|with|from|for)$", stripped))


def _caption_spans(
    pages: list[PageRecord],
    page_spans: list[tuple[int, int, int]],
) -> list[tuple[int, int, int, int, bool]]:
    """Locate caption lines in the original cleaned stream.

    Detection is restricted to numbered labels at original line starts. Each
    matching line is retained as its own region so preceding chart debris and
    adjacent table content are not classified as caption text.
    """

    spans: list[tuple[int, int, int, int, bool]] = []
    for page, (page_start, _, _) in zip(pages, page_spans):
        page_text = _value(page, "text")
        lines = page_text.splitlines(keepends=True)
        line_offset = 0
        line_starts: list[int] = []
        current_offset = 0
        for line in lines:
            line_starts.append(current_offset)
            current_offset += len(line)

        for line_index, line in enumerate(lines):
            line_text = line.rstrip("\r\n")
            match = _CAPTION_LINE_RE.match(line_text)
            if match:
                # A colon-style prefix remains part of the existing Punkt
                # sentence text. Period-style labels are consumed as
                # abbreviations, so only their description is sentence text.
                delimiter = line_text[match.end() - 1 : match.end()]
                marker_only = delimiter not in (":", ".")
                body_line_index = line_index + 1 if marker_only else line_index
                if body_line_index >= len(lines):
                    line_offset += len(line)
                    continue

                if marker_only:
                    content_offset = line_starts[body_line_index]
                elif delimiter == ":":
                    content_offset = line_offset
                else:
                    content_offset = line_offset + match.end()

                body_line = lines[body_line_index]
                body_text = body_line.rstrip("\r\n")
                # A bare "Table 3" can be a table cell in extracted layouts
                # (for example, a provenance column). Require a substantial
                # following line before treating a marker-only line as a
                # caption block.
                if marker_only and len(body_text.strip()) < 20:
                    line_offset += len(line)
                    continue
                first_line_end = line_starts[body_line_index] + len(body_text)
                caption_end = first_line_end
                previous_text = body_text.rstrip()
                next_index = body_line_index + 1
                follows_panel_marker = False
                while next_index < len(lines):
                    continuation_text = lines[next_index].rstrip("\r\n")
                    continuation = continuation_text.strip()
                    first_character = next(
                        (character for character in continuation if character.isalpha()),
                        "",
                    )
                    continues_panel_labels = bool(
                        re.search(r"\([a-z]\)\s*$", previous_text, re.IGNORECASE)
                    )
                    if (
                        not continuation
                        or _is_panel_label_line(continuation_text)
                        or previous_text.endswith((".", "!", "?"))
                        or not first_character
                        or not (
                            first_character.islower()
                            or continues_panel_labels
                            or _caption_line_is_unfinished(previous_text)
                        )
                    ):
                        break
                    follows_panel_marker = follows_panel_marker or continues_panel_labels
                    caption_end += len(lines[next_index])
                    # A lower-case line can wrap a caption and then continue
                    # with ordinary prose on that same extracted line. Retain
                    # only the caption sentence(s). Punkt may split inside an
                    # unfinished parenthetical such as ``(e.g. ... )``, so do
                    # not stop at a boundary until the prefix is complete.
                    if first_character.islower():
                        continuation_spans = _sentence_spans(continuation)
                        if len(continuation_spans) > 1:
                            boundary_index = 0
                            while boundary_index < len(continuation_spans) - 1:
                                candidate_end = continuation_spans[boundary_index][1]
                                if not _caption_line_is_unfinished(
                                    continuation[:candidate_end].rstrip()
                                ):
                                    break
                                boundary_index += 1
                            continuation_end = continuation_spans[boundary_index][1]
                            if continuation_end < len(continuation):
                                caption_end = line_starts[next_index] + continuation_end
                                break
                    previous_text = continuation_text.rstrip()
                    next_index += 1
                spans.append(
                    (
                        page_start + line_offset,
                        page_start + content_offset,
                        page_start + first_line_end,
                        page_start + caption_end,
                        follows_panel_marker,
                    )
                )
            line_offset += len(line)
    return spans


def _sentence_spans_with_captions(
    stream: str,
    caption_spans: list[tuple[int, int, int, int, bool]],
) -> list[tuple[int, int]]:
    """Split caption source regions without retokenizing unrelated text.

    Existing Punkt spans are preserved except where a detected caption starts
    or ends. The numbered label is layout metadata, so it is excluded from
    sentence text; caption descriptions are Punkt-tokenized independently.
    """

    spans: list[tuple[int, int]] = []
    for span_start, span_end in _sentence_spans(stream):
        boundaries = {span_start, span_end}
        for caption_start, content_start, first_line_end, caption_end, _ in caption_spans:
            boundaries.update(
                position
                for position in (caption_start, content_start, first_line_end, caption_end)
                if span_start < position < span_end
            )
        ordered = sorted(boundaries)
        for start, end in zip(ordered, ordered[1:]):
            if start >= end:
                continue
            region = next(
                (
                    (content_start, first_line_end, caption_end)
                    for _, content_start, first_line_end, caption_end, _ in caption_spans
                    if content_start <= start < caption_end
                ),
                None,
            )
            if any(caption_start <= start < content_start for caption_start, content_start, _, _, _ in caption_spans):
                continue
            if region is None:
                spans.append((start, end))
                continue
            spans.extend(
                (start + sentence_start, start + sentence_end)
                for sentence_start, sentence_end in _sentence_spans(stream[start:end])
            )
    return spans


def split_sentences(pages: list[PageRecord], cfg: dict) -> list[dict]:
    """Split each document stream into sentences with page metadata.

    Sentence indices are zero-based and contiguous within each document.
    Length filtering uses character counts after segmentation and defaults to
    the project convention of 5 through 500 characters inclusive.
    """

    _validate_pages(pages)
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dictionary")
    if not pages:
        return []

    try:
        minimum = int(_config_value(cfg, "min_sentence_length", 5))
        maximum = int(_config_value(cfg, "max_sentence_length", 500))
    except (TypeError, ValueError) as exc:
        raise ValueError("sentence length limits must be integers") from exc
    if minimum < 0 or maximum < minimum:
        raise ValueError("sentence length limits must satisfy 0 <= min <= max")

    output: list[dict] = []
    for document_pages in _document_pages(pages):
        stream, page_spans = _build_stream(document_pages)
        caption_spans = _caption_spans(document_pages, page_spans)
        reference_offset = None
        references_start = find_references_start(document_pages)
        if references_start is not None:
            reference_page_index, local_offset = references_start
            reference_offset = page_spans[reference_page_index][0] + local_offset
        sentence_index = 0
        for start, end in _sentence_spans_with_captions(stream, caption_spans):
            text = re.sub(r"\s+", " ", stream[start:end]).strip()
            if not text or not minimum <= len(text) <= maximum:
                continue
            is_caption = any(
                start < caption_end
                and end > content_start
                and (
                    start < original_line_end
                    # An expanded source span is established from adjacent
                    # original caption lines before Punkt runs. Trust every
                    # sentence inside that bounded block, including an
                    # uppercase fragment after an abbreviation or parenthesis.
                    or caption_end > original_line_end
                    or next((character for character in text if character.isalpha()), "").islower()
                    or follows_panel_marker
                )
                for _, content_start, original_line_end, caption_end, follows_panel_marker in caption_spans
            )
            output.append(
                {
                    "text": text,
                    "page_number": _page_for_position(start, page_spans),
                    "page_end": _page_for_position(end - 1, page_spans),
                    "sentence_index": sentence_index,
                    "in_references": (
                        reference_offset is not None and start >= reference_offset
                    ),
                    "is_caption": is_caption or _is_caption(text),
                }
            )
            sentence_index += 1
    return output


def build_sentence_records(
    pages: list[PageRecord], cfg: dict
) -> list[SentenceRecord]:
    """Build deterministic SentenceRecord-shaped dictionaries for a corpus.

    Page cleaning is performed once, then each document is split as a complete
    stream.  The repository's shared schema is not available in this scaffold,
    so the returned dictionaries use exactly the documented SentenceRecord
    fields and remain compatible with a later TypedDict/dataclass schema.
    """

    _validate_pages(pages)
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dictionary")
    if not pages:
        return []

    cleaned = clean_pages(pages, cfg)
    preprocessing = cfg.get("preprocessing", {})
    if preprocessing is None:
        preprocessing = {}
    if not isinstance(preprocessing, Mapping):
        raise ValueError("cfg['preprocessing'] must be a dictionary")
    lowercase = preprocessing.get("lowercase", True)
    remove_stopwords = preprocessing.get("remove_stopwords", False)

    records: list[dict[str, Any]] = []
    for document_pages in _document_pages(cleaned):
        document_id = _value(document_pages[0], "document_id")
        document_name = _value(document_pages[0], "filename")
        sentences = split_sentences(document_pages, cfg)
        for sentence in sentences:
            tokens = tokenize(sentence["text"])
            normalized_tokens = normalize_tokens(
                tokens,
                lowercase=lowercase,
                remove_stopwords=remove_stopwords,
            )
            sentence_index = sentence["sentence_index"]
            records.append(
                {
                    "sentence_id": f"{document_id}_s{sentence_index:05d}",
                    "document_id": document_id,
                    "document_name": document_name,
                    "page_number": sentence["page_number"],
                    "page_end": sentence["page_end"],
                    "sentence_index": sentence_index,
                    "text": sentence["text"],
                    "tokens": tokens,
                    "normalized_tokens": normalized_tokens,
                    "in_references": sentence["in_references"],
                    "is_caption": sentence["is_caption"],
                }
            )
    return records
