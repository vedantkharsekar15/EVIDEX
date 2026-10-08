"""Cleaning helpers for raw document pages.

The functions in this module operate on the shared ``PageRecord`` contract
(``document_id``, ``filename``, ``page_number``, and ``text``).  Cleaning is
kept page-aware so that later sentence splitting can recover page boundaries.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utils.schemas import PageRecord


_PAGE_NUMBER_RE = re.compile(r"(?m)^[ \t]*\d+[ \t]*\n?")
_LINE_HYPHEN_RE = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")


def normalize_unicode(text: str) -> str:
    """Apply a stable compatibility Unicode normalization to ``text``.

    NFC normalizes canonically equivalent forms while retaining scientific
    symbols such as ``Σ`` and subscripts rather than deleting them.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Normalize line whitespace while retaining paragraph and line breaks."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        lines.append(line)
    # Keep one blank line as a meaningful paragraph separator, but avoid
    # allowing extraction noise to create long runs of empty lines.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def dehyphenate(text: str) -> str:
    """Join a word split by a hyphen at a line break.

    Only a hyphen immediately followed by a line break and word characters on
    both sides is changed; ordinary hyphenated words remain untouched.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _LINE_HYPHEN_RE.sub("", text)


def remove_page_numbers(text: str) -> str:
    """Remove lines containing only an Arabic page number."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = _PAGE_NUMBER_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _record_value(page: Any, key: str) -> Any:
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
        if any(_record_value(page, key) is None for key in required):
            raise ValueError(f"malformed page at index {index}")
        if not isinstance(_record_value(page, "text"), str):
            raise ValueError(f"page text at index {index} must be a string")
        if not isinstance(_record_value(page, "page_number"), int):
            raise ValueError(f"page number at index {index} must be an integer")


def _boundary_lines(text: str) -> set[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return set()
    return {lines[0], lines[-1]}


def _with_text(page: Any, text: str) -> Any:
    """Return a page copy with only its text replaced."""

    if isinstance(page, Mapping):
        updated = dict(page)
        updated["text"] = text
        return updated
    # PageRecord is normally a mapping/TypedDict.  This fallback supports a
    # simple object-shaped implementation without adding any fields.
    import copy

    updated = copy.copy(page)
    setattr(updated, "text", text)
    return updated


def remove_repeated_headers_footers(
    pages: list[PageRecord], min_page_ratio: float = 0.5
) -> list[PageRecord]:
    """Remove repeated non-empty boundary lines while preserving page order.

    A line is considered a header/footer candidate only when it is the first or
    last non-empty line on a page.  This avoids deleting repeated body phrases.
    The threshold is measured within each document, so unrelated documents do
    not influence one another.
    """

    _validate_pages(pages)
    if not 0 < min_page_ratio <= 1:
        raise ValueError("min_page_ratio must be greater than 0 and at most 1")

    result: list[Any] = []
    for document_id in dict.fromkeys(_record_value(page, "document_id") for page in pages):
        indices = [i for i, page in enumerate(pages) if _record_value(page, "document_id") == document_id]
        counts: dict[str, int] = {}
        for index in indices:
            for line in _boundary_lines(_record_value(pages[index], "text")):
                counts[line] = counts.get(line, 0) + 1
        threshold = math.ceil(len(indices) * min_page_ratio)
        repeated = {
            line
            for line, count in counts.items()
            if count >= 2 and count >= threshold
        }
        for index in indices:
            page = pages[index]
            text = _record_value(page, "text")
            kept = [line for line in text.splitlines() if line.strip() not in repeated]
            result.append((index, _with_text(page, "\n".join(kept).strip())))

    return [page for _, page in sorted(result, key=lambda item: item[0])]


def find_references_start(pages: list[PageRecord]) -> tuple[int, int] | None:
    """Return the page index and character offset of a References heading."""

    _validate_pages(pages)
    heading_re = re.compile(
        r"(?im)^[ \t]*(?:references|bibliography)[ \t]*:?[ \t]*$"
    )
    for page_index, page in enumerate(pages):
        match = heading_re.search(_record_value(page, "text"))
        if match:
            return page_index, match.start()
    return None


def alpha_ratio(text: str) -> float:
    """Return alphabetic-character count divided by total character count."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return sum(character.isalpha() for character in text) / len(text) if text else 0.0


def _config_value(cfg: dict, key: str, default: Any) -> Any:
    """Read a setting from processing, preprocessing, then root scope.

    ``processing`` is checked first because it is the canonical scope for
    pipeline-wide thresholds such as ``min_alpha_ratio``.  A present value is
    returned unchanged rather than being replaced by a module default.
    """

    if not isinstance(cfg, dict):
        return default
    for scope in ("processing", "preprocessing"):
        values = cfg.get(scope)
        if isinstance(values, Mapping) and key in values:
            return values[key]
    return cfg[key] if key in cfg else default


def _filter_debris(
    text: str,
    min_alpha_ratio: float,
    references_offset: int | None,
    preserve_all: bool = False,
) -> str:
    """Drop sparse lines before References while preserving reference text."""

    kept: list[str] = []
    offset = 0
    for line in text.splitlines():
        line_end = offset + len(line)
        in_references = preserve_all or (
            references_offset is not None and offset >= references_offset
        )
        if in_references or not line.strip() or alpha_ratio(line) >= min_alpha_ratio:
            kept.append(line)
        offset = line_end + 1
    return "\n".join(kept).strip()


def clean_pages(pages: list[PageRecord], cfg: dict) -> list[PageRecord]:
    """Apply page-level normalization and extraction-debris filtering.

    The returned records contain the original metadata and cleaned ``text``;
    no sentence-level records or additional fields are created.
    """

    _validate_pages(pages)
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dictionary")

    normalized: list[Any] = []
    for page in pages:
        text = normalize_whitespace(normalize_unicode(_record_value(page, "text")))
        normalized.append(_with_text(page, text))

    without_headers = remove_repeated_headers_footers(
        normalized,
        float(_config_value(cfg, "min_page_ratio", 0.5)),
    )
    # The default removes sparse extraction artifacts such as equations while
    # retaining ordinary prose and short scientific labels.
    min_alpha = float(_config_value(cfg, "min_alpha_ratio", 0.25))
    if not 0 <= min_alpha <= 1:
        raise ValueError("min_alpha_ratio must be between 0 and 1")

    prepared = [
        _with_text(
            page,
            normalize_whitespace(
                dehyphenate(remove_page_numbers(_record_value(page, "text")))
            ),
        )
        for page in without_headers
    ]

    cleaned_by_index: dict[int, Any] = {}
    for document_id in dict.fromkeys(
        _record_value(page, "document_id") for page in prepared
    ):
        document_indices = [
            index
            for index, page in enumerate(prepared)
            if _record_value(page, "document_id") == document_id
        ]
        document_pages = [prepared[index] for index in document_indices]
        references_start = find_references_start(document_pages)
        reference_page_index = references_start[0] if references_start else None
        reference_offset = references_start[1] if references_start else None
        for page_index, page in enumerate(document_pages):
            local_offset = (
                reference_offset if page_index == reference_page_index else None
            )
            preserve_all = (
                reference_page_index is not None
                and page_index > reference_page_index
            )
            text = _filter_debris(
                _record_value(page, "text"),
                min_alpha,
                local_offset,
                preserve_all,
            )
            cleaned_by_index[document_indices[page_index]] = _with_text(page, text)

    return [cleaned_by_index[index] for index in range(len(prepared))]
