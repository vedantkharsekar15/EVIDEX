"""Tokenizer-focused tests for Module 2."""

from src.preprocessing.tokenizer import (
    normalize_tokens,
    preprocess_query,
    tokenize,
)
from src.preprocessing.sentence_splitter import build_sentence_records, split_sentences


def test_tokenize_preserves_case_and_punctuation() -> None:
    assert tokenize("The Model improves accuracy.") == [
        "The",
        "Model",
        "improves",
        "accuracy",
        ".",
    ]


def test_tokenize_empty_and_whitespace_input() -> None:
    assert tokenize("") == []
    assert tokenize(" \t\n ") == []


def test_normalize_tokens_lowercases_without_removing_punctuation() -> None:
    tokens = tokenize("The Model improves accuracy!")
    assert normalize_tokens(tokens) == ["the", "model", "improves", "accuracy", "!"]


def test_normalize_tokens_can_preserve_case() -> None:
    assert normalize_tokens(["The", "Model", "."], lowercase=False) == [
        "The",
        "Model",
        ".",
    ]


def test_stopword_option_is_explicit() -> None:
    tokens = tokenize("The model is in the lab.")
    assert normalize_tokens(tokens, remove_stopwords=False) == [
        "the",
        "model",
        "is",
        "in",
        "the",
        "lab",
        ".",
    ]
    assert normalize_tokens(tokens, remove_stopwords=True) == ["model", "lab", "."]


def test_normalization_does_not_strip_punctuation() -> None:
    assert normalize_tokens(["RAG", "-", "based", ",", "!"]) == [
        "rag",
        "-",
        "based",
        ",",
        "!",
    ]


def test_preprocess_query_matches_corpus_path() -> None:
    query = "Foo Bar"
    cfg = {"preprocessing": {"lowercase": True, "remove_stopwords": False}}
    assert preprocess_query(query, cfg) == normalize_tokens(tokenize(query))


def test_preprocess_query_respects_lowercase_setting() -> None:
    cfg = {"preprocessing": {"lowercase": False, "remove_stopwords": False}}
    assert preprocess_query("Foo Bar", cfg) == ["Foo", "Bar"]


def test_preprocess_query_respects_stopword_setting() -> None:
    cfg = {"preprocessing": {"lowercase": True, "remove_stopwords": True}}
    assert preprocess_query("The Model is ready.", cfg) == ["model", "ready", "."]


def test_preprocess_query_is_deterministic() -> None:
    cfg = {"preprocessing": {"lowercase": True, "remove_stopwords": False}}
    assert preprocess_query("RAG-based retrieval 95.4%", cfg) == preprocess_query(
        "RAG-based retrieval 95.4%", cfg
    )


def test_scientific_and_numeric_text_is_safe() -> None:
    tokens = tokenize("The state-of-the-art model reached 95.4% accuracy.")
    assert tokens
    assert "95.4" in tokens
    assert "%" in tokens
    assert "state-of-the-art" in tokens


def _page(document_id: str, page_number: int, text: str) -> dict:
    return {
        "document_id": document_id,
        "filename": f"{document_id}.pdf",
        "page_number": page_number,
        "text": text,
    }


def _sentence_cfg(**processing: int) -> dict:
    return {"processing": processing}


def test_split_sentences_empty_input() -> None:
    assert split_sentences([], {}) == []


def test_split_sentences_rejects_malformed_pages() -> None:
    malformed = [{"document_id": "doc", "filename": "doc.pdf", "page_number": 1}]
    try:
        split_sentences(malformed, {})
    except ValueError:
        pass
    else:
        raise AssertionError("malformed pages must raise ValueError")


def test_split_sentences_orders_sentences_and_indices() -> None:
    pages = [_page("doc", 1, "First sentence. Second sentence."), _page("doc", 2, "Third sentence.")]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=100))
    assert [item["text"] for item in result] == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    assert [item["sentence_index"] for item in result] == [0, 1, 2]


def test_split_sentences_maps_cross_page_sentence() -> None:
    pages = [_page("doc", 1, "The proposed framework significantly improves"), _page("doc", 2, "retrieval accuracy on scientific documents.")]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert len(result) == 1
    assert result[0]["text"] == "The proposed framework significantly improves retrieval accuracy on scientific documents."
    assert result[0]["page_number"] == 1
    assert result[0]["page_end"] == 2


def test_split_sentences_references_continue_across_pages() -> None:
    pages = [
        _page("doc", 1, "A result is reported."),
        _page("doc", 2, "References\n[1] First reference."),
        _page("doc", 3, "[2] Second reference."),
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert result[0]["in_references"] is False
    assert all(item["in_references"] for item in result[1:])
    assert result[-1]["page_number"] == 3


def test_split_sentences_marks_figure_and_table_captions() -> None:
    pages = [_page("doc", 1, "Figure 1: System architecture. Table 2: Validation results.")]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert [item["is_caption"] for item in result] == [True, True]


def test_split_sentences_marks_period_style_caption_lines() -> None:
    pages = [
        _page(
            "doc",
            1,
            "Fig. 1. Data curation and species-disjoint split.\n"
            "Fig. 2. Cross-patient memory fusion architecture.\n"
            "Table 1. Tabular summary of the methods.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert len(result) == 3
    assert all(item["is_caption"] for item in result)


def test_split_sentences_recognizes_uppercase_standalone_markers() -> None:
    pages = [
        _page(
            "doc",
            1,
            "FIGURE 1\nRate of species discovery in the ocean. Data were collected annually.\n"
            "TABLE 1\nSummary of the observed measurements.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert [item["text"] for item in result] == [
        "Rate of species discovery in the ocean.",
        "Data were collected annually.",
        "Summary of the observed measurements.",
    ]
    assert [item["is_caption"] for item in result] == [True, True, True]


def test_wrapped_caption_continuation_can_start_uppercase() -> None:
    pages = [
        _page(
            "doc",
            1,
            "Figure 2. Radiographs from Dataset (a) normal (b)\n"
            "Radiograph of a bacterial case (c) Radiograph of a viral case.\n"
            "The next paragraph reports the experiment.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    caption_sentences = [item for item in result if "Radiograph" in item["text"]]
    following = [item for item in result if item["text"].startswith("The next paragraph")]
    assert len(caption_sentences) == 2
    assert all(item["is_caption"] for item in caption_sentences)
    assert len(following) == 1 and following[0]["is_caption"] is False


def test_ocean_figure_three_caption_keeps_punkt_fragment_in_source_block() -> None:
    pages = [
        _page(
            "doc",
            1,
            "FIGURE 3\n"
            "A summary of the approaches. We note that the methods benefit studies\n"
            "of evolutionary taxonomy, including examples (e.g. New Ways of Working and Standardisation).\n"
            "Together these represent the Ocean Census integrated taxonomic strategy.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=300))
    fragment = [item for item in result if "New Ways of Working" in item["text"]]
    following = [item for item in result if item["text"].startswith("Together these")]
    assert fragment and all(item["is_caption"] for item in fragment)
    assert following and all(item["is_caption"] is False for item in following)


def test_ocean_figure_four_caption_keeps_parenthetical_continuation() -> None:
    pages = [
        _page(
            "doc",
            1,
            "FIGURE 4\n"
            "Diagram depicting the workflow (Digitisation, Adoption of New Technologies,\n"
            "Capacity Development, Expeditions). Image of MinION sequencer is from Oxford Nanopore Technologies.\n"
            "The following paragraph discusses the study methods.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=300))
    continuation = [item for item in result if "Capacity Development" in item["text"]]
    source_note = [item for item in result if "Image of MinION" in item["text"]]
    following = [item for item in result if item["text"].startswith("The following paragraph")]
    assert continuation and all(item["is_caption"] for item in continuation)
    assert source_note and all(item["is_caption"] for item in source_note)
    assert following and all(item["is_caption"] is False for item in following)


def test_standalone_panel_labels_are_not_caption_continuations() -> None:
    pages = [
        _page(
            "doc",
            1,
            "Figure 1. Methodology overview\n(a) (b) (c)\n"
            "Figure 2. Caption text with an (a) note.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    panel = [item for item in result if item["text"] == "(a) (b) (c)"]
    assert len(panel) == 1 and panel[0]["is_caption"] is False
    assert [item["is_caption"] for item in result if "Methodology overview" in item["text"]] == [True]
    assert [item["is_caption"] for item in result if "Caption text" in item["text"]] == [True]


def test_short_table_cell_after_bare_number_is_not_a_caption() -> None:
    pages = [_page("doc", 1, "Previous sentence.\nTable 3\nTerminal log\nNext sentence.")]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert not any(item["is_caption"] for item in result)


def test_multi_sentence_caption_marks_each_sentence() -> None:
    pages = [
        _page(
            "doc",
            1,
            "Fig. 2. Cross-patient memory fusion architecture. "
            "Solid arrows show query-time evidence.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert len(result) == 2
    assert [item["is_caption"] for item in result] == [True, True]


def test_wrapped_caption_continuation_is_marked_without_marking_following_prose() -> None:
    pages = [
        _page(
            "doc",
            1,
            "Fig. 2. Cross-patient memory fusion architecture. "
            "Solid arrows show query-time evidence; dashed\n"
            "arrows show construction from patient-disjoint source memory. "
            "The residual cache is a separate evaluated variant.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    caption = [item for item in result if "arrows show construction" in item["text"]]
    following_prose = [item for item in result if "The residual cache" in item["text"]]
    assert caption and all(item["is_caption"] for item in caption)
    assert following_prose and all(item["is_caption"] is False for item in following_prose)


def test_caption_flag_survives_consumed_marker_and_record_building() -> None:
    pages = [
        _page(
            "doc",
            1,
            "S k k k Val Trustworthiness, Trust Value,\n"
            "Figure 1: Proposed layered trust model.",
        )
    ]
    cfg = _sentence_cfg(min_sentence_length=1, max_sentence_length=200)
    sentences = split_sentences(pages, cfg)
    assert len(sentences) == 2
    assert sentences[0]["is_caption"] is False
    assert sentences[1]["is_caption"] is True
    assert sentences[1]["text"] == "Figure 1: Proposed layered trust model."

    records = build_sentence_records(pages, cfg)
    caption_records = [record for record in records if "Proposed layered trust model" in record["text"]]
    assert len(caption_records) == 1
    assert caption_records[0]["is_caption"] is True
    assert caption_records[0]["sentence_id"] == "doc_s00001"
    assert caption_records[0]["sentence_index"] == 1
    assert caption_records[0]["page_number"] == caption_records[0]["page_end"] == 1
    assert records == build_sentence_records(pages, cfg)


def test_caption_detection_does_not_mark_ordinary_figure_or_table_prose() -> None:
    pages = [
        _page(
            "doc",
            1,
            "As shown in Figure 3, the model improves retrieval.\n"
            "Figure 3 demonstrates that the model is robust.\n"
            "Table 2 shows the results across datasets.\n"
            "The figure illustrates the proposed architecture.",
        )
    ]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert len(result) == 4
    assert all(item["is_caption"] is False for item in result)


def test_split_sentences_does_not_mark_ordinary_figure_sentence() -> None:
    pages = [_page("doc", 1, "Figure 2 shows the performance improvement.")]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=200))
    assert result[0]["is_caption"] is False


def test_split_sentences_applies_length_limits() -> None:
    pages = [_page("doc", 1, "Hi. This fits. This sentence is definitely too long.")]
    minimum_result = split_sentences(
        pages, _sentence_cfg(min_sentence_length=10, max_sentence_length=100)
    )
    assert [item["text"] for item in minimum_result] == ["This fits.", "This sentence is definitely too long."]
    maximum_result = split_sentences(
        pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=20)
    )
    assert [item["text"] for item in maximum_result] == ["Hi.", "This fits."]


def test_split_sentences_has_independent_indices_per_document() -> None:
    pages = [_page("one", 1, "First document sentence."), _page("two", 1, "Second document sentence.")]
    result = split_sentences(pages, _sentence_cfg(min_sentence_length=1, max_sentence_length=100))
    assert [item["sentence_index"] for item in result] == [0, 0]


def test_split_sentences_is_deterministic_on_fixture() -> None:
    import json
    from pathlib import Path

    fixture = json.loads(Path("tests/fixtures/pages_preproc.json").read_text(encoding="utf-8"))
    cfg = _sentence_cfg(min_sentence_length=1, max_sentence_length=500)
    assert split_sentences(fixture, cfg) == split_sentences(fixture, cfg)


def test_build_sentence_records_empty_input() -> None:
    assert build_sentence_records([], {}) == []


def test_build_sentence_records_populates_contract_and_tokens() -> None:
    pages = [
        _page("doc_one", 1, "The Model improves"),
        _page("doc_one", 2, "retrieval accuracy. Figure 1: System architecture."),
    ]
    cfg = {
        "processing": {"min_sentence_length": 1, "max_sentence_length": 500},
        "preprocessing": {"lowercase": True, "remove_stopwords": True},
    }
    records = build_sentence_records(pages, cfg)
    assert records[0]["sentence_id"] == "doc_one_s00000"
    assert records[0]["document_id"] == "doc_one"
    assert records[0]["document_name"] == "doc_one.pdf"
    assert records[0]["sentence_index"] == 0
    assert records[0]["page_number"] == 1
    assert records[0]["page_end"] == 2
    assert records[0]["text"] == "The Model improves retrieval accuracy."
    assert records[0]["tokens"][:2] == ["The", "Model"]
    assert records[0]["normalized_tokens"][:2] == ["model", "improves"]
    assert records[0]["in_references"] is False
    assert records[1]["is_caption"] is True


def test_build_sentence_records_propagates_references_and_document_indices() -> None:
    pages = [
        _page("doc_one", 1, "A result is reported."),
        _page("doc_two", 1, "References\n[1] First reference."),
        _page("doc_two", 2, "[2] Second reference."),
    ]
    cfg = _sentence_cfg(min_sentence_length=1, max_sentence_length=500)
    first = build_sentence_records(pages, cfg)
    second = build_sentence_records(pages, cfg)
    assert first == second
    assert first[0]["sentence_id"] == "doc_one_s00000"
    assert first[1]["sentence_id"] == "doc_two_s00000"
    assert first[1]["in_references"] is True
    assert first[2]["in_references"] is True
    assert [item["sentence_index"] for item in first] == [0, 0, 1]
