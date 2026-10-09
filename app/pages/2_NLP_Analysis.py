"""EVIDEX Module 4: NLP Analysis Streamlit Page.

Provides interactive inspection of:
- N-Gram Language Model (unigrams, bigrams, trigrams, next-word predictions, perplexity)
- HMM POS Tagger (Viterbi decoding, Treebank held-out accuracy, exploratory disclaimer)
- Corpus Term Frequencies (ranked table, reference exclusion, stopword filter)
- Word Cloud Visualization
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis.frequency import frequency_table, term_frequencies
from src.analysis.wordcloud import build_wordcloud
from src.language_model.ngram_model import NGramModel
from src.pos.hmm_tagger import HMMTagger, MissingNLTKDataError

# Tag description dictionary for Penn Treebank tags
PENN_TAG_DESCRIPTIONS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition / subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal verb",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
    ".": "Sentence-final punctuation",
    ",": "Comma",
    ":": "Colon / semi-colon",
    "(": "Left parenthesis",
    ")": "Right parenthesis",
    "''": "Closing quotation mark",
    "``": "Opening quotation mark",
    "$": "Dollar sign",
    "#": "Pound sign",
}


@st.cache_resource(show_spinner="Training HMM POS Tagger on Penn Treebank...")
def get_cached_hmm_tagger() -> tuple[HMMTagger, float]:
    """Train or load the HMM Tagger on Penn Treebank with fixed seed for determinism."""
    tagger = HMMTagger()
    try:
        acc = tagger.train_from_treebank(test_split=0.1, seed=42)
    except MissingNLTKDataError:
        acc = 0.0
    return tagger, acc


def load_default_fixture_records() -> list[dict]:
    """Load default NLP sentence fixture records for demonstration."""
    fixture_path = (
        Path(__file__).resolve().parent.parent.parent
        / "tests"
        / "fixtures"
        / "sentences_nlp.json"
    )
    if fixture_path.exists():
        with open(fixture_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def render_nlp_analysis_page() -> None:
    """Render the main NLP Analysis Streamlit page."""
    st.set_page_config(page_title="EVIDEX — NLP Analysis", page_icon="🔬", layout="wide")

    st.title("🔬 NLP Analysis & Language Modeling")
    st.caption("Module 4: Statistical N-Grams, HMM POS Tagger, Term Frequencies & Word Cloud")

    # Retrieve corpus from session state or load fallback fixture
    corpus_data = st.session_state.get("corpus", None)
    records: list[dict] = []

    if corpus_data and isinstance(corpus_data, dict):
        records = corpus_data.get("records", [])
        ngram_model = corpus_data.get("ngram_model", None)
    else:
        records = load_default_fixture_records()
        ngram_model = None

    if not records:
        st.warning(
            "⚠️ No corpus records found in session state. Please upload research papers on the Home page "
            "or use the test dataset."
        )
        return

    # Non-reference sentences for language modeling
    valid_sentences = [
        r["normalized_tokens"]
        for r in records
        if not r.get("in_references", False) and r.get("normalized_tokens")
    ]

    # Initialize / fit NGram model if not cached
    if ngram_model is None:
        ngram_model = NGramModel(n_values=(1, 2, 3), k=0.1).fit(valid_sentences)

    # Initialize HMM POS tagger
    hmm_tagger, hmm_accuracy = get_cached_hmm_tagger()

    # Tabs for different NLP components
    tab1, tab2, tab3 = st.tabs(
        ["📊 N-Gram Language Model", "🏷️ HMM POS Tagging", "☁️ Term Frequencies & Word Cloud"]
    )

    # ==========================================
    # TAB 1: N-Gram Language Model
    # ==========================================
    with tab1:
        st.header("Statistical N-Gram Language Model")
        st.markdown(
            "An in-house statistical language model fitted over normalized corpus sentences "
            "with sentence boundary markers (`<s>`, `</s>`) and **Add-k (Lidstone) smoothing**."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Vocabulary Size", f"{len(ngram_model.vocab):,}")
        col2.metric("Total Unigrams", f"{ngram_model.total_unigrams:,}")
        col3.metric("Smoothing Parameter ($k$)", f"{ngram_model.k}")
        pp_val = ngram_model.perplexity(valid_sentences)
        col4.metric("Corpus Perplexity", f"{pp_val:.2f}" if pp_val < 1e6 else "N/A")

        st.divider()

        # Subsection: Top N-Grams
        st.subheader("Top N-Grams Explorer")
        c1, c2 = st.columns([1, 2])
        with c1:
            n_order = st.radio(
                "N-Gram Order:",
                options=[1, 2, 3],
                format_func=lambda x: {1: "1: Unigram", 2: "2: Bigram", 3: "3: Trigram"}[x],
                horizontal=True,
            )
            top_k_ngrams = st.slider("Number of top n-grams:", min_value=5, max_value=50, value=15)

        top_ngrams_list = ngram_model.top_ngrams(n=n_order, top=top_k_ngrams)
        with c2:
            if top_ngrams_list:
                df_ngrams = pd.DataFrame(
                    [
                        {
                            "Rank": i + 1,
                            "N-Gram": " ".join(item[0]),
                            "Count": item[1],
                        }
                        for i, item in enumerate(top_ngrams_list)
                    ]
                )
                st.dataframe(df_ngrams, width="stretch", hide_index=True)
            else:
                st.info("No n-grams found for this order.")

        st.divider()

        # Subsection: Next-Word Prediction Demo
        st.subheader("Interactive Next-Word Prediction")
        st.markdown(
            "Predict the most probable following tokens given a context sequence using smoothed conditional probabilities $P(w | context)$."
        )

        pred_col1, pred_col2 = st.columns([2, 1])
        with pred_col1:
            context_input = st.text_input(
                "Enter context phrase (e.g., 'the model' or 'retrieval-augmented'):",
                value="the model",
            )
        with pred_col2:
            top_preds = st.slider("Top continuations:", min_value=1, max_value=10, value=5)

        if context_input.strip():
            context_tokens = tuple(context_input.strip().lower().split())
            predictions = ngram_model.predict_next(context_tokens, top=top_preds)

            if predictions:
                pred_df = pd.DataFrame(
                    [
                        {
                            "Candidate Next Word": word,
                            "Conditional Probability P(w|context)": f"{prob:.6f}",
                            "Probability Bar": prob,
                        }
                        for word, prob in predictions
                    ]
                )
                st.dataframe(
                    pred_df,
                    column_config={
                        "Probability Bar": st.column_config.ProgressColumn(
                            "Probability",
                            help="Smoothed probability score",
                            format="%.4f",
                            min_value=0.0,
                            max_value=max(p for _, p in predictions) * 1.1 if predictions else 1.0,
                        )
                    },
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No predictions available.")

    # ==========================================
    # TAB 2: HMM POS Tagging
    # ==========================================
    with tab2:
        st.header("HMM Part-of-Speech Tagger")

        # Prominent required notice from project contract
        st.info(
            "ℹ️ **Note on POS Tagger:** Trained on news text (Penn Treebank sample); "
            "exploratory analysis only, does not drive retrieval."
        )

        acc_col1, acc_col2 = st.columns([1, 2])
        with acc_col1:
            if hmm_accuracy > 0:
                st.metric(
                    "Held-out Test Accuracy (10% Treebank split)",
                    f"{hmm_accuracy * 100:.2f}%",
                )
            else:
                st.warning("Treebank data not loaded. Please run `python -m nltk.downloader treebank`.")

        with acc_col2:
            st.markdown(
                """
                - **Algorithm:** First-Order Hidden Markov Model with Viterbi Decoding in log-space.
                - **Features:** Laplace-smoothed transitions & emissions, plus morphology/suffix heuristics for unknown words (`-ing`, `-ed`, `-ly`, `-tion`, capitalized words).
                """
            )

        st.divider()
        st.subheader("Interactive Sentence Tagging")

        # Sentence selection or manual input
        options = ["Choose a corpus sentence...", "Enter custom sentence..."] + [
            f"[{r['document_name']} p.{r['page_number']}] {r['text'][:80]}..."
            for r in records[:10]
        ]
        selected_option = st.selectbox("Select sentence source:", options=options)

        if selected_option == "Enter custom sentence...":
            input_text = st.text_area(
                "Sentence to tag:",
                value="The neural generator produces accurate scientific evidence passages.",
            )
        elif selected_option == "Choose a corpus sentence...":
            input_text = records[0]["text"] if records else ""
        else:
            idx = options.index(selected_option) - 2
            input_text = records[idx]["text"]

        if input_text.strip():
            # Simple whitespace/punctuation tokenization for demo
            import re
            tokens = re.findall(r"[\w'-]+|[.,!?;]", input_text.strip())
            tagged_tokens = hmm_tagger.tag(tokens)

            st.markdown("##### Tagged Sequence:")
            # Render visually rich chips
            chips_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; margin-bottom: 20px;">'
            for word, tag in tagged_tokens:
                desc = PENN_TAG_DESCRIPTIONS.get(tag, "Part of Speech")
                chips_html += f"""
                <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 6px 12px; text-align: center;">
                    <div style="font-weight: 600; font-size: 15px; color: #E0E0E0;">{word}</div>
                    <div style="font-size: 11px; font-weight: 700; color: #4FC3F7; text-transform: uppercase;" title="{desc}">{tag}</div>
                </div>
                """
            chips_html += "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

            with st.expander("🔍 View Tag Breakdown & Legend"):
                tag_df = pd.DataFrame(
                    [
                        {
                            "Token": word,
                            "POS Tag": tag,
                            "Tag Meaning": PENN_TAG_DESCRIPTIONS.get(tag, "Unknown / Other"),
                        }
                        for word, tag in tagged_tokens
                    ]
                )
                st.dataframe(tag_df, width="stretch", hide_index=True)

    # ==========================================
    # TAB 3: Term Frequencies & Word Cloud
    # ==========================================
    with tab3:
        st.header("Corpus Term Frequencies & Word Cloud")
        st.markdown(
            "Analyzes lexical distribution across all valid corpus sentences "
            "(sentences belonging to reference sections are automatically excluded)."
        )

        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            remove_stops = st.checkbox("Filter out common stopwords", value=True)
        with filter_col2:
            top_words_count = st.slider("Top N terms to display:", min_value=10, max_value=100, value=30)

        freqs_list = term_frequencies(
            records, top_n=top_words_count, remove_stopwords=remove_stops
        )
        freqs_dict = dict(freqs_list)

        freq_col, cloud_col = st.columns([1, 1])

        with freq_col:
            st.subheader("Ranked Frequency Table")
            df_freq = frequency_table(records, top_n=top_words_count, remove_stopwords=remove_stops)
            st.dataframe(df_freq, width="stretch", hide_index=True)

        with cloud_col:
            st.subheader("Corpus Word Cloud")
            if freqs_dict:
                wc_image = build_wordcloud(freqs_dict, max_words=top_words_count)
                if wc_image:
                    st.image(wc_image, caption="Corpus Word Cloud", width="stretch")
                else:
                    st.info("Unable to generate word cloud.")
            else:
                st.info("No valid terms available to generate word cloud.")


if __name__ == "__main__":
    render_nlp_analysis_page()
