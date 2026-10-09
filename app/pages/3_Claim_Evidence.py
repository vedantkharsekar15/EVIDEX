import streamlit as st
import pandas as pd

from src.claims.claim_candidates import find_claim_candidates, make_user_claim
from src.similarity.retrieval import retrieve
from src.utils.config import load_config

def main():
    st.set_page_config(page_title="Claim Evidence", page_icon="🔍")
    st.title("Claim Evidence Retrieval")

    st.markdown("### ⚠️ Important Note")
    st.info("**High similarity means related, not supported.**")

    # 1. Handle missing corpus state
    if "corpus" not in st.session_state:
        st.warning("No corpus has been processed yet. Please ingest PDFs and build the pipeline first.")
        st.stop()

    corpus = st.session_state["corpus"]
    records = corpus.get("records")
    tfidf_index = corpus.get("tfidf_index")

    if not records or not tfidf_index:
        st.warning("The loaded corpus is incomplete. Expected records and TF-IDF index.")
        st.stop()

    cfg = load_config()

    # 2. Candidate sentences exactly labelled
    st.subheader("Claim candidates")
    candidates = find_claim_candidates(records, cfg)
    
    selected_candidate = None
    if candidates:
        candidate_options = {c["text"]: c for c in candidates}
        selected_text = st.selectbox(
            "Select an extracted candidate claim:", 
            ["-- None --"] + list(candidate_options.keys())
        )
        if selected_text != "-- None --":
            selected_candidate = candidate_options[selected_text]
    else:
        st.write("No candidates were automatically extracted from the current corpus.")

    st.subheader("Custom Claim")
    # 3. Text input for a user's claim
    user_text = st.text_area("Or type your own claim here:")

    # 4. Toggle exactly labelled
    allow_same = st.checkbox("Allow same-paper evidence", value=True)

    if st.button("Search Evidence"):
        claim = None
        
        # Validation
        if user_text.strip():
            try:
                claim = make_user_claim(user_text)
            except ValueError as e:
                st.error(f"Invalid input: {e}")
                st.stop()
        elif selected_candidate:
            claim = selected_candidate
        else:
            st.error("Please enter a custom claim or select a candidate from the dropdown.")
            st.stop()

        records_by_id = {r["sentence_id"]: r for r in records}

        # Retrieve
        with st.spinner("Retrieving evidence..."):
            results = retrieve(
                claim=claim,
                index=tfidf_index,
                records_by_id=records_by_id,
                cfg=cfg,
                allow_same_document=allow_same
            )

        # 5. Clear validation and error messages for unavailable evidence
        if not results:
            st.warning("No evidence found meeting the minimum similarity threshold.")
        else:
            st.success(f"Found {len(results)} related passages.")
            
            # 6. Evidence table containing source/document name, page, similarity, passage
            table_data = []
            for r in results:
                table_data.append({
                    "Document Name": r.get("document_name", "Unknown"),
                    "Page": r.get("page_number", "N/A"),
                    "Similarity": f"{r.get('similarity', 0):.4f}",
                    "Passage": r.get("text", "")
                })
                
            df = pd.DataFrame(table_data)
            st.table(df)

if __name__ == "__main__":
    main()
