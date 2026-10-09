import streamlit as st
import tempfile
from pathlib import Path
import time

from src.pipeline.build_corpus import build_corpus

def main():
    st.set_page_config(page_title="EVIDEX Foundation", page_icon="📄")
    st.title("EVIDEX Foundation Pipeline")

    st.markdown("""
    Welcome to **EVIDEX**! Upload your research paper PDFs to build the corpus.
    Once processed, use the sidebar to navigate to **NLP Analysis** or **Claim Evidence**.
    """)

    st.warning("High similarity means related, not supported.")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Building corpus pipeline..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    paths = []
                    for uf in uploaded_files:
                        path = Path(tmpdir) / uf.name
                        with open(path, "wb") as f:
                            f.write(uf.getvalue())
                        paths.append(path)

                    result = build_corpus(paths, force=True)

                    st.session_state["corpus"] = result

                    failures = result.get("failures", [])
                    if failures:
                        st.error(f"Encountered {len(failures)} processing errors:")
                        for fail in failures:
                            st.write(f"- {fail.get('filename', 'Unknown')}: {fail.get('error', 'Error')}")

                    records = result.get("records", [])
                    if records:
                        st.success(f"Successfully processed {len(records)} sentences.")
                        st.write("### Corpus Summary")
                        st.write(f"Total Sentences: {len(records)}")

                        if st.checkbox("Show sample records"):
                            st.json(records[:3])

                        st.info("Corpus is ready! Use the sidebar to navigate to NLP Analysis or Claim Evidence.")
                    else:
                        st.warning("No valid sentences were extracted.")

if __name__ == "__main__":
    main()
