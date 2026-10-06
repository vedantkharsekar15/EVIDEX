# EVIDEX Mini Project Plan

## Overview
EVIDEX is an NLP-based research intelligence system. The complete project is divided into modules.

## Module 1: Foundation, Ingestion, Pipeline, GitHub
**Owner**: Vedant Kharsekar
**Branch**: feat/foundation-ingestion

Provides the foundation and PDF ingestion layer that the other team members will build on.

### 1. Shared Contract
Records to be implemented in `src/utils/schemas.py`:
- `PageRecord`: document_id, filename, page_number, text
- `SentenceRecord`: sentence_id, document_id, document_name, page_number, page_end, sentence_index, text, tokens, normalized_tokens, in_references, is_caption
- `Claim`: claim_id, text, source
- `ClaimCandidate`: candidate_id, sentence_id, document_id, page_number, text, score, signals
- `Evidence`: rank, similarity, sentence_id, document_id, document_name, page_number, text

**ID Rules**:
- `document_id` = sha256(file bytes).hexdigest()[:12]
- `sentence_id` = f"{document_id}_s{sentence_index:05d}" (zero-based index)

### 2. Configuration & Utilities
- Configuration loaded from `config.yaml` using PyYAML.
- Reusable logger.
- Deterministic file and hashing utilities (`sha256_bytes`, `document_id_from_bytes`, JSON save/load, restricted artifact save/load).
- `corpus_fingerprint`: Deterministic hash of document IDs and relevant config.

### 3. PDF Ingestion & Validation
- Validation rejects missing, empty, invalid/corrupt, or fake PDFs.
- PyMuPDF extraction preserving page numbers, provenance, and reading order.
- Tolerates empty pages and handles batch files gracefully.

### 4. Metadata
- Tracks document provenance.
- Duplicate detection based on document_id hash in a central registry.

## Modules 2, 3, 4 (Outside Module 1 Scope)
- Text Extraction Refinement & NLP Cleaning
- Sentence Splitting & Claim Detection
- N-Gram, HMM POS, TF-IDF
- Pipeline integration
