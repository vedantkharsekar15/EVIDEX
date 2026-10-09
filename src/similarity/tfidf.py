import math
import json
import os
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

class TfidfIndex:
    sentence_ids: List[str]
    vocab: Dict[Tuple[str, ...], int]
    idf: Dict[Tuple[str, ...], float]
    tfidf_matrix: List[Dict[int, float]]
    cfg: dict

    def __init__(self, sentence_ids: List[str], vocab: Dict[Tuple[str, ...], int], idf: Dict[Tuple[str, ...], float], tfidf_matrix: List[Dict[int, float]], cfg: dict):
        self.sentence_ids = sentence_ids
        self.vocab = vocab
        self.idf = idf
        self.tfidf_matrix = tfidf_matrix
        self.cfg = cfg

    @classmethod
    def build(cls, records: List[dict], cfg: dict) -> "TfidfIndex":
        ngram_range = cfg.get("tfidf", {}).get("ngram_range", [1, 2])
        min_df = cfg.get("tfidf", {}).get("min_df", 1)
        sublinear_tf = cfg.get("tfidf", {}).get("sublinear_tf", True)

        filtered_records = [r for r in records if not r.get("in_references", False)]
        sentence_ids = []
        doc_frequencies = defaultdict(int)
        term_frequencies_per_doc = []

        N = len(filtered_records)

        for record in filtered_records:
            sentence_ids.append(record["sentence_id"])
            tokens = record.get("normalized_tokens", [])
            ngrams = []
            
            if tokens:
                for n in range(ngram_range[0], ngram_range[1] + 1):
                    for i in range(len(tokens) - n + 1):
                        ngrams.append(tuple(tokens[i:i+n]))

            tf_counter = Counter(ngrams)
            term_frequencies_per_doc.append(tf_counter)

            for ngram in tf_counter.keys():
                doc_frequencies[ngram] += 1

        vocab = {}
        idf = {}
        for ngram, df in doc_frequencies.items():
            if df >= min_df:
                vocab[ngram] = len(vocab)
                idf[ngram] = math.log((1 + N) / (1 + df)) + 1.0

        tfidf_matrix = []
        for tf_counter in term_frequencies_per_doc:
            doc_vector = {}
            norm_sq = 0.0
            for ngram, count in tf_counter.items():
                if ngram in vocab:
                    idx = vocab[ngram]
                    tf = (1.0 + math.log(count)) if sublinear_tf else float(count)
                    weight = tf * idf[ngram]
                    doc_vector[idx] = weight
                    norm_sq += weight * weight

            norm = math.sqrt(norm_sq)
            if norm > 0:
                for idx in doc_vector:
                    doc_vector[idx] /= norm

            tfidf_matrix.append(doc_vector)

        return cls(sentence_ids, vocab, idf, tfidf_matrix, cfg)

    def transform_query(self, tokens: List[str]) -> Dict[int, float]:
        ngram_range = self.cfg.get("tfidf", {}).get("ngram_range", [1, 2])
        sublinear_tf = self.cfg.get("tfidf", {}).get("sublinear_tf", True)

        ngrams = []
        if tokens:
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(tokens) - n + 1):
                    ngrams.append(tuple(tokens[i:i+n]))

        tf_counter = Counter(ngrams)
        query_vector = {}
        norm_sq = 0.0

        for ngram, count in tf_counter.items():
            if ngram in self.vocab:
                idx = self.vocab[ngram]
                tf = (1.0 + math.log(count)) if sublinear_tf else float(count)
                weight = tf * self.idf[ngram]
                query_vector[idx] = weight
                norm_sq += weight * weight

        norm = math.sqrt(norm_sq)
        if norm > 0:
            for idx in query_vector:
                query_vector[idx] /= norm

        return query_vector

    def save(self, fingerprint: str) -> None:
        artifacts_dir = self.cfg.get("paths", {}).get("artifacts", "data/artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        path = os.path.join(artifacts_dir, f"tfidf_{fingerprint}.json")

        vocab_str = {json.dumps(list(k)): v for k, v in self.vocab.items()}
        idf_str = {json.dumps(list(k)): v for k, v in self.idf.items()}

        data = {
            "fingerprint": fingerprint,
            "sentence_ids": self.sentence_ids,
            "vocab": vocab_str,
            "idf": idf_str,
            "tfidf_matrix": self.tfidf_matrix,
            "cfg": self.cfg
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, fingerprint: str) -> Optional["TfidfIndex"]:
        try:
            cfg = load_config()
            artifacts_dir = cfg.get("paths", {}).get("artifacts", "data/artifacts")
            path = os.path.join(artifacts_dir, f"tfidf_{fingerprint}.json")

            if not os.path.exists(path):
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("fingerprint") != fingerprint:
                return None

            sentence_ids = data["sentence_ids"]
            vocab_str = data["vocab"]
            idf_str = data["idf"]

            vocab = {tuple(json.loads(k)): v for k, v in vocab_str.items()}
            idf = {tuple(json.loads(k)): v for k, v in idf_str.items()}

            tfidf_matrix = []
            for row in data["tfidf_matrix"]:
                tfidf_matrix.append({int(k): float(v) for k, v in row.items()})

            saved_cfg = data["cfg"]
            return cls(sentence_ids, vocab, idf, tfidf_matrix, saved_cfg)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt artifact tfidf_{fingerprint}.json: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load tfidf artifact {fingerprint}: {e}")
            return None
