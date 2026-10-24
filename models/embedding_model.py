"""
Semantic Embedding Pipeline using sentence-transformers (all-MiniLM-L6-v2)
with graceful scikit-learn deterministic fallback.
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class SemanticEmbeddingPipeline:
    """
    Transforms student mistake contexts into dense semantic vectors.
    Uses sentence-transformers 'all-MiniLM-L6-v2' (384-d).
    Includes a deterministic TF-IDF + SVD fallback for offline or restricted environments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.backend = "uninitialized"
        self.embedding_dim = 384
        self._cache: Dict[str, np.ndarray] = {}
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads sentence-transformers model if available, otherwise falls back gracefully."""
        try:
            from sentence_transformers import SentenceTransformer
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Check for local workspace weights first (instant offline loading)
            local_weights = os.path.join(
                os.path.dirname(__file__), "weights", "all-MiniLM-L6-v2"
            )
            if os.path.exists(local_weights):
                self.model = SentenceTransformer(local_weights)
                self.backend = f"sentence-transformers ({self.model_name})"
            else:
                try:
                    self.model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    self.model = SentenceTransformer(self.model_name)
                self.backend = f"sentence-transformers ({self.model_name})"
            if hasattr(self.model, "get_embedding_dimension"):
                self.embedding_dim = self.model.get_embedding_dimension()
            else:
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            # Fallback to scikit-learn TF-IDF + LSA
            self.backend = f"scikit-learn-fallback (tfidf+lsa, reason: {type(e).__name__})"
            self.embedding_dim = 64
            self.model = None

    def construct_mistake_text(self, record: Dict[str, Any]) -> str:
        """
        Synthesizes mistake context into a rich semantic string.
        Combines concept tag, question text, student's answer, expected answer,
        error rationale, and student reasoning.
        """
        concept = record.get("concept_id", "general")
        q_text = record.get("question_text", "").strip()
        ans = str(record.get("student_answer", "")).strip()
        expected = str(record.get("correct_answer") or record.get("expected_answer", "")).strip()
        err_ctx = str(record.get("error_context") or record.get("rationale", "")).strip()
        reasoning = str(record.get("student_reasoning") or record.get("student_reasoning_quote", "")).strip()

        components = [
            f"Concept: {concept}",
            f"Question: {q_text}",
            f"Student Answer: {ans}",
            f"Expected Answer: {expected}"
        ]
        if err_ctx:
            components.append(f"Error Pattern: {err_ctx}")
        if reasoning:
            components.append(f"Student Explanation: {reasoning}")

        return " | ".join(components)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes list of strings into normalized 2D numpy array (N, D)."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Check cache
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            if "sentence-transformers" in self.backend and self.model is not None:
                try:
                    embeddings = self.model.encode(
                        uncached_texts,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    embeddings = np.array(embeddings, dtype=np.float32)
                except Exception:
                    embeddings = self._fallback_encode(uncached_texts)
            else:
                embeddings = self._fallback_encode(uncached_texts)

            for idx, emb in zip(uncached_indices, embeddings):
                self._cache[uncached_texts[uncached_indices.index(idx)]] = emb
                results[idx] = emb

        return np.vstack(results)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Deterministic fallback embedding using TF-IDF and TruncatedSVD."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=256)
        tfidf = vectorizer.fit_transform(texts)
        
        n_samples, n_features = tfidf.shape
        n_comp = min(self.embedding_dim, n_samples, n_features)
        
        if n_comp > 1 and n_samples > 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            dense = svd.fit_transform(tfidf)
        else:
            dense = tfidf.toarray()

        # Pad or truncate to self.embedding_dim
        if dense.shape[1] < self.embedding_dim:
            pad = np.zeros((dense.shape[0], self.embedding_dim - dense.shape[1]), dtype=np.float32)
            dense = np.hstack([dense, pad])
        elif dense.shape[1] > self.embedding_dim:
            dense = dense[:, :self.embedding_dim]

        # L2 Normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype(np.float32)
