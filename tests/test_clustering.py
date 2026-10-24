"""
Tests for Semantic Embeddings, Clustering, and Dimensionality Reduction.
"""
import pytest
import numpy as np
from models.embedding_model import SemanticEmbeddingPipeline
from models.clustering import MistakeClusterer


@pytest.fixture
def embedder():
    return SemanticEmbeddingPipeline()


@pytest.fixture
def clusterer():
    return MistakeClusterer(distance_threshold=0.40)


def test_embedding_pipeline_encoding(embedder):
    texts = [
        "Concept: demorgan | Student Answer: A'.B' | Expected: A'+B' | Error Pattern: split bar without flip",
        "Concept: demorgan | Student Answer: X'.Y' | Expected: X'+Y' | Error Pattern: split bar without flip"
    ]
    embeddings = embedder.encode(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == embedder.embedding_dim
    assert embedder.embedding_dim in [384, 64]

    # Check normalization: L2 norm of each row should be approximately 1.0
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-3


def test_clustering_handles_empty_and_single_sample(clusterer):
    # Empty
    empty_res = clusterer.cluster_mistakes(np.empty((0, 384)), [])
    assert empty_res["n_samples"] == 0
    assert empty_res["n_clusters"] == 0

    # Single sample
    single_emb = np.ones((1, 384), dtype=np.float32)
    single_res = clusterer.cluster_mistakes(single_emb, [{"id": "single"}])
    assert single_res["n_samples"] == 1
    assert single_res["labels"] == [0]
    assert single_res["silhouette"] is None


def test_clustering_groups_semantically_similar_mistakes(embedder, clusterer):
    # Create two pairs of distinct errors
    records = [
        # De Morgan split without flip errors (should cluster together)
        {
            "concept_id": "demorgan",
            "question_text": "Simplify (A.B)'",
            "student_answer": "A'.B'",
            "correct_answer": "A'+B'",
            "error_context": "Kept AND operator",
            "student_reasoning": "Split bar over variables without changing dot",
            "misconception_tag": "demorgan_split_no_flip"
        },
        {
            "concept_id": "demorgan",
            "question_text": "Simplify (X.Y.Z)'",
            "student_answer": "X'.Y'.Z'",
            "correct_answer": "X'+Y'+Z'",
            "error_context": "Distributed bar without toggling to OR",
            "student_reasoning": "Bar split into three individual primes",
            "misconception_tag": "demorgan_split_no_flip"
        },
        # K-Map diagonal error (distinct, should be in a separate cluster)
        {
            "concept_id": "karnaugh_maps",
            "question_text": "Can corner diagonal cells be grouped?",
            "student_answer": "Yes corner diagonal",
            "correct_answer": "No diagonal prohibited",
            "error_context": "Diagonal grouping attempt",
            "student_reasoning": "Diagonals touch vertices",
            "misconception_tag": "kmap_diagonal_grouping"
        },
        {
            "concept_id": "karnaugh_maps",
            "question_text": "Are cell 00 and 11 adjacent?",
            "student_answer": "Yes they touch diagonally",
            "correct_answer": "No only horizontal or vertical",
            "error_context": "Diagonal adjacency assumption",
            "student_reasoning": "Diagonal cells are neighbors",
            "misconception_tag": "kmap_diagonal_grouping"
        }
    ]

    texts = [embedder.construct_mistake_text(r) for r in records]
    embs = embedder.encode(texts)
    results = clusterer.cluster_mistakes(embs, records, method="agglomerative")

    labels = results["labels"]
    assert len(labels) == 4
    # The two De Morgan mistakes should share the same label
    assert labels[0] == labels[1]
    # The two K-Map mistakes should share the same label
    assert labels[2] == labels[3]
    # De Morgan and K-Map should be in different clusters
    assert labels[0] != labels[2]


def test_pca_2d_projection(clusterer):
    embs = np.random.randn(5, 384).astype(np.float32)
    # Normalize
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    coords, var_ratios = clusterer._compute_pca_projection(embs)

    assert coords.shape == (5, 2)
    assert len(var_ratios) == 2
    assert 0.0 <= sum(var_ratios) <= 1.0


def test_reproducibility(embedder, clusterer):
    sample = ["Concept: demorgan | Question: (A.B)' | Student Answer: A'.B' | Expected: A'+B'"]
    e1 = embedder.encode(sample)
    e2 = embedder.encode(sample)
    np.testing.assert_allclose(e1, e2, atol=1e-5)
