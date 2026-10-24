"""
Tests for Scientific Evaluation Pipeline and Benchmark Dataset.
"""
import pytest
from models.evaluation import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator()


def test_benchmark_data_loading(evaluator):
    data = evaluator.load_benchmark_data()
    assert len(data) >= 20, "Benchmark dataset should contain at least 20 records"

    for r in data:
        assert "response_id" in r
        assert "question_id" in r
        assert "concept_id" in r
        assert "student_answer" in r
        assert "expected_answer" in r
        assert "is_correct" in r
        if not r["is_correct"]:
            assert "ground_truth_misconception" in r


def test_benchmark_experiment_execution(evaluator):
    results = evaluator.run_benchmark_experiment()

    assert "clustering_results" in results
    assert "metrics_comparison" in results
    assert "dataset_notes" in results
    assert "case_study_successes" in results
    assert "case_study_failures" in results

    comp = results["metrics_comparison"]
    assert "ari_semantic_vs_concept" in comp
    assert "ari_semantic_vs_ground_truth" in comp
    assert "semantic_improvement_ari" in comp

    # Verify that semantic clustering shows improvement over raw concept-tag baseline
    # (or at least non-negative improvement on the benchmark)
    assert comp["semantic_improvement_ari"] >= 0.0

    notes = results["dataset_notes"]
    assert notes["is_statistically_conclusive"] is False
    assert len(notes["limitation_statement"]) > 20
