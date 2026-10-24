"""
Tests for Baseline Concept-Gap Detection and Statistical Measures.
"""
import pytest
import pandas as pd
from models.baseline import BaselineGapDetector, ConceptStats


@pytest.fixture
def detector():
    return BaselineGapDetector(
        alpha_prior=1.0,
        beta_prior=1.0,
        mastery_threshold=0.80,
        remediation_threshold=0.60
    )


def test_zero_attempts_baseline(detector):
    empty_df = pd.DataFrame(columns=["concept_id", "is_correct"])
    stats = detector.evaluate_concept("demorgan", "De Morgan's Theorems", empty_df)

    assert stats.attempts_count == 0
    assert stats.correct_count == 0
    assert stats.error_count == 0
    assert stats.observed_accuracy == 0.0
    assert stats.smoothed_mastery == 0.50
    assert stats.evidence_level == "None"
    assert stats.status == "Unassessed"


def test_laplace_smoothing_prevents_extreme_estimates(detector):
    # Case 1: Student gets 1/1 correct
    # Observed = 100%, but Laplace = (1 + 1)/(1 + 2) = 2/3 = 0.667
    m1 = detector.calculate_laplace_mastery(correct=1, total=1)
    assert round(m1, 3) == 0.667

    # Case 2: Student gets 0/1 correct (1 error)
    # Observed = 0%, but Laplace = (0 + 1)/(1 + 2) = 1/3 = 0.333
    m2 = detector.calculate_laplace_mastery(correct=0, total=1)
    assert round(m2, 3) == 0.333

    # Case 3: Student gets 5/5 correct
    # Laplace = (5 + 1)/(5 + 2) = 6/7 = 0.857
    m3 = detector.calculate_laplace_mastery(correct=5, total=5)
    assert round(m3, 3) == 0.857


def test_wilson_lower_bound_penalizes_small_n(detector):
    w_small = detector.calculate_wilson_lower_bound(correct=1, total=1)
    w_large = detector.calculate_wilson_lower_bound(correct=10, total=10)

    # 10/10 must yield substantially higher lower-bound confidence than 1/1
    assert w_large > w_small
    assert w_small > 0.0
    assert w_large > 0.65


def test_recent_accuracy_exponential_weighting(detector):
    # Earlier attempts incorrect [0, 0], recent attempts correct [1, 1]
    outcomes = [0, 0, 1, 1]
    rec_acc = detector.calculate_recent_accuracy(outcomes, window_size=4)
    # Simple unweighted mean is 0.50. Weighted recent should be > 0.50
    assert rec_acc > 0.50


def test_status_transitions(detector):
    # 1 attempt incorrect -> Needs Practice
    df_fail = pd.DataFrame([{"concept_id": "c1", "is_correct": 0}])
    s_fail = detector.evaluate_concept("c1", "Concept 1", df_fail)
    assert s_fail.status == "Needs Practice"
    assert s_fail.evidence_level == "Low (Preliminary)"

    # 4 attempts all correct -> Provisionally Mastered
    df_pass = pd.DataFrame([{"concept_id": "c1", "is_correct": 1}] * 4)
    s_pass = detector.evaluate_concept("c1", "Concept 1", df_pass)
    assert s_pass.status == "Provisionally Mastered"
    assert s_pass.consecutive_correct == 4
    assert s_pass.evidence_level == "Moderate (Formative)"
