"""
Tests for Quiz Engine and Question Bank Verification.
"""
import pytest
from services.quiz_service import QuizService


@pytest.fixture
def quiz_service():
    return QuizService()


def test_question_bank_integrity(quiz_service):
    questions = quiz_service.get_all_questions()
    assert len(questions) >= 16, "Question bank should contain at least 16 questions"

    concepts = quiz_service.get_all_concepts()
    concept_ids = {c["id"] for c in concepts}

    for q in questions:
        assert "id" in q and q["id"]
        assert "concept_id" in q and q["concept_id"] in concept_ids, f"Invalid concept_id for {q['id']}"
        assert "question_text" in q and len(q["question_text"]) > 10
        assert "correct_answer" in q and q["correct_answer"]
        assert "question_type" in q and q["question_type"] in ["multiple_choice", "short_answer"]

        if q["question_type"] == "multiple_choice":
            assert "options" in q and len(q["options"]) >= 3
            option_ids = {opt["id"] for opt in q["options"]}
            assert q["correct_answer"] in option_ids, f"Correct answer {q['correct_answer']} not in options for {q['id']}"


def test_verified_answer_keys(quiz_service):
    """Verifies that submitting the verified correct answer yields is_correct=True."""
    questions = quiz_service.get_all_questions()
    for q in questions:
        expected = q["correct_answer"]
        is_corr, tag, rationale, ans = quiz_service.validate_answer(q["id"], expected)
        assert is_corr is True, f"Verified answer key failed validation for {q['id']} with answer '{expected}'"


def test_mcq_distractor_misconception_detection(quiz_service):
    # Test De Morgan distractor option B (split bar without flip)
    is_corr, tag, rationale, expected = quiz_service.validate_answer("q_demorgan_01", "B")
    assert is_corr is False
    assert tag == "demorgan_split_no_flip"
    assert "conjunction" in rationale.lower() or "flip" in rationale.lower() or "and" in rationale.lower()

    # Test K-Map diagonal distractor option B
    is_corr, tag, rationale, expected = quiz_service.validate_answer("q_kmap_02", "B")
    assert is_corr is False
    assert tag == "kmap_diagonal_grouping"


def test_short_answer_normalization(quiz_service):
    # q_demorgan_04 expected is "AB"
    test_inputs = ["AB", "ab", "A.B", "a.b", "A * B", " A B "]
    for inp in test_inputs:
        is_corr, tag, rationale, expected = quiz_service.validate_answer("q_demorgan_04", inp)
        assert is_corr is True, f"Failed normalization on input '{inp}' for q_demorgan_04"


def test_quiz_deck_generation(quiz_service):
    # Deck of 6 questions balanced
    deck = quiz_service.generate_quiz_deck(num_questions=6, seed=42)
    assert len(deck) == 6
    unique_ids = {q["id"] for q in deck}
    assert len(unique_ids) == 6, "Deck must not have duplicate questions"

    # Deck filtered by concept
    demorgan_deck = quiz_service.generate_quiz_deck(num_questions=3, concept_id="demorgan", seed=42)
    assert len(demorgan_deck) == 3
    for q in demorgan_deck:
        assert q["concept_id"] == "demorgan"
