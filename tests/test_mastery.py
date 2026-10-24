"""
Tests for Adaptive Learning Loop and Explanation Engine.
"""
import pytest
from services.quiz_service import QuizService
from services.explanation_service import ExplanationService
from services.mastery_service import MasteryService
from database.db import init_db, log_attempt


@pytest.fixture
def temp_env(tmp_path):
    db_file = str(tmp_path / "test_mastery.db")
    init_db(db_file)
    quiz_svc = QuizService()
    expl_svc = ExplanationService(db_path=db_file)
    mastery_svc = MasteryService(quiz_service=quiz_svc, db_path=db_file)
    return {
        "db": db_file,
        "quiz": quiz_svc,
        "expl": expl_svc,
        "mastery": mastery_svc
    }


def test_explanation_fallback_structure(temp_env):
    expl_svc = temp_env["expl"]
    result = expl_svc.generate_explanation(
        session_id="test_sess",
        concept_id="demorgan",
        misconception_tag="demorgan_split_no_flip",
        force_local=True
    )

    assert result["provider"] == "local_pedagogical_rules"
    assert "suspected_gap" in result and len(result["suspected_gap"]) > 5
    assert "explanation_text" in result and len(result["explanation_text"]) > 10
    assert "worked_example" in result and len(result["worked_example"]) > 10
    assert "follow_up_question" in result and len(result["follow_up_question"]) > 5
    assert "prerequisite_hint" in result and len(result["prerequisite_hint"]) > 5


def test_adaptive_prerequisite_transition(temp_env):
    mastery_svc = temp_env["mastery"]
    db_file = temp_env["db"]

    # Simulate 2 consecutive errors on demorgan questions
    log_attempt("stu_01", "q_demorgan_01", "demorgan", "multiple_choice", "B", "A", False, db_path=db_file)
    log_attempt("stu_01", "q_demorgan_02", "demorgan", "multiple_choice", "C", "A", False, db_path=db_file)

    next_q, rationale = mastery_svc.select_adaptive_next_question(
        session_id="stu_01",
        last_question_id="q_demorgan_02",
        last_is_correct=False
    )

    assert next_q is not None
    # Prerequisite of demorgan is boolean_laws
    assert next_q["concept_id"] == "boolean_laws"
    assert "prerequisite" in rationale.lower()


def test_adaptive_success_progression(temp_env):
    mastery_svc = temp_env["mastery"]
    db_file = temp_env["db"]

    # Simulate successful answer on boolean_laws
    log_attempt("stu_02", "q_boolean_01", "boolean_laws", "multiple_choice", "A", "A", True, db_path=db_file)
    mastery_svc.update_session_mastery("stu_02", "boolean_laws")

    next_q, rationale = mastery_svc.select_adaptive_next_question(
        session_id="stu_02",
        last_question_id="q_boolean_01",
        last_is_correct=True
    )

    assert next_q is not None
    assert "progression" in rationale.lower() or "assessment" in rationale.lower() or "advanc" in rationale.lower()
