"""
Tests for SQLite Database layer in StudyPath.
"""
import os
import pytest
import sqlite3
import pandas as pd
from database.db import (
    init_db,
    get_connection,
    create_or_get_session,
    log_attempt,
    get_attempts,
    update_mastery_record,
    get_mastery_records,
    log_explanation,
    get_explanations,
    export_attempts_csv,
    reset_database,
    load_benchmark_into_session,
    add_task,
    get_tasks,
    toggle_task_completion,
    delete_task,
    get_courses_and_chapters,
    toggle_chapter_progress,
    get_course_progress_summary,
    log_pomodoro_session,
    get_pomodoro_stats,
    get_study_streak,
    check_and_award_badges,
    get_user_badges,
    get_weekly_analytics,
    get_dashboard_summary
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_studypath.db")
    init_db(db_file)
    return db_file


def test_init_db_creates_tables(temp_db):
    conn = get_connection(temp_db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cur.fetchall()]
    conn.close()

    assert "sessions" in tables
    assert "quiz_attempts" in tables
    assert "concept_mastery" in tables
    assert "explanations" in tables
    assert "study_tasks" in tables
    assert "course_chapters" in tables
    assert "chapter_progress" in tables
    assert "pomodoro_sessions" in tables
    assert "user_badges" in tables


def test_session_lifecycle(temp_db):
    s1 = create_or_get_session("test_sess_01", alias="Alice", db_path=temp_db)
    assert s1["session_id"] == "test_sess_01"
    assert s1["alias"] == "Alice"

    # Fetching existing session should return same alias
    s2 = create_or_get_session("test_sess_01", alias="Bob", db_path=temp_db)
    assert s2["session_id"] == "test_sess_01"
    assert s2["alias"] == "Alice"


def test_log_and_query_attempts(temp_db):
    create_or_get_session("sess_01", db_path=temp_db)

    row_id = log_attempt(
        session_id="sess_01",
        question_id="q_demorgan_01",
        concept_id="demorgan",
        question_type="multiple_choice",
        student_answer="B",
        correct_answer="A",
        is_correct=False,
        misconception_tag="demorgan_split_no_flip",
        error_context="Did not flip operator",
        student_reasoning="Kept AND sign",
        time_taken_sec=12.5,
        db_path=temp_db
    )
    assert row_id is not None and row_id > 0

    df = get_attempts(session_id="sess_01", db_path=temp_db)
    assert len(df) == 1
    assert df.iloc[0]["is_correct"] == 0
    assert df.iloc[0]["misconception_tag"] == "demorgan_split_no_flip"
    assert df.iloc[0]["time_taken_sec"] == 12.5


def test_update_and_get_mastery(temp_db):
    create_or_get_session("sess_01", db_path=temp_db)

    update_mastery_record(
        session_id="sess_01",
        concept_id="boolean_laws",
        attempts_count=4,
        correct_count=3,
        error_count=1,
        observed_accuracy=0.75,
        smoothed_mastery=0.667,
        confidence=0.30,
        evidence_level="Moderate (Formative)",
        status="In Progress",
        consecutive_correct=2,
        db_path=temp_db
    )

    df = get_mastery_records("sess_01", db_path=temp_db)
    assert len(df) == 1
    assert df.iloc[0]["concept_id"] == "boolean_laws"
    assert df.iloc[0]["attempts_count"] == 4
    assert df.iloc[0]["status"] == "In Progress"


def test_export_csv_and_reset(temp_db):
    create_or_get_session("sess_export", db_path=temp_db)
    log_attempt(
        session_id="sess_export",
        question_id="q_01",
        concept_id="demorgan",
        question_type="multiple_choice",
        student_answer="A",
        correct_answer="A",
        is_correct=True,
        db_path=temp_db
    )

    csv_data = export_attempts_csv("sess_export", db_path=temp_db)
    assert "sess_export" in csv_data
    assert "q_01" in csv_data

    # Test reset without confirmation fails
    reset_database(confirmation=False, db_path=temp_db)
    df_after_fail = get_attempts("sess_export", db_path=temp_db)
    assert len(df_after_fail) == 1

    # Test reset with confirmation clears data
    success = reset_database(confirmation=True, db_path=temp_db)
    assert success is True
    df_after_reset = get_attempts("sess_export", db_path=temp_db)
    assert len(df_after_reset) == 0


def test_load_benchmark_into_session(temp_db):
    count = load_benchmark_into_session("bench_sess", db_path=temp_db)
    assert count > 0

    df = get_attempts("bench_sess", db_path=temp_db)
    assert len(df) == count


def test_study_task_management(temp_db):
    create_or_get_session("task_user", db_path=temp_db)

    # 1. Add Task
    tid = add_task("task_user", "Review K-Maps Chapter 1", "karnaugh_maps", "2026-12-01", "High", db_path=temp_db)
    assert tid > 0

    # 2. Query Pending Tasks
    pending = get_tasks("task_user", is_completed=0, db_path=temp_db)
    assert len(pending) == 1
    assert pending[0]["title"] == "Review K-Maps Chapter 1"
    assert pending[0]["priority"] == "High"

    # 3. Mark Complete
    toggled = toggle_task_completion(tid, "task_user", db_path=temp_db)
    assert toggled is True

    completed = get_tasks("task_user", is_completed=1, db_path=temp_db)
    assert len(completed) == 1
    assert completed[0]["is_completed"] == 1

    # 4. Reopen Task
    toggled_again = toggle_task_completion(tid, "task_user", db_path=temp_db)
    assert toggled_again is True
    assert len(get_tasks("task_user", is_completed=0, db_path=temp_db)) == 1

    # 5. Delete Task
    deleted = delete_task(tid, "task_user", db_path=temp_db)
    assert deleted is True
    assert len(get_tasks("task_user", db_path=temp_db)) == 0


def test_course_and_chapter_progress(temp_db):
    create_or_get_session("course_user", db_path=temp_db)

    courses = get_courses_and_chapters("course_user", db_path=temp_db)
    assert len(courses) == 4
    first_course = courses[0]
    assert len(first_course["chapters"]) > 0

    # Toggle chapter progress
    ch_id = first_course["chapters"][0]["chapter_id"]
    c_id = first_course["course_id"]
    toggled = toggle_chapter_progress("course_user", c_id, ch_id, db_path=temp_db)
    assert toggled is True

    summary = get_course_progress_summary("course_user", db_path=temp_db)
    assert summary["completed_chapters"] >= 1
    assert summary["overall_progress"] > 0


def test_pomodoro_logging_and_stats(temp_db):
    create_or_get_session("pomo_user", db_path=temp_db)

    pid = log_pomodoro_session("pomo_user", 25, "study", "Deep focus", db_path=temp_db)
    assert pid > 0

    stats = get_pomodoro_stats("pomo_user", db_path=temp_db)
    assert stats["total_sessions"] == 1
    assert stats["total_minutes"] == 25
    assert stats["total_hours"] == 0.4


def test_streaks_badges_and_dashboard_summary(temp_db):
    create_or_get_session("streak_user", db_path=temp_db)

    # Log study activity
    log_pomodoro_session("streak_user", 25, "study", db_path=temp_db)
    add_task("streak_user", "Task 1", deadline="2026-11-20", priority="Medium", db_path=temp_db)

    streak = get_study_streak("streak_user", db_path=temp_db)
    assert streak["current_streak"] >= 1

    badges = get_user_badges("streak_user", db_path=temp_db)
    unlocked = [b["title"] for b in badges if b["is_unlocked"]]
    assert "First Step" in unlocked
    assert "Focus Pioneer" in unlocked

    analytics = get_weekly_analytics("streak_user", db_path=temp_db)
    assert len(analytics["daily_data"]) == 7
    assert analytics["total_weekly_hours"] >= 0.4

    summary = get_dashboard_summary("streak_user", db_path=temp_db)
    assert summary["total_courses"] == 4
    assert summary["pending_tasks_count"] == 1
    assert summary["pomodoro_sessions"] == 1
    assert summary["unlocked_badges"] >= 2
