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
    load_benchmark_into_session
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
