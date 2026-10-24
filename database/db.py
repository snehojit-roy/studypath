"""
SQLite database interface for StudyPath.
Manages sessions, attempt history, concept mastery updates, and explanation logs.
"""
from __future__ import annotations

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "studypath.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a connection to SQLite database, ensuring directory exists."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Executes schema.sql if tables do not exist."""
    conn = get_connection(db_path)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def create_or_get_session(
    session_id: str,
    alias: str = "Student",
    notes: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Retrieves an existing session or creates a new one."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if row:
            return dict(row)

        cur.execute(
            "INSERT INTO sessions (session_id, alias, notes) VALUES (?, ?, ?)",
            (session_id, alias, notes or "Default Session")
        )
        conn.commit()
        cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        new_row = cur.fetchone()
        return dict(new_row) if new_row else {"session_id": session_id, "alias": alias}
    finally:
        conn.close()


def log_attempt(
    session_id: str,
    question_id: str,
    concept_id: str,
    question_type: str,
    student_answer: str,
    correct_answer: str,
    is_correct: bool,
    misconception_tag: Optional[str] = None,
    error_context: Optional[str] = None,
    student_reasoning: Optional[str] = None,
    time_taken_sec: float = 0.0,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Logs a single quiz answer attempt."""
    conn = get_connection(db_path)
    try:
        create_or_get_session(session_id, db_path=db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO quiz_attempts (
                session_id, question_id, concept_id, question_type,
                student_answer, correct_answer, is_correct,
                misconception_tag, error_context, student_reasoning,
                time_taken_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question_id,
                concept_id,
                question_type,
                str(student_answer),
                str(correct_answer),
                1 if is_correct else 0,
                misconception_tag,
                error_context,
                student_reasoning,
                float(time_taken_sec)
            )
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_attempts(
    session_id: Optional[str] = None,
    concept_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Returns a pandas DataFrame of quiz attempts matching query filters."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM quiz_attempts WHERE 1=1"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if concept_id:
            query += " AND concept_id = ?"
            params.append(concept_id)
        query += " ORDER BY id ASC"

        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def update_mastery_record(
    session_id: str,
    concept_id: str,
    attempts_count: int,
    correct_count: int,
    error_count: int,
    observed_accuracy: float,
    smoothed_mastery: float,
    confidence: float,
    evidence_level: str,
    status: str,
    consecutive_correct: int,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Inserts or updates the estimated mastery for a given concept."""
    conn = get_connection(db_path)
    try:
        create_or_get_session(session_id, db_path=db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO concept_mastery (
                session_id, concept_id, attempts_count, correct_count,
                error_count, observed_accuracy, smoothed_mastery,
                confidence, evidence_level, status, consecutive_correct,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id, concept_id) DO UPDATE SET
                attempts_count=excluded.attempts_count,
                correct_count=excluded.correct_count,
                error_count=excluded.error_count,
                observed_accuracy=excluded.observed_accuracy,
                smoothed_mastery=excluded.smoothed_mastery,
                confidence=excluded.confidence,
                evidence_level=excluded.evidence_level,
                status=excluded.status,
                consecutive_correct=excluded.consecutive_correct,
                last_updated=CURRENT_TIMESTAMP
            """,
            (
                session_id, concept_id, attempts_count, correct_count,
                error_count, observed_accuracy, smoothed_mastery,
                confidence, evidence_level, status, consecutive_correct
            )
        )
        conn.commit()
    finally:
        conn.close()


def get_mastery_records(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Returns DataFrame of concept mastery for the given session."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM concept_mastery WHERE session_id = ? ORDER BY concept_id ASC"
        return pd.read_sql_query(query, conn, params=[session_id])
    finally:
        conn.close()


def log_explanation(
    session_id: str,
    concept_id: str,
    misconception_tag: Optional[str],
    provider: str,
    suspected_gap: str,
    explanation_text: str,
    worked_example: str,
    follow_up_question: str,
    prerequisite_hint: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Logs an explanation delivered to a student."""
    conn = get_connection(db_path)
    try:
        create_or_get_session(session_id, db_path=db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO explanations (
                session_id, concept_id, misconception_tag, provider,
                suspected_gap, explanation_text, worked_example,
                follow_up_question, prerequisite_hint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, concept_id, misconception_tag, provider,
                suspected_gap, explanation_text, worked_example,
                follow_up_question, prerequisite_hint
            )
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_explanations(
    session_id: str,
    concept_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieves explanation logs."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM explanations WHERE session_id = ?"
        params = [session_id]
        if concept_id:
            query += " AND concept_id = ?"
            params.append(concept_id)
        query += " ORDER BY id DESC"
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def export_attempts_csv(
    session_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> str:
    """Exports attempts to a CSV string."""
    df = get_attempts(session_id=session_id, db_path=db_path)
    return df.to_csv(index=False)


def reset_database(
    confirmation: bool = False,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Clears all records if confirmation is explicitly True."""
    if not confirmation:
        return False
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM explanations")
        cur.execute("DELETE FROM concept_mastery")
        cur.execute("DELETE FROM quiz_attempts")
        cur.execute("DELETE FROM sessions")
        conn.commit()
        return True
    finally:
        conn.close()


def load_benchmark_into_session(
    session_id: str = "demo_student",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Loads benchmark dataset into database under the specified session."""
    init_db(db_path)
    create_or_get_session(session_id, alias="Demo Student (Benchmark)", notes="Preloaded benchmark dataset for evaluation", db_path=db_path)
    
    benchmark_file = os.path.join(DEFAULT_DB_DIR, "benchmark_dataset.json")
    if not os.path.exists(benchmark_file):
        return 0

    with open(benchmark_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    inserted = 0
    for r in records:
        log_attempt(
            session_id=session_id,
            question_id=r.get("question_id", "q_unknown"),
            concept_id=r.get("concept_id", "demorgan"),
            question_type="multiple_choice",
            student_answer=r.get("student_answer", ""),
            correct_answer=r.get("expected_answer", ""),
            is_correct=r.get("is_correct", False),
            misconception_tag=r.get("ground_truth_misconception"),
            error_context=r.get("error_context"),
            student_reasoning=r.get("student_reasoning_quote"),
            time_taken_sec=15.0,
            db_path=db_path
        )
        inserted += 1

    return inserted
