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
    """Executes schema.sql if tables do not exist and seeds initial data."""
    conn = get_connection(db_path)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        seed_default_chapters(conn)
    finally:
        conn.close()


def seed_default_chapters(conn: sqlite3.Connection) -> None:
    """Populates default chapters for each syllabus course/concept if empty."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM course_chapters")
    if cur.fetchone()[0] > 0:
        return

    default_chapters = [
        # Boolean Laws and Basic Simplification
        ("boolean_laws", "bool_ch1", "Fundamental Identities & Postulates", 1),
        ("boolean_laws", "bool_ch2", "Involution & Complement Laws", 2),
        ("boolean_laws", "bool_ch3", "Absorption & Redundancy Theorems", 3),
        ("boolean_laws", "bool_ch4", "Second Distributive Law & Consensus", 4),
        ("boolean_laws", "bool_ch5", "Algebraic Simplification Practice", 5),

        # De Morgan's Theorems
        ("demorgan", "dem_ch1", "Complementation Mechanics & Duality", 1),
        ("demorgan", "dem_ch2", "Two-Variable De Morgan Expansions", 2),
        ("demorgan", "dem_ch3", "Multi-Variable (3+) Complementation", 3),
        ("demorgan", "dem_ch4", "Dual-Expression Reduction Techniques", 4),

        # Karnaugh Map (K-Map) Minimization
        ("karnaugh_maps", "kmap_ch1", "Gray Code Adjacency & Grid Layout", 1),
        ("karnaugh_maps", "kmap_ch2", "2-Variable & 3-Variable Minterm Grouping", 2),
        ("karnaugh_maps", "kmap_ch3", "4-Variable K-Map & Toroidal Wrap-Around", 3),
        ("karnaugh_maps", "kmap_ch4", "Prime & Essential Prime Implicants", 4),
        ("karnaugh_maps", "kmap_ch5", "Don't-Care Conditions & POS Minimization", 5),

        # Universal Logic (NAND/NOR Implementation)
        ("universal_logic", "univ_ch1", "Functional Completeness of NAND & NOR", 1),
        ("universal_logic", "univ_ch2", "Inverter Realization using Universal Gates", 2),
        ("universal_logic", "univ_ch3", "AND / OR Synthesis with NAND Gates", 3),
        ("universal_logic", "univ_ch4", "AND / OR Synthesis with NOR Gates", 4),
        ("universal_logic", "univ_ch5", "Multilevel Universal Circuit Optimization", 5),
    ]

    cur.executemany(
        "INSERT OR IGNORE INTO course_chapters (course_id, chapter_id, title, order_num) VALUES (?, ?, ?, ?)",
        default_chapters
    )
    conn.commit()


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
        cur.execute("DELETE FROM user_badges")
        cur.execute("DELETE FROM pomodoro_sessions")
        cur.execute("DELETE FROM chapter_progress")
        cur.execute("DELETE FROM study_tasks")
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


# ========================================================
# DAILY STUDY PLANNER (TASKS)
# ========================================================

def add_task(
    session_id: str,
    title: str,
    course_id: Optional[str] = None,
    deadline: Optional[str] = None,
    priority: str = "Medium",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Creates a new study task for the session."""
    clean_title = (title or "").strip()
    if not clean_title:
        raise ValueError("Task title cannot be empty.")
    
    valid_priorities = {"Low", "Medium", "High"}
    clean_priority = priority if priority in valid_priorities else "Medium"

    conn = get_connection(db_path)
    try:
        create_or_get_session(session_id, db_path=db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO study_tasks (session_id, title, course_id, deadline, priority, is_completed)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (session_id, clean_title, course_id, deadline, clean_priority)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_tasks(
    session_id: str,
    is_completed: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves tasks for a session, optionally filtered by completion status."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        query = "SELECT * FROM study_tasks WHERE session_id = ?"
        params: List[Any] = [session_id]

        if is_completed is not None:
            query += " AND is_completed = ?"
            params.append(1 if is_completed else 0)

        query += " ORDER BY is_completed ASC, CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 WHEN 'Low' THEN 3 ELSE 4 END ASC, id DESC"
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def toggle_task_completion(
    task_id: int,
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Toggles completion state of a task and updates completed_at."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT is_completed FROM study_tasks WHERE id = ? AND session_id = ?", (task_id, session_id))
        row = cur.fetchone()
        if not row:
            return False

        current_status = row["is_completed"]
        new_status = 0 if current_status == 1 else 1

        if new_status == 1:
            cur.execute(
                "UPDATE study_tasks SET is_completed = 1, completed_at = CURRENT_TIMESTAMP WHERE id = ? AND session_id = ?",
                (task_id, session_id)
            )
        else:
            cur.execute(
                "UPDATE study_tasks SET is_completed = 0, completed_at = NULL WHERE id = ? AND session_id = ?",
                (task_id, session_id)
            )
        conn.commit()
        check_and_award_badges(session_id, db_path=db_path)
        return True
    finally:
        conn.close()


def delete_task(
    task_id: int,
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Deletes a study task."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM study_tasks WHERE id = ? AND session_id = ?", (task_id, session_id))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ========================================================
# COURSE PROGRESS TRACKER
# ========================================================

def get_courses_and_chapters(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves all courses with constituent chapters and user completion flags."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        query = """
            SELECT 
                c.course_id,
                c.chapter_id,
                c.title AS chapter_title,
                c.order_num,
                COALESCE(p.is_completed, 0) AS is_completed,
                p.completed_at
            FROM course_chapters c
            LEFT JOIN chapter_progress p 
                ON c.course_id = p.course_id 
                AND c.chapter_id = p.chapter_id 
                AND p.session_id = ?
            ORDER BY c.course_id, c.order_num ASC
        """
        cur.execute(query, (session_id,))
        rows = cur.fetchall()

        # Map concept ids to names
        concept_names = {
            "boolean_laws": "Boolean Laws & Basic Simplification",
            "demorgan": "De Morgan's Theorems",
            "karnaugh_maps": "Karnaugh Map (K-Map) Minimization",
            "universal_logic": "Universal Logic (NAND/NOR Implementation)"
        }

        courses_dict: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            cid = r["course_id"]
            if cid not in courses_dict:
                courses_dict[cid] = {
                    "course_id": cid,
                    "course_name": concept_names.get(cid, cid.replace("_", " ").title()),
                    "chapters": [],
                    "completed_count": 0,
                    "total_count": 0
                }
            ch_completed = bool(r["is_completed"])
            courses_dict[cid]["chapters"].append({
                "chapter_id": r["chapter_id"],
                "title": r["chapter_title"],
                "order_num": r["order_num"],
                "is_completed": ch_completed,
                "completed_at": r["completed_at"]
            })
            courses_dict[cid]["total_count"] += 1
            if ch_completed:
                courses_dict[cid]["completed_count"] += 1

        result = []
        for cid, cinfo in courses_dict.items():
            tot = cinfo["total_count"]
            comp = cinfo["completed_count"]
            cinfo["progress_percent"] = round((comp / tot * 100), 1) if tot > 0 else 0.0
            cinfo["is_fully_completed"] = (comp == tot and tot > 0)
            result.append(cinfo)

        return result
    finally:
        conn.close()


def toggle_chapter_progress(
    session_id: str,
    course_id: str,
    chapter_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Toggles completion state of a course chapter."""
    conn = get_connection(db_path)
    try:
        create_or_get_session(session_id, db_path=db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT is_completed FROM chapter_progress WHERE session_id = ? AND course_id = ? AND chapter_id = ?",
            (session_id, course_id, chapter_id)
        )
        row = cur.fetchone()

        if row is None:
            cur.execute(
                """
                INSERT INTO chapter_progress (session_id, course_id, chapter_id, is_completed, completed_at)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                """,
                (session_id, course_id, chapter_id)
            )
        else:
            new_status = 0 if row["is_completed"] == 1 else 1
            completed_val = "CURRENT_TIMESTAMP" if new_status == 1 else "NULL"
            if new_status == 1:
                cur.execute(
                    "UPDATE chapter_progress SET is_completed = 1, completed_at = CURRENT_TIMESTAMP WHERE session_id = ? AND course_id = ? AND chapter_id = ?",
                    (session_id, course_id, chapter_id)
                )
            else:
                cur.execute(
                    "UPDATE chapter_progress SET is_completed = 0, completed_at = NULL WHERE session_id = ? AND course_id = ? AND chapter_id = ?",
                    (session_id, course_id, chapter_id)
                )

        conn.commit()
        check_and_award_badges(session_id, db_path=db_path)
        return True
    finally:
        conn.close()


def get_course_progress_summary(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Computes overall course progress statistics."""
    courses = get_courses_and_chapters(session_id, db_path=db_path)
    total_courses = len(courses)
    completed_courses = sum(1 for c in courses if c["is_fully_completed"])
    total_chapters = sum(c["total_count"] for c in courses)
    completed_chapters = sum(c["completed_count"] for c in courses)
    overall_progress = round((completed_chapters / total_chapters * 100), 1) if total_chapters > 0 else 0.0

    return {
        "total_courses": total_courses,
        "completed_courses": completed_courses,
        "total_chapters": total_chapters,
        "completed_chapters": completed_chapters,
        "overall_progress": overall_progress,
        "courses": courses
    }


# ========================================================
# POMODORO FOCUS TIMER
# ========================================================

def log_pomodoro_session(
    session_id: str,
    duration_minutes: int = 25,
    session_type: str = "study",
    notes: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Logs a completed Pomodoro study or break session."""
    conn = get_connection(db_path)
    try:
        create_or_get_session(session_id, db_path=db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO pomodoro_sessions (session_id, duration_minutes, session_type, notes, completed_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (session_id, int(duration_minutes), session_type, notes)
        )
        conn.commit()
        check_and_award_badges(session_id, db_path=db_path)
        return cur.lastrowid
    finally:
        conn.close()


def get_pomodoro_stats(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Retrieves aggregate pomodoro focus session metrics."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 
                COUNT(*) AS total_sessions,
                COALESCE(SUM(duration_minutes), 0) AS total_minutes
            FROM pomodoro_sessions 
            WHERE session_id = ? AND session_type = 'study'
            """,
            (session_id,)
        )
        row = cur.fetchone()
        sessions = row["total_sessions"] if row else 0
        minutes = row["total_minutes"] if row else 0
        hours = round(minutes / 60.0, 1)

        return {
            "total_sessions": sessions,
            "total_minutes": minutes,
            "total_hours": hours
        }
    finally:
        conn.close()


# ========================================================
# STUDY STREAKS & BADGES
# ========================================================

def get_study_streak(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Computes consecutive active study days based on quiz attempts, tasks, and pomodoros.
    """
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        query = """
            SELECT DISTINCT date_str FROM (
                SELECT DATE(timestamp) AS date_str FROM quiz_attempts WHERE session_id = ?
                UNION
                SELECT DATE(completed_at) AS date_str FROM study_tasks WHERE session_id = ? AND is_completed = 1 AND completed_at IS NOT NULL
                UNION
                SELECT DATE(completed_at) AS date_str FROM pomodoro_sessions WHERE session_id = ?
            ) ORDER BY date_str DESC
        """
        cur.execute(query, (session_id, session_id, session_id))
        rows = [r["date_str"] for r in cur.fetchall() if r["date_str"]]

        if not rows:
            return {"current_streak": 0, "longest_streak": 0, "active_dates": []}

        date_objs = sorted([datetime.strptime(d, "%Y-%m-%d").date() for d in rows])
        today = datetime.now().date()

        # Compute longest streak
        longest_streak = 1
        current_run = 1
        for i in range(1, len(date_objs)):
            diff = (date_objs[i] - date_objs[i - 1]).days
            if diff == 1:
                current_run += 1
                if current_run > longest_streak:
                    longest_streak = current_run
            elif diff > 1:
                current_run = 1

        # Compute current streak
        current_streak = 0
        last_active = date_objs[-1]
        days_since_last = (today - last_active).days

        if days_since_last == 0 or days_since_last == 1:
            # Active today or yesterday, count backwards
            current_streak = 1
            check_date = last_active
            for prev_date in reversed(date_objs[:-1]):
                if (check_date - prev_date).days == 1:
                    current_streak += 1
                    check_date = prev_date
                else:
                    break

        return {
            "current_streak": current_streak,
            "longest_streak": max(longest_streak, current_streak),
            "active_dates": [d.strftime("%Y-%m-%d") for d in date_objs],
            "last_active": last_active.strftime("%Y-%m-%d") if date_objs else None
        }
    finally:
        conn.close()


def check_and_award_badges(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[str]:
    """Evaluates criteria and unlocks achievement badges for the session."""
    conn = get_connection(db_path)
    new_unlocks = []
    try:
        create_or_get_session(session_id, db_path=db_path)
        cur = conn.cursor()

        # Gather metrics
        cur.execute("SELECT COUNT(*) FROM quiz_attempts WHERE session_id = ?", (session_id,))
        attempt_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM study_tasks WHERE session_id = ? AND is_completed = 1", (session_id,))
        tasks_done = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*), COALESCE(SUM(duration_minutes), 0) FROM pomodoro_sessions WHERE session_id = ? AND session_type = 'study'", (session_id,))
        pomo_row = cur.fetchone()
        pomo_count = pomo_row[0]
        pomo_minutes = pomo_row[1]

        cur.execute("SELECT COUNT(*) FROM concept_mastery WHERE session_id = ? AND status = 'Provisionally Mastered'", (session_id,))
        mastered_concepts = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chapter_progress WHERE session_id = ? AND is_completed = 1", (session_id,))
        chapters_done = cur.fetchone()[0]

        streak_info = get_study_streak(session_id, db_path=db_path)
        curr_streak = streak_info["current_streak"]

        # Criteria mappings
        badge_criteria = [
            ("first_step", attempt_count >= 1 or tasks_done >= 1 or pomo_count >= 1),
            ("focus_pioneer", pomo_count >= 1),
            ("task_crusher", tasks_done >= 3),
            ("streak_3", curr_streak >= 3),
            ("streak_7", curr_streak >= 7),
            ("concept_master", mastered_concepts >= 1),
            ("logic_scholar", attempt_count >= 15),
            ("century_focus", pomo_minutes >= 100),
            ("syllabus_explorer", chapters_done >= 5)
        ]

        for badge_key, condition in badge_criteria:
            if condition:
                cur.execute(
                    "INSERT OR IGNORE INTO user_badges (session_id, badge_key) VALUES (?, ?)",
                    (session_id, badge_key)
                )
                if cur.rowcount > 0:
                    new_unlocks.append(badge_key)

        conn.commit()
        return new_unlocks
    finally:
        conn.close()


def get_user_badges(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Returns all badges with unlocked statuses and descriptions."""
    check_and_award_badges(session_id, db_path=db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT badge_key, unlocked_at FROM user_badges WHERE session_id = ?", (session_id,))
        unlocked = {r["badge_key"]: r["unlocked_at"] for r in cur.fetchall()}

        catalog = [
            {
                "key": "first_step",
                "title": "First Step",
                "icon": "🌱",
                "desc": "Completed your first quiz attempt, task, or focus session.",
                "category": "Getting Started"
            },
            {
                "key": "focus_pioneer",
                "title": "Focus Pioneer",
                "icon": "⏱️",
                "desc": "Successfully completed your first 25-minute Pomodoro focus block.",
                "category": "Productivity"
            },
            {
                "key": "task_crusher",
                "title": "Task Crusher",
                "icon": "📋",
                "desc": "Completed 3 or more planned daily study tasks.",
                "category": "Organization"
            },
            {
                "key": "streak_3",
                "title": "3-Day Momentum",
                "icon": "🔥",
                "desc": "Maintained a continuous 3-day active learning streak.",
                "category": "Consistency"
            },
            {
                "key": "streak_7",
                "title": "Week Warrior",
                "icon": "⚡",
                "desc": "Unstoppable dedication: 7 consecutive days of active study.",
                "category": "Consistency"
            },
            {
                "key": "concept_master",
                "title": "Concept Master",
                "icon": "🎯",
                "desc": "Achieved Bayesian provisional mastery on a syllabus concept.",
                "category": "Academic"
            },
            {
                "key": "logic_scholar",
                "title": "Logic Scholar",
                "icon": "🧠",
                "desc": "Answered 15+ practice questions across Digital Logic concepts.",
                "category": "Academic"
            },
            {
                "key": "century_focus",
                "title": "Century Focus",
                "icon": "🏆",
                "desc": "Accumulated over 100 minutes of deep study focus.",
                "category": "Productivity"
            },
            {
                "key": "syllabus_explorer",
                "title": "Syllabus Explorer",
                "icon": "📚",
                "desc": "Marked 5 or more chapters completed across course topics.",
                "category": "Curriculum"
            }
        ]

        for b in catalog:
            b["is_unlocked"] = b["key"] in unlocked
            b["unlocked_at"] = unlocked.get(b["key"])

        return catalog
    finally:
        conn.close()


# ========================================================
# WEEKLY STUDY ANALYTICS & DASHBOARD AGGREGATION
# ========================================================

def get_weekly_analytics(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Generates 7-day daily study statistics and productivity trends."""
    from datetime import timedelta
    today = datetime.now().date()
    date_range = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    date_strs = [d.strftime("%Y-%m-%d") for d in date_range]
    day_names = [d.strftime("%a") for d in date_range]

    conn = get_connection(db_path)
    try:
        cur = conn.cursor()

        # Daily Pomodoro Minutes
        cur.execute(
            """
            SELECT DATE(completed_at) AS day_str, COALESCE(SUM(duration_minutes), 0) AS mins
            FROM pomodoro_sessions
            WHERE session_id = ? AND session_type = 'study' AND DATE(completed_at) >= ?
            GROUP BY DATE(completed_at)
            """,
            (session_id, date_strs[0])
        )
        pomo_by_day = {r["day_str"]: r["mins"] for r in cur.fetchall()}

        # Daily Quiz Practice Time (sec -> min)
        cur.execute(
            """
            SELECT DATE(timestamp) AS day_str, COUNT(*) AS q_count, COALESCE(SUM(time_taken_sec), 0) AS secs
            FROM quiz_attempts
            WHERE session_id = ? AND DATE(timestamp) >= ?
            GROUP BY DATE(timestamp)
            """,
            (session_id, date_strs[0])
        )
        quiz_by_day = {r["day_str"]: {"count": r["q_count"], "mins": r["secs"] / 60.0} for r in cur.fetchall()}

        # Daily Tasks Completed
        cur.execute(
            """
            SELECT DATE(completed_at) AS day_str, COUNT(*) AS t_count
            FROM study_tasks
            WHERE session_id = ? AND is_completed = 1 AND completed_at IS NOT NULL AND DATE(completed_at) >= ?
            GROUP BY DATE(completed_at)
            """,
            (session_id, date_strs[0])
        )
        tasks_by_day = {r["day_str"]: r["t_count"] for r in cur.fetchall()}

        daily_data = []
        total_weekly_minutes = 0.0

        for d_str, d_name in zip(date_strs, day_names):
            p_mins = pomo_by_day.get(d_str, 0)
            q_info = quiz_by_day.get(d_str, {"count": 0, "mins": 0.0})
            q_mins = q_info["mins"]
            q_count = q_info["count"]
            t_count = tasks_by_day.get(d_str, 0)

            day_total_hours = round((p_mins + q_mins) / 60.0, 2)
            total_weekly_minutes += (p_mins + q_mins)

            daily_data.append({
                "date": d_str,
                "day_name": d_name,
                "pomodoro_hours": round(p_mins / 60.0, 2),
                "quiz_hours": round(q_mins / 60.0, 2),
                "total_hours": day_total_hours,
                "tasks_completed": t_count,
                "questions_attempted": q_count
            })

        total_weekly_hours = round(total_weekly_minutes / 60.0, 1)
        daily_average_hours = round(total_weekly_hours / 7.0, 1)

        # Find most productive day
        best_day = max(daily_data, key=lambda x: x["total_hours"]) if daily_data else None
        most_productive_label = f"{best_day['day_name']} ({best_day['total_hours']}h)" if (best_day and best_day["total_hours"] > 0) else "None yet"

        return {
            "daily_data": daily_data,
            "total_weekly_hours": total_weekly_hours,
            "daily_average_hours": daily_average_hours,
            "most_productive_day": most_productive_label
        }
    finally:
        conn.close()


def get_dashboard_summary(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Combines metrics for the executive dashboard overview cards."""
    # 1. Course Progress
    course_stats = get_course_progress_summary(session_id, db_path=db_path)

    # 2. Tasks
    pending_tasks = get_tasks(session_id, is_completed=0, db_path=db_path)
    completed_tasks = get_tasks(session_id, is_completed=1, db_path=db_path)

    # 3. Focus & Quiz Time
    pomo_stats = get_pomodoro_stats(session_id, db_path=db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(SUM(time_taken_sec), 0) FROM quiz_attempts WHERE session_id = ?", (session_id,))
        quiz_sec = cur.fetchone()[0]
        quiz_hours = quiz_sec / 3600.0
    finally:
        conn.close()

    total_study_hours = round(pomo_stats["total_hours"] + quiz_hours, 1)

    # 4. Streak
    streak_info = get_study_streak(session_id, db_path=db_path)

    # 5. Badges
    badges = get_user_badges(session_id, db_path=db_path)
    unlocked_badge_count = sum(1 for b in badges if b["is_unlocked"])

    return {
        "total_courses": course_stats["total_courses"],
        "completed_courses": course_stats["completed_courses"],
        "pending_tasks_count": len(pending_tasks),
        "completed_tasks_count": len(completed_tasks),
        "total_study_hours": total_study_hours,
        "overall_learning_progress": course_stats["overall_progress"],
        "current_streak": streak_info["current_streak"],
        "longest_streak": streak_info["longest_streak"],
        "pomodoro_sessions": pomo_stats["total_sessions"],
        "unlocked_badges": unlocked_badge_count,
        "total_badges": len(badges)
    }
