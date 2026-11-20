CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alias TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    question_type TEXT NOT NULL,
    student_answer TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    is_correct INTEGER NOT NULL,
    misconception_tag TEXT,
    error_context TEXT,
    student_reasoning TEXT,
    time_taken_sec REAL DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS concept_mastery (
    session_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    attempts_count INTEGER NOT NULL DEFAULT 0,
    correct_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    observed_accuracy REAL NOT NULL DEFAULT 0.0,
    smoothed_mastery REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.0,
    evidence_level TEXT NOT NULL DEFAULT 'None',
    status TEXT NOT NULL DEFAULT 'Unassessed',
    consecutive_correct INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, concept_id),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    misconception_tag TEXT,
    provider TEXT NOT NULL,
    suspected_gap TEXT,
    explanation_text TEXT,
    worked_example TEXT,
    follow_up_question TEXT,
    prerequisite_hint TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- ========================================================
-- Study Tasks (Daily Study Planner)
-- ========================================================
CREATE TABLE IF NOT EXISTS study_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    course_id TEXT,
    deadline TEXT,
    priority TEXT NOT NULL DEFAULT 'Medium',
    is_completed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- ========================================================
-- Course Chapters & Progress Tracking
-- ========================================================
CREATE TABLE IF NOT EXISTS course_chapters (
    course_id TEXT NOT NULL,
    chapter_id TEXT NOT NULL,
    title TEXT NOT NULL,
    order_num INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (course_id, chapter_id)
);

CREATE TABLE IF NOT EXISTS chapter_progress (
    session_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    chapter_id TEXT NOT NULL,
    is_completed INTEGER NOT NULL DEFAULT 0,
    completed_at TIMESTAMP,
    PRIMARY KEY (session_id, course_id, chapter_id),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- ========================================================
-- Pomodoro Focus Sessions
-- ========================================================
CREATE TABLE IF NOT EXISTS pomodoro_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL DEFAULT 25,
    session_type TEXT NOT NULL DEFAULT 'study',
    notes TEXT,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- ========================================================
-- User Achievement Badges
-- ========================================================
CREATE TABLE IF NOT EXISTS user_badges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    badge_key TEXT NOT NULL,
    unlocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, badge_key),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_attempts_session ON quiz_attempts(session_id);
CREATE INDEX IF NOT EXISTS idx_attempts_concept ON quiz_attempts(concept_id);
CREATE INDEX IF NOT EXISTS idx_mastery_session ON concept_mastery(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON study_tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_chapter_progress_session ON chapter_progress(session_id);
CREATE INDEX IF NOT EXISTS idx_pomodoro_session ON pomodoro_sessions(session_id);
