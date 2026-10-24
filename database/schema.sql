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

CREATE INDEX IF NOT EXISTS idx_attempts_session ON quiz_attempts(session_id);
CREATE INDEX IF NOT EXISTS idx_attempts_concept ON quiz_attempts(concept_id);
CREATE INDEX IF NOT EXISTS idx_mastery_session ON concept_mastery(session_id);
