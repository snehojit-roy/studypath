"""
Database module for StudyPath.
"""
from .db import (
    init_db,
    get_connection,
    create_or_get_session,
    log_attempt,
    get_attempts,
    get_mastery_records,
    update_mastery_record,
    log_explanation,
    get_explanations,
    export_attempts_csv,
    reset_database,
    load_benchmark_into_session,
    DEFAULT_DB_PATH
)

__all__ = [
    "init_db",
    "get_connection",
    "create_or_get_session",
    "log_attempt",
    "get_attempts",
    "get_mastery_records",
    "update_mastery_record",
    "log_explanation",
    "get_explanations",
    "export_attempts_csv",
    "reset_database",
    "load_benchmark_into_session",
    "DEFAULT_DB_PATH"
]
