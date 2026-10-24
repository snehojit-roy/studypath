"""
Adaptive Learning Loop and Concept Mastery Service.
Maintains state transitions, updates mastery estimates persistently in SQLite,
and determines the next question adaptively based on error streaks and prerequisites.
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from database.db import (
    get_attempts,
    get_mastery_records,
    update_mastery_record,
    DEFAULT_DB_PATH
)
from models.baseline import BaselineGapDetector, ConceptStats
from services.quiz_service import QuizService


class MasteryService:
    """
    Coordinates mastery state transitions and adaptive question selection.
    """

    def __init__(
        self,
        quiz_service: Optional[QuizService] = None,
        db_path: str = DEFAULT_DB_PATH
    ):
        self.quiz_service = quiz_service or QuizService()
        self.db_path = db_path
        self.detector = BaselineGapDetector(
            alpha_prior=1.0,
            beta_prior=1.0,
            mastery_threshold=0.80,
            remediation_threshold=0.60
        )

    def update_session_mastery(self, session_id: str, concept_id: str) -> ConceptStats:
        """
        Recalculates baseline statistics for a concept and updates concept_mastery table.
        """
        attempts_df = get_attempts(session_id=session_id, concept_id=concept_id, db_path=self.db_path)
        concept_meta = self.quiz_service.get_concept(concept_id) or {"id": concept_id, "name": concept_id}
        stats = self.detector.evaluate_concept(concept_id, concept_meta.get("name", concept_id), attempts_df)

        update_mastery_record(
            session_id=session_id,
            concept_id=concept_id,
            attempts_count=stats.attempts_count,
            correct_count=stats.correct_count,
            error_count=stats.error_count,
            observed_accuracy=stats.observed_accuracy,
            smoothed_mastery=stats.smoothed_mastery,
            confidence=stats.wilson_lower_bound,
            evidence_level=stats.evidence_level,
            status=stats.status,
            consecutive_correct=stats.consecutive_correct,
            db_path=self.db_path
        )

        return stats

    def update_all_session_concepts(self, session_id: str) -> List[ConceptStats]:
        """Recalculates mastery across all syllabus concepts for the session."""
        all_concepts = self.quiz_service.get_all_concepts()
        attempts_df = get_attempts(session_id=session_id, db_path=self.db_path)
        stats_list = self.detector.evaluate_all_concepts(all_concepts, attempts_df)

        for s in stats_list:
            update_mastery_record(
                session_id=session_id,
                concept_id=s.concept_id,
                attempts_count=s.attempts_count,
                correct_count=s.correct_count,
                error_count=s.error_count,
                observed_accuracy=s.observed_accuracy,
                smoothed_mastery=s.smoothed_mastery,
                confidence=s.wilson_lower_bound,
                evidence_level=s.evidence_level,
                status=s.status,
                consecutive_correct=s.consecutive_correct,
                db_path=self.db_path
            )

        return stats_list

    def get_session_mastery_summary(self, session_id: str) -> Dict[str, Any]:
        """Returns structured overview of a student's performance."""
        attempts_df = get_attempts(session_id=session_id, db_path=self.db_path)
        mastery_df = get_mastery_records(session_id, db_path=self.db_path)

        total_attempts = len(attempts_df)
        total_correct = int(attempts_df["is_correct"].sum()) if total_attempts > 0 else 0
        overall_accuracy = (total_correct / total_attempts) if total_attempts > 0 else 0.0

        mastered_concepts = []
        needs_practice = []
        in_progress = []

        if not mastery_df.empty:
            for _, row in mastery_df.iterrows():
                cid = row["concept_id"]
                st = row["status"]
                c_meta = self.quiz_service.get_concept(cid)
                c_name = c_meta["name"] if c_meta else cid
                item = {
                    "concept_id": cid,
                    "concept_name": c_name,
                    "smoothed_mastery": row["smoothed_mastery"],
                    "observed_accuracy": row["observed_accuracy"],
                    "attempts": row["attempts_count"],
                    "evidence": row["evidence_level"],
                    "status": st
                }
                if st == "Provisionally Mastered":
                    mastered_concepts.append(item)
                elif st == "Needs Practice":
                    needs_practice.append(item)
                else:
                    in_progress.append(item)

        return {
            "session_id": session_id,
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "overall_accuracy": round(overall_accuracy, 3),
            "mastered_concepts": mastered_concepts,
            "needs_practice_concepts": needs_practice,
            "in_progress_concepts": in_progress,
            "mastery_df": mastery_df,
            "attempts_df": attempts_df
        }

    def select_adaptive_next_question(
        self,
        session_id: str,
        last_question_id: Optional[str] = None,
        last_is_correct: Optional[bool] = None
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Determines the next pedagogical step based on response history.
        Returns:
            (question_dict, adaptation_rationale)
        """
        attempts_df = get_attempts(session_id=session_id, db_path=self.db_path)
        attempted_qids = set(attempts_df["question_id"].tolist()) if not attempts_df.empty else set()
        all_questions = self.quiz_service.get_all_questions()

        # If student just answered incorrectly:
        if last_question_id and last_is_correct is False:
            last_q = self.quiz_service.get_question(last_question_id)
            if last_q:
                cid = last_q["concept_id"]
                # Count recent consecutive errors on this concept
                c_attempts = attempts_df[attempts_df["concept_id"] == cid]["is_correct"].tolist()
                consec_errors = 0
                for o in reversed(c_attempts):
                    if o == 0:
                        consec_errors += 1
                    else:
                        break

                # If 2 or more consecutive errors, check if a prerequisite concept exists
                c_meta = self.quiz_service.get_concept(cid)
                prereqs = c_meta.get("prerequisites", []) if c_meta else []
                if consec_errors >= 2 and prereqs:
                    prereq_id = prereqs[0]
                    prereq_meta = self.quiz_service.get_concept(prereq_id)
                    prereq_name = prereq_meta["name"] if prereq_meta else prereq_id
                    # Find candidate question in prerequisite
                    prereq_candidates = [
                        q for q in all_questions
                        if q["concept_id"] == prereq_id and q["id"] not in attempted_qids
                    ]
                    if not prereq_candidates:
                        prereq_candidates = [q for q in all_questions if q["concept_id"] == prereq_id]

                    if prereq_candidates:
                        chosen = min(prereq_candidates, key=lambda x: 0 if x.get("difficulty") == "basic" else 1)
                        rationale = (
                            f"Adaptive Remediation: You experienced {consec_errors} consecutive errors on "
                            f"'{c_meta.get('name', cid)}'. Stepping back to prerequisite concept: '{prereq_name}'."
                        )
                        return chosen, rationale

                # Same concept remediation question
                same_concept_candidates = [
                    q for q in all_questions
                    if q["concept_id"] == cid and q["id"] not in attempted_qids
                ]
                if not same_concept_candidates:
                    same_concept_candidates = [q for q in all_questions if q["concept_id"] == cid and q["id"] != last_question_id]
                
                if same_concept_candidates:
                    chosen = min(same_concept_candidates, key=lambda x: 0 if x.get("difficulty") == "basic" else 1)
                    rationale = (
                        f"Targeted Reinforcement: Re-testing concept '{c_meta.get('name', cid)}' "
                        f"with a differently framed question to check if the error pattern persists."
                    )
                    return chosen, rationale

        # If student was correct or first question:
        # Check concepts that need practice or are unassessed
        mastery_df = get_mastery_records(session_id, db_path=self.db_path)
        all_concepts = self.quiz_service.get_all_concepts()

        # Find concept with lowest smoothed mastery or fewest attempts
        target_concept_id = None
        if not mastery_df.empty:
            # Look for "Needs Practice" first
            needs_prac = mastery_df[mastery_df["status"] == "Needs Practice"]
            if not needs_prac.empty:
                target_concept_id = needs_prac.sort_values("smoothed_mastery").iloc[0]["concept_id"]
            else:
                # Look for "In Progress"
                in_prog = mastery_df[mastery_df["status"] == "In Progress"]
                if not in_prog.empty:
                    target_concept_id = in_prog.sort_values("smoothed_mastery").iloc[0]["concept_id"]

        if not target_concept_id:
            # Check for unassessed concepts
            assessed_cids = set(mastery_df["concept_id"].tolist()) if not mastery_df.empty else set()
            unassessed = [c["id"] for c in all_concepts if c["id"] not in assessed_cids]
            if unassessed:
                target_concept_id = unassessed[0]
            else:
                target_concept_id = all_concepts[0]["id"]

        c_meta = self.quiz_service.get_concept(target_concept_id)
        c_name = c_meta.get("name", target_concept_id)

        # Select unattempted question
        unattempted = [
            q for q in all_questions
            if q["concept_id"] == target_concept_id and q["id"] not in attempted_qids
        ]
        if unattempted:
            chosen = unattempted[0]
            rationale = f"Curriculum Progression: Testing '{c_name}' to advance concept mastery."
            return chosen, rationale

        # If all attempted, pick random remaining question
        remaining_pool = [q for q in all_questions if q["id"] not in attempted_qids]
        if remaining_pool:
            chosen = remaining_pool[0]
            rationale = "General Assessment: Expanding coverage across remaining syllabus questions."
            return chosen, rationale

        # If all syllabus questions have been attempted at least once:
        fallback_q = all_questions[0]
        rationale = "Comprehensive Review: All 16 question bank items completed; initiating reinforcement cycle."
        return fallback_q, rationale
