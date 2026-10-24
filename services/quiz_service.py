"""
Quiz engine service for question loading, validation, and attempt recording.
"""
from __future__ import annotations

import os
import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from database.db import log_attempt, DEFAULT_DB_PATH


class QuizService:
    def __init__(self, questions_path: Optional[str] = None, concepts_path: Optional[str] = None):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.questions_path = questions_path or os.path.join(base_dir, "questions.json")
        self.concepts_path = concepts_path or os.path.join(base_dir, "concepts.json")
        
        self.questions: List[Dict[str, Any]] = self._load_json(self.questions_path)
        self.concepts: List[Dict[str, Any]] = self._load_json(self.concepts_path)
        self.question_lookup: Dict[str, Dict[str, Any]] = {q["id"]: q for q in self.questions}
        self.concept_lookup: Dict[str, Dict[str, Any]] = {c["id"]: c for c in self.concepts}

    @staticmethod
    def _load_json(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        return self.concepts

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        return self.concept_lookup.get(concept_id)

    def get_all_questions(self) -> List[Dict[str, Any]]:
        return self.questions

    def get_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        return self.question_lookup.get(question_id)

    def generate_quiz_deck(
        self,
        num_questions: int = 6,
        concept_id: Optional[str] = None,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generates a balanced deck of questions, optionally filtered by concept."""
        if seed is not None:
            random.seed(seed)

        pool = [q for q in self.questions if concept_id is None or q["concept_id"] == concept_id]
        if not pool:
            pool = list(self.questions)

        if len(pool) <= num_questions:
            deck = list(pool)
            random.shuffle(deck)
            return deck

        # Balance by concept if selecting full mix
        if concept_id is None:
            by_concept: Dict[str, List[Dict[str, Any]]] = {}
            for q in pool:
                cid = q["concept_id"]
                by_concept.setdefault(cid, []).append(q)

            deck = []
            keys = list(by_concept.keys())
            while len(deck) < num_questions and any(by_concept.values()):
                for cid in keys:
                    if by_concept[cid] and len(deck) < num_questions:
                        chosen = random.choice(by_concept[cid])
                        by_concept[cid].remove(chosen)
                        deck.append(chosen)
            random.shuffle(deck)
            return deck

        return random.sample(pool, num_questions)

    def normalize_answer(self, text: str) -> str:
        """Normalizes user response string for robust comparison."""
        if text is None:
            return ""
        s = str(text).strip()
        # Remove redundant whitespace and internal spaces
        s = re.sub(r"\s+", "", s)
        # Normalize multiplication representations: 'A*B', 'A.B' -> 'AB'
        s = s.replace("*", "").replace(".", "")
        return s.upper()

    def validate_answer(
        self,
        question_id: str,
        student_response: str
    ) -> Tuple[bool, Optional[str], Optional[str], str]:
        """
        Validates student response against verified answer key.
        Returns:
            (is_correct, misconception_tag, error_rationale, expected_answer)
        """
        q = self.get_question(question_id)
        if not q:
            return False, None, "Question not found", ""

        expected = q.get("correct_answer", "")
        q_type = q.get("question_type", "multiple_choice")
        distractor_map = q.get("distractor_misconceptions", {})

        if q_type == "multiple_choice":
            clean_resp = str(student_response).strip().upper()
            is_correct = (clean_resp == expected.upper())
            
            if is_correct:
                return True, None, "Correct answer.", expected

            # Check if option key or text maps to a distractor misconception
            distractor_info = distractor_map.get(clean_resp)
            if not distractor_info:
                # Check by option text match
                for opt in q.get("options", []):
                    if opt.get("text", "").strip() == str(student_response).strip():
                        distractor_info = distractor_map.get(opt.get("id"))
                        break

            tag = distractor_info.get("tag") if distractor_info else "general_misconception"
            rationale = distractor_info.get("rationale") if distractor_info else "Incorrect option chosen."
            return False, tag, rationale, expected

        else:
            # Short-answer question
            norm_student = self.normalize_answer(student_response)
            norm_expected = self.normalize_answer(expected)

            is_correct = (norm_student == norm_expected)
            if is_correct:
                return True, None, "Correct answer.", expected

            # Check distractor misconceptions
            tag = "calculation_or_algebraic_error"
            rationale = "Expression does not match verified minimal form."
            for wrong_key, info in distractor_map.items():
                if self.normalize_answer(wrong_key) == norm_student:
                    tag = info.get("tag", tag)
                    rationale = info.get("rationale", rationale)
                    break

            return False, tag, rationale, expected

    def record_attempt(
        self,
        session_id: str,
        question_id: str,
        student_answer: str,
        student_reasoning: Optional[str] = None,
        time_taken_sec: float = 0.0,
        db_path: str = DEFAULT_DB_PATH
    ) -> Dict[str, Any]:
        """Validates response and persists the outcome to SQLite database."""
        q = self.get_question(question_id)
        if not q:
            raise ValueError(f"Question ID {question_id} not found in question bank.")

        is_correct, misconception_tag, rationale, expected = self.validate_answer(
            question_id, student_answer
        )

        attempt_id = log_attempt(
            session_id=session_id,
            question_id=question_id,
            concept_id=q["concept_id"],
            question_type=q.get("question_type", "multiple_choice"),
            student_answer=student_answer,
            correct_answer=expected,
            is_correct=is_correct,
            misconception_tag=misconception_tag if not is_correct else None,
            error_context=rationale if not is_correct else None,
            student_reasoning=student_reasoning,
            time_taken_sec=time_taken_sec,
            db_path=db_path
        )

        return {
            "attempt_id": attempt_id,
            "session_id": session_id,
            "question_id": question_id,
            "concept_id": q["concept_id"],
            "is_correct": is_correct,
            "expected_answer": expected,
            "student_answer": student_answer,
            "misconception_tag": misconception_tag,
            "rationale": rationale,
            "explanation": q.get("explanation", ""),
            "follow_up_hint": q.get("follow_up_hint", "")
        }
