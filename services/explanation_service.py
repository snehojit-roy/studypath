"""
Personalized Explanation Engine for StudyPath.
Provides structured diagnostic remediation (suspected gap, simple explanation,
worked example, targeted follow-up question, and prerequisite review).
Supports configurable LLM API with deterministic local rule-based fallback.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, Optional
from database.db import log_explanation, DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


# Deterministic Local Explanation Bank
RULE_BASED_EXPLANATIONS = {
    "demorgan_split_no_flip": {
        "suspected_gap": "Distributing complement/overbar over literals without changing the binary operator (e.g., treating (A.B)' as A'.B' instead of A' + B').",
        "explanation_text": "De Morgan's Theorem requires dual inversion: when you break the negation bar over terms, you MUST also invert the operator between them. AND (.) becomes OR (+), and OR (+) becomes AND (.). A simple mnemonic is: 'Break the line, change the sign.'",
        "worked_example": "Consider A=1, B=0:\n1. LHS: (A . B)' = (1 . 0)' = (0)' = 1.\n2. Your formula (A' . B'): (1)' . (0)' = 0 . 1 = 0 (Mismatch!).\n3. Correct De Morgan (A' + B'): (1)' + (0)' = 0 + 1 = 1 (Verified match!).",
        "follow_up_question": "What is the correct De Morgan expansion of (X + Y)'?",
        "follow_up_options": ["X' . Y'", "X' + Y'", "X . Y", "(X . Y)'"],
        "follow_up_answer": "X' . Y'",
        "prerequisite_hint": "Review the duality principle in Boolean Laws: AND and OR are dual operators linked by negation."
    },
    "demorgan_partial_inversion": {
        "suspected_gap": "Inverting the operator but neglecting to complement each variable, or dropping an existing bar during involution.",
        "explanation_text": "Every literal directly beneath the complemented overbar must have its inversion state toggled. If a literal was already negated (like Y'), negating it again gives (Y')' = Y by the Law of Involution.",
        "worked_example": "To simplify (P + Q')':\nStep 1: Change the '+' to '.' -> P' . (Q')'\nStep 2: Apply involution (Q')' = Q.\nFinal form = P' . Q.",
        "follow_up_question": "Simplify ((A')' + B')'. What is the result?",
        "follow_up_options": ["A' . B", "A . B", "A + B'", "A' + B"],
        "follow_up_answer": "A' . B",
        "prerequisite_hint": "Remember the double negation rule: NOT(NOT(A)) = A."
    },
    "absorption_neglect": {
        "suspected_gap": "Failing to apply the Absorption Law A + AB = A, or confusing it with the Redundancy Law A + A'B = A + B.",
        "explanation_text": "Whenever an expression has a literal added to a product containing that same literal (A + A.B), the single literal completely absorbs the product term. This happens because if A is 1, the entire sum is 1; if A is 0, both terms are 0.",
        "worked_example": "Expression: X + X . Y\nFactor out X: X . (1 + Y)\nSince 1 + Y = 1 (Null/Dominance law):\nX . (1) = X.",
        "follow_up_question": "Simplify: M + M . N . P",
        "follow_up_options": ["M", "M . N . P", "M + N + P", "N . P"],
        "follow_up_answer": "M",
        "prerequisite_hint": "Review the Null Element law: 1 + any_term = 1."
    },
    "distributive_addition_error": {
        "suspected_gap": "Applying ordinary real algebra assumptions to Boolean addition, or failing to recognize the second distributive law A + BC = (A + B)(A + C).",
        "explanation_text": "In Boolean algebra, OR distributes over AND just like AND distributes over OR! This means A + (B . C) = (A + B) . (A + C). Ordinary math does not allow this, but Boolean algebra does because variables are strictly binary {0, 1}.",
        "worked_example": "Simplify: A + A' . B\nApply distributive law: (A + A') . (A + B)\nSince A + A' = 1 (Complement law):\n1 . (A + B) = A + B.",
        "follow_up_question": "What does (X + Y) . (X + Z) simplify to when multiplied out?",
        "follow_up_options": ["X + Y . Z", "X . Y + Z", "X + Y + Z", "X . (Y + Z)"],
        "follow_up_answer": "X + Y . Z",
        "prerequisite_hint": "Review the Complement Laws: A + A' = 1 and A . A' = 0."
    },
    "idempotent_double_count": {
        "suspected_gap": "Treating Boolean addition as integer summation (e.g. A + A = 2A) or conjunction as powers (A . A = A^2).",
        "explanation_text": "In digital logic, voltages are either HIGH (1) or LOW (0). Combining HIGH with HIGH in an OR gate still yields HIGH (1 OR 1 = 1). Therefore, A + A = A, and A + A + A = A. There are no numerical coefficients in Boolean algebra.",
        "worked_example": "Expression: Z + Z + Z\nZ=1: 1 + 1 + 1 = 1 (not 3).\nZ=0: 0 + 0 + 0 = 0.\nIn all states, Z + Z + Z = Z.",
        "follow_up_question": "Simplify: B . B . B",
        "follow_up_options": ["B", "B^3", "3B", "1"],
        "follow_up_answer": "B",
        "prerequisite_hint": "Review the definition of the OR truth table: 0+0=0, 0+1=1, 1+0=1, 1+1=1."
    },
    "kmap_diagonal_grouping": {
        "suspected_gap": "Attempting to group diagonally adjacent 1s on a Karnaugh map.",
        "explanation_text": "K-map adjacency represents Gray code, where only ONE variable changes between neighboring cells. Diagonal cells differ by TWO variables simultaneously. Grouping diagonals fails to eliminate any variable and produces an invalid logic minimization.",
        "worked_example": "Cell (0,0) = A'B' and Cell (1,1) = AB.\nNotice both A toggled (0->1) and B toggled (0->1).\nNo common literal can be factored out. They must remain separate implicants unless covered by orthogonal neighbors.",
        "follow_up_question": "Which of the following cell pairings in a K-map CAN be validly grouped?",
        "follow_up_options": ["Cells sharing an edge horizontally or vertically", "Cells touching only at a diagonal corner", "Any two cells anywhere in the map", "Cells separated by a 0"],
        "follow_up_answer": "Cells sharing an edge horizontally or vertically",
        "prerequisite_hint": "Understand Gray code sequences: 00, 01, 11, 10 (each step changes only one bit)."
    },
    "kmap_non_power_two": {
        "suspected_gap": "Forming groups with sizes other than powers of two (e.g., grouping 3, 5, or 6 cells).",
        "explanation_text": "Each variable eliminated in a Boolean minimization halves the truth table possibilities. Therefore, valid groups in a K-map must strictly contain 2^k cells: 1, 2, 4, 8, or 16. A group of 6 or 3 cannot algebraically simplify cleanly.",
        "worked_example": "If you see three 1s in a row (e.g. cells 0, 1, 2):\nYou cannot make one group of 3. You must make a group of 2 (cells 0, 1) and another overlapping group of 2 (cells 0 or 2 with another neighbor).",
        "follow_up_question": "Can 6 adjacent 1s be combined into a single valid K-map group?",
        "follow_up_options": ["No, 6 is not a power of 2; use two overlapping groups of 4 if possible", "Yes, any rectangular block is valid", "Yes, if they are in the same row", "Yes, size 6 eliminates 3 variables"],
        "follow_up_answer": "No, 6 is not a power of 2; use two overlapping groups of 4 if possible",
        "prerequisite_hint": "Remember that 2^0=1, 2^1=2, 2^2=4, 2^3=8. Only these counts eliminate variables."
    },
    "kmap_wrap_around_miss": {
        "suspected_gap": "Failing to realize that the edges and corners of a K-map wrap around like a torus.",
        "explanation_text": "Because column 00 and column 10 differ by only the first bit, the far-left column and far-right column are physically adjacent. Similarly, the top row (00) and bottom row (10) are adjacent. The four corner cells together form a single valid group of 4!",
        "worked_example": "Corners m0(0000), m2(0010), m8(1000), m10(1010):\n- Rows (00 and 10) share B=0 (B')\n- Cols (00 and 10) share D=0 (D')\nCombined 4-cell group simplifies directly to B' . D'.",
        "follow_up_question": "Do cell (row 0, col 0) and cell (row 3, col 0) wrap around to be adjacent?",
        "follow_up_options": ["Yes, top and bottom edges are adjacent in Gray code", "No, they are on opposite sides of the grid", "Only if the center cells are 1", "No, wrap-around only works horizontally"],
        "follow_up_answer": "Yes, top and bottom edges are adjacent in Gray code",
        "prerequisite_hint": "Check the binary labels for top row (00) and bottom row (10): only the first bit differs!"
    },
    "nand_or_implementation": {
        "suspected_gap": "Believing an OR gate is made by putting an inverter at the output of a NAND gate, rather than inverting the inputs.",
        "explanation_text": "Putting an inverter at the output of a NAND gives (A . B)'' = A . B, which is an AND gate! To obtain an OR gate (A + B), use De Morgan: A + B = (A' . B')'. This means you must invert input A and input B first, then pass them into a NAND gate.",
        "worked_example": "Synthesizing OR using NAND:\nGate 1: NOT(A) = A NAND A = A'\nGate 2: NOT(B) = B NAND B = B'\nGate 3: A' NAND B' = (A' . B')' = (A')' + (B')' = A + B.\nTotal: 3 NAND gates.",
        "follow_up_question": "What logic function is produced if you feed A and B into a NAND gate, and then feed that output into a second NAND gate wired as an inverter?",
        "follow_up_options": ["AND gate (A . B)", "OR gate (A + B)", "NOR gate", "XOR gate"],
        "follow_up_answer": "AND gate (A . B)",
        "prerequisite_hint": "Review De Morgan's theorem: (A' . B')' = A + B."
    },
    "nor_and_confusion": {
        "suspected_gap": "Confusing the dual relationship between NAND and NOR logic synthesis.",
        "explanation_text": "NAND and NOR are duals. To make AND from NOR gates: A . B = (A + B)'' = (A' + B')' = A' NOR B'. Thus, NOR requires inverted inputs to make AND (3 NOR gates), just as NAND requires inverted inputs to make OR (3 NAND gates).",
        "worked_example": "NOR gate with inputs tied together:\nOutput = (X + X)' = X' (NOT gate).\nFeeding inverted inputs A' and B' into a NOR gives (A' + B')' = A . B (AND gate).",
        "follow_up_question": "How many 2-input NOR gates are needed to build a 2-input OR gate?",
        "follow_up_options": ["2 NOR gates (one NOR followed by an inverter)", "3 NOR gates", "1 NOR gate", "4 NOR gates"],
        "follow_up_answer": "2 NOR gates (one NOR followed by an inverter)",
        "prerequisite_hint": "Remember: Inverting NOR's output produces OR: ((A+B)')' = A+B."
    },
    "general_misconception": {
        "suspected_gap": "Incomplete algebraic reduction or unverified logical transformation.",
        "explanation_text": "Digital logic simplification requires applying canonical axioms step by step. When uncertain, verify using a truth table with all 2^n input combinations to ensure equivalence.",
        "worked_example": "To verify any Boolean equivalence F1 == F2, construct the truth table for all possible variable states {0, 1}. If outputs match on all rows, the simplification is correct.",
        "follow_up_question": "What is the primary method to conclusively prove any Boolean equivalence?",
        "follow_up_options": ["Constructing and comparing a complete truth table", "Estimating with decimal numbers", "Using intuition", "Checking only when all inputs are 1"],
        "follow_up_answer": "Constructing and comparing a complete truth table",
        "prerequisite_hint": "Review the fundamental truth tables for NOT, AND, and OR."
    }
}


class ExplanationService:
    """
    Delivers targeted, misconception-specific remedial content.
    Prioritizes local pedagogical rule-base for deterministic, fast execution.
    Can optionally call an LLM (OpenAI or Gemini) if API keys are configured.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def generate_explanation(
        self,
        session_id: str,
        concept_id: str,
        misconception_tag: Optional[str] = None,
        question_text: str = "",
        student_answer: str = "",
        expected_answer: str = "",
        error_context: str = "",
        force_local: bool = False
    ) -> Dict[str, Any]:
        """
        Generates structured 5-part explanation and logs it to SQLite.
        """
        tag = misconception_tag or "general_misconception"
        
        # 1. Try LLM if configured and not explicitly forced local
        if not force_local and (self.openai_key or self.gemini_key):
            try:
                llm_result = self._generate_llm_explanation(
                    concept_id=concept_id,
                    tag=tag,
                    question_text=question_text,
                    student_answer=student_answer,
                    expected_answer=expected_answer,
                    error_context=error_context
                )
                if llm_result:
                    log_explanation(
                        session_id=session_id,
                        concept_id=concept_id,
                        misconception_tag=tag,
                        provider=llm_result.get("provider", "llm"),
                        suspected_gap=llm_result.get("suspected_gap", ""),
                        explanation_text=llm_result.get("explanation_text", ""),
                        worked_example=llm_result.get("worked_example", ""),
                        follow_up_question=llm_result.get("follow_up_question", ""),
                        prerequisite_hint=llm_result.get("prerequisite_hint", ""),
                        db_path=self.db_path
                    )
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM explanation generation failed; falling back to rule base: {e}")

        # 2. Deterministic Local Fallback
        content = RULE_BASED_EXPLANATIONS.get(tag, RULE_BASED_EXPLANATIONS["general_misconception"])
        
        result = {
            "provider": "local_pedagogical_rules",
            "concept_id": concept_id,
            "misconception_tag": tag,
            "suspected_gap": content["suspected_gap"],
            "explanation_text": content["explanation_text"],
            "worked_example": content["worked_example"],
            "follow_up_question": content["follow_up_question"],
            "follow_up_options": content.get("follow_up_options", []),
            "follow_up_answer": content["follow_up_answer"],
            "prerequisite_hint": content.get("prerequisite_hint", "")
        }

        log_explanation(
            session_id=session_id,
            concept_id=concept_id,
            misconception_tag=tag,
            provider="local_pedagogical_rules",
            suspected_gap=result["suspected_gap"],
            explanation_text=result["explanation_text"],
            worked_example=result["worked_example"],
            follow_up_question=result["follow_up_question"],
            prerequisite_hint=result["prerequisite_hint"],
            db_path=self.db_path
        )

        return result

    def _generate_llm_explanation(
        self,
        concept_id: str,
        tag: str,
        question_text: str,
        student_answer: str,
        expected_answer: str,
        error_context: str
    ) -> Optional[Dict[str, Any]]:
        """Invokes LLM API with constrained prompt containing no student PII."""
        # Clean prompt with strictly subject material
        prompt = f"""
You are a senior Boolean Algebra tutor. A student made an error on the concept '{concept_id}'.
Question: {question_text}
Student's Answer: {student_answer}
Expected Answer: {expected_answer}
Diagnosed Error Pattern: {tag} ({error_context})

Provide a structured remediation in valid JSON with these exact keys:
{{
  "suspected_gap": "1-2 sentence description of the conceptual misunderstanding",
  "explanation_text": "Clear, concise mathematical explanation of why this rule works",
  "worked_example": "A minimal numerical or logical evaluation demonstrating the correction",
  "follow_up_question": "One targeted single-step verification question",
  "follow_up_options": ["Option A", "Option B", "Option C", "Option D"],
  "follow_up_answer": "Option A",
  "prerequisite_hint": "One sentence reviewing prerequisite concept"
}}
Return only valid JSON.
"""
        if self.gemini_key:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw = response.text.strip()
            if raw.startswith("```json"):
                raw = raw[7:-3].strip()
            data = json.loads(raw)
            data["provider"] = "gemini-2.5-flash"
            data["concept_id"] = concept_id
            data["misconception_tag"] = tag
            return data

        return None
