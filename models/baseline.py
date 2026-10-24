"""
Baseline Concept-Gap Detection Module.
Provides transparent statistical performance estimates, Laplace smoothing,
Wilson score intervals, recent-trend analysis, and evidence-level tiers.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ConceptStats:
    concept_id: str
    concept_name: str
    attempts_count: int
    correct_count: int
    error_count: int
    observed_accuracy: float
    error_rate: float
    smoothed_mastery: float
    wilson_lower_bound: float
    recent_accuracy: float
    consecutive_correct: int
    evidence_level: str
    status: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaselineGapDetector:
    """
    Transparent statistical baseline calculator.
    Distinguishes observed empirical accuracy from estimated mastery,
    preventing premature mastery or failure labels on sparse samples.
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        mastery_threshold: float = 0.80,
        remediation_threshold: float = 0.60,
        z_score: float = 1.96  # 95% confidence
    ):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.mastery_threshold = mastery_threshold
        self.remediation_threshold = remediation_threshold
        self.z_score = z_score

    def calculate_laplace_mastery(self, correct: int, total: int) -> float:
        """
        Beta-Binomial posterior mean with uniform Beta(alpha, beta) prior.
        Formula: (C + alpha) / (N + alpha + beta)
        """
        return (correct + self.alpha_prior) / (total + self.alpha_prior + self.beta_prior)

    def calculate_wilson_lower_bound(self, correct: int, total: int) -> float:
        """
        Wilson score interval 95% lower bound for binomial proportion.
        Penalizes small sample sizes conservatively.
        """
        if total <= 0:
            return 0.0
        p_hat = correct / total
        z = self.z_score
        denominator = 1.0 + (z ** 2) / total
        center = p_hat + (z ** 2) / (2.0 * total)
        spread = z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z ** 2) / (4.0 * (total ** 2)))
        return max(0.0, (center - spread) / denominator)

    def calculate_recent_accuracy(self, outcomes: List[int], window_size: int = 5, decay: float = 0.7) -> float:
        """
        Exponentially weighted recent performance over the last window_size attempts.
        Recent attempts carry higher weight.
        """
        if not outcomes:
            return 0.0
        window = outcomes[-window_size:]
        n = len(window)
        # Weights: w_i = decay^(n - 1 - i)
        weights = [decay ** (n - 1 - i) for i in range(n)]
        norm_weights = np.array(weights) / sum(weights)
        return float(np.dot(norm_weights, window))

    def determine_evidence_level(self, total_attempts: int) -> str:
        """Categorizes data support level to prevent overconfident claims."""
        if total_attempts == 0:
            return "None"
        elif total_attempts <= 2:
            return "Low (Preliminary)"
        elif total_attempts <= 5:
            return "Moderate (Formative)"
        else:
            return "Strong (High Confidence)"

    def calculate_consecutive_correct(self, outcomes: List[int]) -> int:
        """Counts continuous correct answers from the most recent attempt backwards."""
        count = 0
        for val in reversed(outcomes):
            if val == 1:
                count += 1
            else:
                break
        return count

    def evaluate_concept(
        self,
        concept_id: str,
        concept_name: str,
        attempts_df: pd.DataFrame
    ) -> ConceptStats:
        """Evaluates baseline statistics for a single concept from attempt history."""
        c_df = attempts_df[attempts_df["concept_id"] == concept_id] if not attempts_df.empty else pd.DataFrame()
        
        total = len(c_df)
        if total == 0:
            return ConceptStats(
                concept_id=concept_id,
                concept_name=concept_name,
                attempts_count=0,
                correct_count=0,
                error_count=0,
                observed_accuracy=0.0,
                error_rate=0.0,
                smoothed_mastery=0.50,
                wilson_lower_bound=0.0,
                recent_accuracy=0.0,
                consecutive_correct=0,
                evidence_level="None",
                status="Unassessed",
                rationale="No questions attempted yet for this concept."
            )

        outcomes = c_df["is_correct"].astype(int).tolist()
        correct = sum(outcomes)
        errors = total - correct
        observed_acc = correct / total
        error_rate = errors / total
        smoothed_mastery = self.calculate_laplace_mastery(correct, total)
        wilson_lower = self.calculate_wilson_lower_bound(correct, total)
        recent_acc = self.calculate_recent_accuracy(outcomes)
        consecutive_corr = self.calculate_consecutive_correct(outcomes)
        evidence = self.determine_evidence_level(total)

        # Explicit status transition logic
        if total >= 3 and smoothed_mastery >= self.mastery_threshold and consecutive_corr >= 2:
            status = "Provisionally Mastered"
            rationale = (
                f"Sufficient evidence (N={total}) with smoothed mastery {smoothed_mastery:.1%} "
                f"and {consecutive_corr} consecutive correct responses."
            )
        elif total >= 1 and smoothed_mastery < self.remediation_threshold:
            status = "Needs Practice"
            rationale = (
                f"Elevated error rate ({error_rate:.1%}) across N={total} attempts. "
                f"Targeted remediation recommended."
            )
        else:
            status = "In Progress"
            rationale = (
                f"Formative phase (N={total}). Observed accuracy is {observed_acc:.1%}; "
                f"more trials required to confirm mastery stability."
            )

        return ConceptStats(
            concept_id=concept_id,
            concept_name=concept_name,
            attempts_count=total,
            correct_count=correct,
            error_count=errors,
            observed_accuracy=round(observed_acc, 3),
            error_rate=round(error_rate, 3),
            smoothed_mastery=round(smoothed_mastery, 3),
            wilson_lower_bound=round(wilson_lower, 3),
            recent_accuracy=round(recent_acc, 3),
            consecutive_correct=consecutive_corr,
            evidence_level=evidence,
            status=status,
            rationale=rationale
        )

    def evaluate_all_concepts(
        self,
        concepts_list: List[Dict[str, Any]],
        attempts_df: pd.DataFrame
    ) -> List[ConceptStats]:
        """Evaluates baseline statistics for all concepts in the syllabus."""
        results = []
        for c in concepts_list:
            cid = c["id"]
            cname = c.get("name", cid)
            stats = self.evaluate_concept(cid, cname, attempts_df)
            results.append(stats)
        return results
