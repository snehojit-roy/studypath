"""
StudyPath — A Personalized Concept-Gap Tutor
College Project for CSET214: Data Analysis Using Python.
Main Streamlit Application.
"""
from __future__ import annotations

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import (
    init_db,
    create_or_get_session,
    get_attempts,
    get_mastery_records,
    get_explanations,
    export_attempts_csv,
    reset_database,
    load_benchmark_into_session
)
from services.quiz_service import QuizService
from services.explanation_service import ExplanationService
from services.mastery_service import MasteryService
from models.embedding_model import SemanticEmbeddingPipeline
from models.clustering import MistakeClusterer
from models.evaluation import Evaluator
from components.ui_helpers import (
    render_custom_css,
    create_mastery_bar_chart,
    create_pca_scatter_plot,
    create_similarity_heatmap,
    create_performance_timeline
)

# Page configuration
st.set_page_config(
    page_title="StudyPath — Concept-Gap Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Render custom CSS
render_custom_css()

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = "student_alpha"
if "student_alias" not in st.session_state:
    st.session_state.student_alias = "Alex (Student)"
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = "adaptive"
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "last_attempt_result" not in st.session_state:
    st.session_state.last_attempt_result = None
if "adaptive_rationale" not in st.session_state:
    st.session_state.adaptive_rationale = ""
if "question_start_time" not in st.session_state:
    st.session_state.question_start_time = time.time()

# Cached services to avoid re-instantiation overhead
@st.cache_resource
def get_services():
    quiz_svc = QuizService()
    expl_svc = ExplanationService()
    mastery_svc = MasteryService(quiz_service=quiz_svc)
    embedder = SemanticEmbeddingPipeline()
    clusterer = MistakeClusterer(distance_threshold=0.28)
    evaluator = Evaluator(embedder=embedder, clusterer=clusterer)
    return quiz_svc, expl_svc, mastery_svc, embedder, clusterer, evaluator

quiz_service, explanation_service, mastery_service, embedder, clusterer, evaluator = get_services()


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.markdown("## 🧠 **StudyPath**")
st.sidebar.caption("Personalized Concept-Gap Tutor | CSET214")

st.sidebar.markdown("---")
# Session Profile
st.sidebar.markdown("### 👤 Student Session")
col_s1, col_s2 = st.sidebar.columns([2, 1])
with col_s1:
    session_id_input = st.text_input("Session ID", value=st.session_state.session_id, key="sess_input")
with col_s2:
    if st.button("Switch"):
        st.session_state.session_id = session_id_input
        st.session_state.current_question = None
        st.session_state.last_attempt_result = None
        st.rerun()

create_or_get_session(st.session_state.session_id, alias=st.session_state.student_alias)

st.sidebar.markdown("---")
nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "🎯 Adaptive Practice & Quiz",
        "📊 Concept-Gap Mastery",
        "🔬 Semantic Clustering (ML)",
        "🧪 Scientific Evaluation",
        "💾 Session Data & Reset"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Pipeline")
st.sidebar.markdown(f"**Embedder:** `{embedder.backend}`")
st.sidebar.markdown(f"**Vector Dim:** `{embedder.embedding_dim}D`")
st.sidebar.markdown(f"**Explanation:** `Local Rule-Base (Deterministic)`")
st.sidebar.markdown(f"**Course Unit:** `Boolean Algebra (4 Concepts)`")


# -------------------------------------------------------------
# VIEW 1: ADAPTIVE PRACTICE & QUIZ
# -------------------------------------------------------------
if nav_choice == "🎯 Adaptive Practice & Quiz":
    st.markdown("## 🎯 Interactive Practice & Adaptive Concept-Gap Tutor")
    st.markdown(
        "StudyPath identifies recurring conceptual misconceptions from your answer patterns, "
        "reassesses your mastery estimate, and dynamically steers your next challenge."
    )

    # Ensure current question exists
    if st.session_state.current_question is None:
        next_q, rationale = mastery_service.select_adaptive_next_question(
            session_id=st.session_state.session_id
        )
        st.session_state.current_question = next_q
        st.session_state.adaptive_rationale = rationale
        st.session_state.question_start_time = time.time()
        st.session_state.last_attempt_result = None

    q = st.session_state.current_question

    if q is None:
        st.info("🎉 All question bank items completed for this session! Check the Mastery Dashboard.")
    else:
        # Show adaptive reason
        if st.session_state.adaptive_rationale:
            st.markdown(
                f"""
                <div class="callout-box">
                    <strong>🧭 Pedagogical Route:</strong> {st.session_state.adaptive_rationale}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Question Presentation Card
        c_meta = quiz_service.get_concept(q["concept_id"])
        c_name = c_meta["name"] if c_meta else q["concept_id"]

        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge-progress">{c_name}</span> &nbsp;
                <span class="badge-unassessed">Difficulty: {q.get('difficulty', 'basic').title()}</span> &nbsp;
                <span class="badge-unassessed">ID: {q['id']}</span>
                <h3 style="margin-top: 14px; color: #ffffff;">{q.get('title', 'Question')}</h3>
                <p style="font-size: 16px; line-height: 1.6; color: #f0f6fc;">{q['question_text']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Response input form
        with st.form("quiz_answer_form"):
            user_response = None
            if q["question_type"] == "multiple_choice":
                opts = q.get("options", [])
                opt_labels = [f"{opt['id']}) {opt['text']}" for opt in opts]
                selected_opt = st.radio("Select your answer:", opt_labels, index=0)
                user_response = selected_opt.split(")")[0].strip() if selected_opt else ""
            else:
                user_response = st.text_input(
                    "Enter your answer:",
                    placeholder="e.g. AB, A.B, 2, or single literal"
                )

            student_reasoning = st.text_input(
                "Your reasoning or step (optional, helps semantic grouping):",
                placeholder="e.g. 'I distributed the bar over each letter and kept the AND sign'"
            )

            col_sub, col_skip = st.columns([1, 4])
            with col_sub:
                submitted = st.form_submit_button("Submit Answer", type="primary")

        if submitted and user_response:
            time_spent = round(time.time() - st.session_state.question_start_time, 1)
            result = quiz_service.record_attempt(
                session_id=st.session_state.session_id,
                question_id=q["id"],
                student_answer=user_response,
                student_reasoning=student_reasoning,
                time_taken_sec=time_spent
            )

            # Update concept mastery
            stats = mastery_service.update_session_mastery(
                session_id=st.session_state.session_id,
                concept_id=q["concept_id"]
            )

            st.session_state.last_attempt_result = {
                "result": result,
                "stats": stats,
                "user_response": user_response,
                "reasoning": student_reasoning
            }

        # Feedback & Remediation
        if st.session_state.last_attempt_result is not None:
            attempt_data = st.session_state.last_attempt_result["result"]
            stats_data = st.session_state.last_attempt_result["stats"]
            is_corr = attempt_data["is_correct"]

            if is_corr:
                st.success(
                    f"✅ **Correct!** Well done. Verified answer: `{attempt_data['expected_answer']}`. "
                    f"Consecutive correct on this concept: **{stats_data.consecutive_correct}**."
                )
                st.info(f"💡 **Proof / Derivation:** {attempt_data['explanation']}")
            else:
                st.error(
                    f"❌ **Incorrect.** You selected `{attempt_data['student_answer']}`, "
                    f"but the verified answer is `{attempt_data['expected_answer']}`."
                )

                # Diagnostic & Explanation Box
                expl = explanation_service.generate_explanation(
                    session_id=st.session_state.session_id,
                    concept_id=attempt_data["concept_id"],
                    misconception_tag=attempt_data.get("misconception_tag"),
                    question_text=q["question_text"],
                    student_answer=attempt_data["student_answer"],
                    expected_answer=attempt_data["expected_answer"],
                    error_context=attempt_data.get("rationale", "")
                )

                st.markdown(
                    f"""
                    <div class="callout-remedy">
                        <h4 style="color: #bc8cff; margin-top: 0;">🔍 Diagnosed Concept Gap</h4>
                        <p><strong>Suspected Misunderstanding:</strong> {expl['suspected_gap']}</p>
                        <p><strong>Pedagogical Explanation:</strong> {expl['explanation_text']}</p>
                        <hr style="border-color: #30363d;"/>
                        <p><strong>Worked Step-by-Step Counter-Example:</strong></p>
                        <pre style="background: #0d1117; color: #58a6ff; padding: 10px; border-radius: 4px;">{expl['worked_example']}</pre>
                        <p><strong>Prerequisite Review:</strong> <em>{expl['prerequisite_hint']}</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Immediate follow-up interactive verification
                st.markdown("#### 🎯 Quick Verification Follow-Up")
                st.markdown(f"**{expl['follow_up_question']}**")
                if expl.get("follow_up_options"):
                    fu_ans = st.radio("Check your understanding:", expl["follow_up_options"], key="fu_radio")
                    if st.button("Check Follow-Up"):
                        if fu_ans == expl["follow_up_answer"]:
                            st.success("🎉 Correct! You've verified the rule.")
                        else:
                            st.warning(f"Not quite. Expected: {expl['follow_up_answer']}")

            # Button to advance
            if st.button("Next Adaptive Question ➡️", type="primary"):
                next_q, rationale = mastery_service.select_adaptive_next_question(
                    session_id=st.session_state.session_id,
                    last_question_id=attempt_data["question_id"],
                    last_is_correct=is_corr
                )
                st.session_state.current_question = next_q
                st.session_state.adaptive_rationale = rationale
                st.session_state.question_start_time = time.time()
                st.session_state.last_attempt_result = None
                st.rerun()


# -------------------------------------------------------------
# VIEW 2: CONCEPT-GAP MASTERY DASHBOARD
# -------------------------------------------------------------
elif nav_choice == "📊 Concept-Gap Mastery":
    st.markdown("## 📊 Concept-Gap Mastery Dashboard")
    st.markdown(
        "Combines observed empirical accuracy with a **Bayesian Laplace-smoothed mastery estimate** "
        "and **Wilson score interval** to avoid prematurely declaring mastery or failure on small samples."
    )

    # Recalculate stats for session
    stats_list = mastery_service.update_all_session_concepts(st.session_state.session_id)
    summary = mastery_service.get_session_mastery_summary(st.session_state.session_id)

    # Top Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{summary['total_attempts']}</div>
                <div class="metric-label">Questions Attempted</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{summary['overall_accuracy']:.1%}</div>
                <div class="metric-label">Overall Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #238636;">{len(summary['mastered_concepts'])}</div>
                <div class="metric-label">Provisionally Mastered</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #da3633;">{len(summary['needs_practice_concepts'])}</div>
                <div class="metric-label">Needs Practice</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Grouped Bar Chart
    stats_dicts = [s.to_dict() for s in stats_list]
    fig_mastery = create_mastery_bar_chart(stats_dicts)
    st.plotly_chart(fig_mastery, use_container_width=True)

    # Concept Breakdown Table
    st.markdown("### 📋 Statistical Breakdown by Concept")
    rows = []
    for s in stats_list:
        rows.append({
            "Concept ID": s.concept_id,
            "Concept Name": s.concept_name,
            "Attempts (N)": s.attempts_count,
            "Correct": s.correct_count,
            "Observed Accuracy": f"{s.observed_accuracy:.1%}",
            "Smoothed Mastery": f"{s.smoothed_mastery:.1%}",
            "95% Wilson Lower": f"{s.wilson_lower_bound:.1%}",
            "Evidence Tier": s.evidence_level,
            "Status": s.status,
            "Consecutive Correct": s.consecutive_correct
        })

    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True)

    # Performance Timeline
    attempts_df = summary["attempts_df"]
    if not attempts_df.empty:
        st.markdown("### 📈 Cumulative Learning Trajectory")
        fig_time = create_performance_timeline(attempts_df)
        st.plotly_chart(fig_time, use_container_width=True)

    # Drilldown into specific mistakes
    st.markdown("### 🔍 Drilldown: Detailed Attempt History")
    if not attempts_df.empty:
        c_filter = st.selectbox(
            "Filter by concept:",
            ["All Concepts"] + list(df_table["Concept Name"].unique())
        )
        filtered_attempts = attempts_df
        if c_filter != "All Concepts":
            cid = df_table[df_table["Concept Name"] == c_filter].iloc[0]["Concept ID"]
            filtered_attempts = attempts_df[attempts_df["concept_id"] == cid]

        display_cols = ["id", "question_id", "concept_id", "student_answer", "correct_answer", "is_correct", "misconception_tag", "time_taken_sec", "timestamp"]
        avail_cols = [c for c in display_cols if c in filtered_attempts.columns]
        st.dataframe(filtered_attempts[avail_cols], use_container_width=True)
    else:
        st.info("No attempts recorded yet for this session. Take a quiz or load benchmark data to see live history.")


# -------------------------------------------------------------
# VIEW 3: TRANSFORMER SEMANTIC CLUSTERING (ML)
# -------------------------------------------------------------
elif nav_choice == "🔬 Semantic Clustering (ML)":
    st.markdown("## 🔬 Transformer Embeddings & Semantic Clustering")
    st.markdown(
        "Investigates whether differently phrased student mistakes share recurring conceptual misunderstandings "
        "by mapping composite error contexts into **384-dimensional dense semantic vectors** using `all-MiniLM-L6-v2`."
    )

    # Collect mistakes from session
    attempts_df = get_attempts(session_id=st.session_state.session_id)
    mistakes_df = attempts_df[attempts_df["is_correct"] == 0] if not attempts_df.empty else pd.DataFrame()

    # Clustering configuration sidebar/expander
    with st.expander("⚙️ Clustering & Dimension Hyperparameters", expanded=False):
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            method = st.selectbox("Clustering Algorithm", ["agglomerative", "dbscan", "kmeans"], index=0)
        with c_m2:
            dist_threshold = st.slider("Cosine Distance Threshold (0.15 - 0.60)", 0.15, 0.60, 0.28, 0.01,
                                       help="Lower threshold demands tighter semantic similarity.")
        with c_m3:
            k_clusters = st.slider("K Clusters (KMeans only)", 2, 8, 4)

    if mistakes_df.empty or len(mistakes_df) < 2:
        st.warning(
            "⚠️ Fewer than 2 mistake records exist for the current session. "
            "To explore the clustering analysis with full benchmark data, click the button below:"
        )
        if st.button("📥 Load 20 Curated Benchmark Mistakes Into Session", type="primary"):
            count = load_benchmark_into_session(session_id=st.session_state.session_id)
            mastery_service.update_all_session_concepts(st.session_state.session_id)
            st.success(f"Loaded {count} benchmark attempts! Refreshing...")
            st.rerun()
    else:
        # Convert mistakes to list of records
        records = mistakes_df.to_dict(orient="records")
        for r in records:
            # Enrich with question text if missing
            q_info = quiz_service.get_question(r.get("question_id", ""))
            if q_info:
                r["question_text"] = q_info.get("question_text", "")
                r["title"] = q_info.get("title", "")

        # Generate embeddings
        with st.spinner("Generating 384-D sentence embeddings..."):
            texts = [embedder.construct_mistake_text(r) for r in records]
            embs = embedder.encode(texts)

        # Run clustering
        c_instance = MistakeClusterer(distance_threshold=dist_threshold)
        cluster_results = c_instance.cluster_mistakes(
            embeddings=embs,
            records=records,
            method=method,
            distance_threshold=dist_threshold,
            n_clusters=k_clusters if method == "kmeans" else None
        )

        # Top Metric Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{cluster_results['n_samples']}</div>
                    <div class="metric-label">Mistakes Analyzed</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{cluster_results['n_clusters']}</div>
                    <div class="metric-label">Discovered Clusters</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            sil_str = f"{cluster_results['silhouette']:.3f}" if cluster_results['silhouette'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #bc8cff;">{sil_str}</div>
                    <div class="metric-label">Silhouette Score</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #8b949e;">{len(cluster_results['outliers'])}</div>
                    <div class="metric-label">Outliers / Noise (-1)</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 2D PCA Scatter Plot
        augmented_records = []
        for c_id, c_data in cluster_results["cluster_diagnostics"].items():
            augmented_records.extend(c_data["members"])
        augmented_records.extend(cluster_results["outliers"])

        fig_pca = create_pca_scatter_plot(augmented_records, cluster_results["pca_variance_ratio"])
        st.plotly_chart(fig_pca, use_container_width=True)

        # Heatmap & Detailed Diagnostics
        tab_clusters, tab_heatmap, tab_baseline_comp = st.tabs([
            "📂 Cluster Diagnoses & Medoids",
            "🔥 Pairwise Similarity Heatmap",
            "⚖️ Semantic vs Concept-Tag Comparison"
        ])

        with tab_clusters:
            st.markdown("### Discovered Misconception Clusters")
            diagnostics = cluster_results["cluster_diagnostics"]
            if not diagnostics:
                st.info("No dense clusters formed. Adjust the distance threshold.")
            for c_id, c_info in diagnostics.items():
                medoid = c_info["representative_example"]
                with st.expander(
                    f"Cluster {c_id}: {c_info['dominant_misconception']} (Size: {c_info['size']} responses | Mean Sim: {c_info['mean_intra_similarity']:.2f})",
                    expanded=True
                ):
                    st.markdown(f"**Dominant Concept:** `{c_info['dominant_concept']}`")
                    st.markdown(f"**Intra-Cluster Cosine Similarity:** `{c_info['mean_intra_similarity']:.3f}`")
                    st.markdown("##### 📌 Representative Example (Cluster Medoid):")
                    st.markdown(
                        f"""
                        - **Question:** {medoid.get('question_text', medoid.get('question_id'))}
                        - **Student Answer:** `{medoid.get('student_answer')}`
                        - **Expected Answer:** `{medoid.get('correct_answer')}`
                        - **Diagnosed Error Rationale:** {medoid.get('error_context') or medoid.get('rationale')}
                        """
                    )
                    st.markdown("##### 👥 Constituent Responses Grouped in this Misconception:")
                    member_df = pd.DataFrame(c_info["members"])[
                        ["id", "question_id", "student_answer", "correct_answer", "misconception_tag", "student_reasoning"]
                    ]
                    st.dataframe(member_df, use_container_width=True)

            if cluster_results["outliers"]:
                with st.expander(f"⚠️ Isolated Outliers (Size: {len(cluster_results['outliers'])})"):
                    st.caption("Responses with low semantic similarity to all other mistakes. Not forced into any cluster.")
                    outlier_df = pd.DataFrame(cluster_results["outliers"])[
                        ["id", "question_id", "student_answer", "correct_answer", "error_context"]
                    ]
                    st.dataframe(outlier_df, use_container_width=True)

        with tab_heatmap:
            st.markdown("### Cosine Similarity Matrix Across Mistake Contexts")
            sim_matrix = cluster_results["similarity_matrix"]
            labels = [f"Q: {r.get('question_id')}, Ans: {r.get('student_answer')}" for r in records]
            fig_heat = create_similarity_heatmap(sim_matrix, labels)
            st.plotly_chart(fig_heat, use_container_width=True)

        with tab_baseline_comp:
            st.markdown("### Semantic Clustering vs. Predefined Concept-Tag Baseline")
            comp = c_instance.compare_with_baseline_concept_tags(records, cluster_results["labels"])
            st.markdown(
                f"""
                - **Concept Groups in Syllabus:** `{comp['num_concept_groups']}`
                - **Semantic Misconception Clusters:** `{comp['num_semantic_clusters']}`
                - **Adjusted Rand Index (ARI) Between Groupings:** `{comp['ari_semantic_vs_concept']:.4f}`
                - **Normalized Mutual Information (NMI):** `{comp['nmi_semantic_vs_concept']:.4f}`
                """
            )
            if comp.get("has_ground_truth"):
                st.markdown(
                    f"""
                    <div class="callout-box">
                        <strong>Benchmark Metric:</strong> Adjusted Rand Index vs Ground Truth Misconceptions<br/>
                        - Semantic Clustering ARI: <strong>{comp.get('ari_semantic_vs_ground_truth', 0):.4f}</strong><br/>
                        - Concept-Tag Baseline ARI: <strong>{comp.get('ari_concept_vs_ground_truth', 0):.4f}</strong><br/>
                        - <strong>Improvement Margin:</strong> <span style="color: #238636;">+{comp.get('semantic_improvement_ari', 0):+.4f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# -------------------------------------------------------------
# VIEW 4: SCIENTIFIC EVALUATION & BENCHMARK EXPERIMENT
# -------------------------------------------------------------
elif nav_choice == "🧪 Scientific Evaluation":
    st.markdown("## 🧪 Scientific Evaluation & Research Experiment")
    st.markdown(
        "> **Central Research Question:** *Can semantic embeddings of student mistakes improve the identification "
        "of recurring conceptual gaps compared with simple concept-tag-based error counting?*"
    )

    with st.spinner("Running benchmark experiment on 24 curated student answers..."):
        exp_results = evaluator.run_benchmark_experiment()

    comp = exp_results["metrics_comparison"]
    notes = exp_results["dataset_notes"]

    # Quantitative Comparison Metrics
    st.markdown("### 📊 Quantitative Benchmark Results")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #1f6feb;">{comp.get('ari_concept_vs_ground_truth', 0):.3f}</div>
                <div class="metric-label">Concept-Tag Baseline ARI</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #238636;">{comp.get('ari_semantic_vs_ground_truth', 0):.3f}</div>
                <div class="metric-label">Semantic Embedding ARI</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        diff = comp.get('semantic_improvement_ari', 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #58a6ff;">+{diff:.3f}</div>
                <div class="metric-label">Semantic Improvement Margin</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="callout-box">
            <strong>Experimental Verdict:</strong> {exp_results['experimental_verdict']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Qualitative Case Studies
    st.markdown("### 🔬 Qualitative Case Studies: Where Semantic Grouping Excels")
    for s in exp_results["case_study_successes"]:
        with st.expander(f"✅ {s['title']}", expanded=True):
            st.markdown(f"**Observed Pattern:** {s['description']}")
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown(f"❌ **Concept-Tag Baseline:** {s['concept_baseline_behavior']}")
            with c_b:
                st.markdown(f"✅ **Semantic Embeddings:** {s['semantic_clustering_behavior']}")

    # Limitations & Ethics
    st.markdown("### ⚠️ Limitations & Scientific Integrity")
    st.markdown(
        f"""
        <div class="callout-warning">
            <strong>Dataset Transparency & Integrity:</strong><br/>
            {notes['limitation_statement']}
        </div>
        """,
        unsafe_allow_html=True
    )

    for f in exp_results["case_study_failures"]:
        with st.expander(f"⚠️ Known Edge Case: {f['title']}"):
            st.markdown(f"**Observation:** {f['description']}")
            st.markdown(f"**Engineering Mitigation:** {f['mitigation']}")


# -------------------------------------------------------------
# VIEW 5: SESSION DATA & RESET
# -------------------------------------------------------------
elif nav_choice == "💾 Session Data & Reset":
    st.markdown("## 💾 Session Data Management & Audit")
    st.markdown("Manage persistent SQLite database records, export CSV reports, or preload demonstration data.")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown("### 📥 Benchmark Demonstration")
        st.markdown("Preload the curated 24-response benchmark dataset into your session for immediate evaluation.")
        if st.button("Load Benchmark Dataset Into Session", type="primary"):
            count = load_benchmark_into_session(session_id=st.session_state.session_id)
            mastery_service.update_all_session_concepts(st.session_state.session_id)
            st.success(f"Successfully loaded {count} benchmark responses into session `{st.session_state.session_id}`!")
            st.rerun()

    with c_b2:
        st.markdown("### 📤 Export Data")
        st.markdown("Download full attempt audit trail for statistical analysis in Python, R, or Excel.")
        csv_data = export_attempts_csv(session_id=st.session_state.session_id)
        st.download_button(
            label="Download Session Attempts CSV",
            data=csv_data,
            file_name=f"studypath_{st.session_state.session_id}_attempts.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("### 🗄️ Raw SQLite Audit Logs")
    tab_att, tab_mast, tab_exp = st.tabs(["Quiz Attempts", "Concept Mastery", "Explanation Logs"])

    with tab_att:
        df_att = get_attempts(session_id=st.session_state.session_id)
        if not df_att.empty:
            st.dataframe(df_att, use_container_width=True)
        else:
            st.info("No attempts in current session.")

    with tab_mast:
        df_m = get_mastery_records(session_id=st.session_state.session_id)
        if not df_m.empty:
            st.dataframe(df_m, use_container_width=True)
        else:
            st.info("No mastery records in current session.")

    with tab_exp:
        df_exp = get_explanations(session_id=st.session_state.session_id)
        if not df_exp.empty:
            st.dataframe(df_exp, use_container_width=True)
        else:
            st.info("No explanation logs in current session.")

    st.markdown("---")
    st.markdown("### ⚠️ Danger Zone: Database Reset")
    st.markdown("Permanently clears all session records and restores initial state.")
    confirm_reset = st.checkbox("I understand that this will erase all quiz attempts and mastery records.")
    if st.button("Reset Entire Database", type="secondary", disabled=not confirm_reset):
        success = reset_database(confirmation=True)
        if success:
            st.session_state.current_question = None
            st.session_state.last_attempt_result = None
            st.success("Database reset complete. All records cleared.")
            st.rerun()
