"""
StudyPath — Modern Student Productivity & Adaptive Learning Platform
Enhanced SaaS platform featuring:
1. Executive Learning & Productivity Dashboard
2. Daily Study Task Planner & Management
3. Course & Chapter Progress Tracker
4. Weekly Study Analytics (Plotly)
5. Pomodoro Focus Timer & Session Tracker
6. Study Streaks & Achievement Badges
7. Adaptive Practice & Concept-Gap Tutor (Original Research Component)
8. Bayesian Concept-Gap Mastery Engine
9. 384-D Semantic Misconception Clustering (sentence-transformers)
10. Scientific Evaluation & Benchmarks
11. Session Audit & SQLite Persistence
"""
from __future__ import annotations

import os
import sys
import json
import time
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.db import (
    init_db,
    create_or_get_session,
    get_attempts,
    get_mastery_records,
    get_explanations,
    export_attempts_csv,
    reset_database,
    load_benchmark_into_session,
    add_task,
    get_tasks,
    toggle_task_completion,
    delete_task,
    get_courses_and_chapters,
    toggle_chapter_progress,
    get_course_progress_summary,
    log_pomodoro_session,
    get_pomodoro_stats,
    get_study_streak,
    check_and_award_badges,
    get_user_badges,
    get_weekly_analytics,
    get_dashboard_summary
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
    create_performance_timeline,
    create_daily_study_hours_chart,
    create_weekly_productivity_chart,
    create_tasks_completed_chart,
    create_course_progress_donut
)

# -------------------------------------------------------------
# APPLICATION CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="StudyPath — Modern Student Productivity Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema & default chapters
init_db()

# Inject modern dark-theme SaaS stylesheet
render_custom_css()

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
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

# Pomodoro state variables
if "pomo_mode" not in st.session_state:
    st.session_state.pomo_mode = "study"  # 'study' or 'break'
if "pomo_seconds_left" not in st.session_state:
    st.session_state.pomo_seconds_left = 25 * 60
if "pomo_is_running" not in st.session_state:
    st.session_state.pomo_is_running = False

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

# Ensure current session exists in SQLite
create_or_get_session(st.session_state.session_id, alias=st.session_state.student_alias)

# Check and update badges
check_and_award_badges(st.session_state.session_id)

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.markdown("## 🧠 **StudyPath**")
st.sidebar.caption("Student Productivity & Learning Management")

# Quick Streak & Badge pill in sidebar
sidebar_streak = get_study_streak(st.session_state.session_id)
sidebar_badges = get_user_badges(st.session_state.session_id)
unlocked_b_count = sum(1 for b in sidebar_badges if b["is_unlocked"])

st.sidebar.markdown(
    f"""
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-around; text-align: center;">
        <div>
            <span style="font-size: 16px;">🔥</span> <strong style="color: #ff7b72;">{sidebar_streak['current_streak']}d</strong>
            <div style="font-size: 10px; color: #8b949e; text-transform: uppercase;">Streak</div>
        </div>
        <div style="border-left: 1px solid #30363d; padding-left: 12px;">
            <span style="font-size: 16px;">🏆</span> <strong style="color: #d29922;">{unlocked_b_count}</strong>
            <div style="font-size: 10px; color: #8b949e; text-transform: uppercase;">Badges</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
# Session Profile
st.sidebar.markdown("### 👤 Student Session")
col_s1, col_s2 = st.sidebar.columns([2, 1])
with col_s1:
    session_id_input = st.text_input("Session ID", value=st.session_state.session_id, key="sess_input")
with col_s2:
    if st.button("Switch"):
        st.session_state.session_id = session_id_input.strip() or "student_alpha"
        st.session_state.current_question = None
        st.session_state.last_attempt_result = None
        st.rerun()

st.sidebar.markdown("---")
nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📅 Daily Study Planner",
        "📚 Course Progress Tracker",
        "📈 Weekly Study Analytics",
        "⏱️ Pomodoro Focus Timer",
        "🏆 Study Streaks & Badges",
        "🎯 Adaptive Practice & Quiz",
        "📉 Concept-Gap Mastery",
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
st.sidebar.markdown(f"**Database:** `SQLite (Persistent)`")
st.sidebar.markdown(f"**Course Unit:** `Digital Logic (4 Modules)`")


# =============================================================
# FEATURE 1: IMPROVED DASHBOARD
# =============================================================
if nav_choice == "📊 Dashboard":
    st.markdown("## 📊 Executive Learning & Productivity Dashboard")
    st.markdown(
        f"Welcome back, **{st.session_state.student_alias}**! 👋 "
        "Here is your daily study pulse, course progress, and active tasks."
    )

    # Dynamic summary stats from SQLite
    dash_stats = get_dashboard_summary(st.session_state.session_id)

    # 5 Attractive Statistics Cards
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    with kpi_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Courses</div>
                <div class="metric-value">{dash_stats['total_courses']}</div>
                <div class="metric-sub">Syllabus Modules</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Completed Courses</div>
                <div class="metric-value" style="color: #3fb950;">{dash_stats['completed_courses']} / {dash_stats['total_courses']}</div>
                <div class="metric-sub">100% Finished</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Pending Tasks</div>
                <div class="metric-value" style="color: {'#f85149' if dash_stats['pending_tasks_count'] > 0 else '#3fb950'};">{dash_stats['pending_tasks_count']}</div>
                <div class="metric-sub">{dash_stats['completed_tasks_count']} Completed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Study Hours</div>
                <div class="metric-value" style="color: #bc8cff;">{dash_stats['total_study_hours']}h</div>
                <div class="metric-sub">{dash_stats['pomodoro_sessions']} Focus Blocks</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Progress</div>
                <div class="metric-value" style="color: #58a6ff;">{dash_stats['overall_learning_progress']}%</div>
                <div class="progress-track" style="margin-top: 8px;">
                    <div class="progress-bar-fill" style="width: {dash_stats['overall_learning_progress']}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Middle Row: Quick Actions & Streak Banner
    col_dash_left, col_dash_right = st.columns([3, 2])

    with col_dash_left:
        st.markdown("### 📚 Syllabus Completion Snapshot")
        courses = get_courses_and_chapters(st.session_state.session_id)
        for c in courses:
            pct = c["progress_percent"]
            st.markdown(
                f"""
                <div class="course-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #f0f6fc; font-size: 15px;">{c['course_name']}</strong>
                        <span style="font-weight: 700; color: {'#3fb950' if pct == 100 else '#58a6ff'};">{pct}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-bar-fill" style="width: {pct}%;"></div>
                    </div>
                    <div style="font-size: 12px; color: #8b949e; display: flex; justify-content: space-between;">
                        <span>{c['completed_count']} of {c['total_count']} chapters mastered</span>
                        <span>{'✅ Fully Completed' if c['is_fully_completed'] else 'In Progress'}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Quick due tasks preview
        st.markdown("### 📋 Upcoming Study Tasks")
        pending_tasks = get_tasks(st.session_state.session_id, is_completed=0)
        if pending_tasks:
            for t in pending_tasks[:3]:
                p_class = f"priority-{t['priority'].lower()}"
                due_str = f"Due: {t['deadline']}" if t['deadline'] else "No deadline"
                st.markdown(
                    f"""
                    <div class="task-card">
                        <div>
                            <span class="{p_class}">{t['priority']}</span> &nbsp;
                            <strong style="color: #f0f6fc;">{t['title']}</strong>
                            <div style="font-size: 12px; color: #8b949e; margin-top: 4px;">{due_str}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="callout-success">
                    🎉 <strong>All caught up!</strong> You have no pending study tasks right now.
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_dash_right:
        # Streak Card
        st.markdown("### 🔥 Study Momentum")
        st.markdown(
            f"""
            <div class="streak-container">
                <div style="font-size: 40px; margin-bottom: 4px;">🔥</div>
                <div class="streak-number">{dash_stats['current_streak']} <span style="font-size: 20px; font-weight: 500; color: #c9d1d9;">Days</span></div>
                <div style="color: #f0f6fc; font-weight: 600; margin-top: 6px;">Continuous Study Streak</div>
                <div style="font-size: 13px; color: #8b949e; margin-top: 4px;">Personal Best: <strong>{dash_stats['longest_streak']} days</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Quick Focus Logging
        st.markdown("### ⏱️ Quick Focus Block")
        st.markdown("Completed 25 minutes of off-screen textbook reading or lecture review?")
        if st.button("➕ Log 25m Completed Focus Session", type="primary"):
            log_pomodoro_session(st.session_state.session_id, 25, "study", "Quick log from Dashboard")
            st.success("🎉 Focus session logged! Study hours and streak updated.")
            st.rerun()

        st.markdown("<br/>", unsafe_allow_html=True)

        # Quick Badges Highlight
        st.markdown("### 🏆 Badges Unlocked")
        badges = get_user_badges(st.session_state.session_id)
        unlocked = [b for b in badges if b["is_unlocked"]]
        if unlocked:
            b_cols = st.columns(min(3, len(unlocked)))
            for i, b in enumerate(unlocked[:3]):
                with b_cols[i]:
                    st.markdown(
                        f"""
                        <div class="badge-card badge-card-unlocked">
                            <div class="badge-icon">{b['icon']}</div>
                            <div class="badge-title">{b['title']}</div>
                            <div class="badge-date">Unlocked</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No badges unlocked yet. Answer a quiz question or finish a Pomodoro to claim your first badge!")


# =============================================================
# FEATURE 2: DAILY STUDY PLANNER
# =============================================================
elif nav_choice == "📅 Daily Study Planner":
    st.markdown("## 📅 Daily Study Task Planner")
    st.markdown(
        "Organize, prioritize, and track your daily study assignments, revision milestones, and problem sets."
    )

    # Task creation form
    with st.expander("➕ **Add New Study Task**", expanded=True):
        with st.form("new_task_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns([3, 2])
            with f_col1:
                task_title = st.text_input("Task Title *", placeholder="e.g. Solve 5 K-Map minimization problems")
            with f_col2:
                course_select = st.selectbox(
                    "Related Course / Concept",
                    [
                        "General Study",
                        "Boolean Laws & Basic Simplification",
                        "De Morgan's Theorems",
                        "Karnaugh Map (K-Map) Minimization",
                        "Universal Logic (NAND/NOR Implementation)"
                    ]
                )

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                task_deadline = st.date_input("Target Deadline", value=date.today())
            with d_col2:
                task_priority = st.selectbox("Priority Level", ["Medium", "High", "Low"], index=0)

            submitted_task = st.form_submit_button("Add Task to Planner", type="primary")

            if submitted_task:
                if not task_title.strip():
                    st.error("Please enter a valid task title.")
                else:
                    course_mapping = {
                        "Boolean Laws & Basic Simplification": "boolean_laws",
                        "De Morgan's Theorems": "demorgan",
                        "Karnaugh Map (K-Map) Minimization": "karnaugh_maps",
                        "Universal Logic (NAND/NOR Implementation)": "universal_logic"
                    }
                    cid = course_mapping.get(course_select)
                    add_task(
                        session_id=st.session_state.session_id,
                        title=task_title,
                        course_id=cid,
                        deadline=task_deadline.strftime("%Y-%m-%d"),
                        priority=task_priority
                    )
                    st.success(f"Task **{task_title}** added successfully!")
                    st.rerun()

    # Retrieve tasks
    pending_tasks = get_tasks(st.session_state.session_id, is_completed=0)
    completed_tasks = get_tasks(st.session_state.session_id, is_completed=1)

    tab_pending, tab_completed = st.tabs([
        f"⏳ Pending Tasks ({len(pending_tasks)})",
        f"✅ Completed Tasks ({len(completed_tasks)})"
    ])

    with tab_pending:
        if not pending_tasks:
            st.markdown(
                """
                <div class="callout-success" style="text-align: center; padding: 30px;">
                    <div style="font-size: 36px; margin-bottom: 8px;">🎉</div>
                    <strong style="font-size: 18px; color: #f0f6fc;">All caught up!</strong>
                    <p style="color: #8b949e; margin-top: 6px;">You have no pending study tasks. Create one above to plan your next session.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for task in pending_tasks:
                tid = task["id"]
                p_class = f"priority-{task['priority'].lower()}"
                
                # Calculate deadline status
                due_display = "No deadline"
                is_overdue = False
                if task["deadline"]:
                    try:
                        d_date = datetime.strptime(task["deadline"], "%Y-%m-%d").date()
                        days_diff = (d_date - date.today()).days
                        if days_diff < 0:
                            due_display = f"⚠️ Overdue ({abs(days_diff)}d ago)"
                            is_overdue = True
                        elif days_diff == 0:
                            due_display = "🔥 Due Today"
                        elif days_diff == 1:
                            due_display = "📅 Due Tomorrow"
                        else:
                            due_display = f"📅 Due in {days_diff} days ({task['deadline']})"
                    except Exception:
                        due_display = f"Due: {task['deadline']}"

                t_c1, t_c2, t_c3 = st.columns([6, 2, 2])
                with t_c1:
                    st.markdown(
                        f"""
                        <div style="padding: 6px 0;">
                            <span class="{p_class}">{task['priority']}</span> &nbsp;
                            <strong style="color: #f0f6fc; font-size: 15px;">{task['title']}</strong>
                            <div style="font-size: 12px; color: {'#f85149' if is_overdue else '#8b949e'}; margin-top: 4px;">
                                {due_display} &nbsp;•&nbsp; Course: <em>{task.get('course_id') or 'General'}</em>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with t_c2:
                    if st.button("Mark Complete ✅", key=f"comp_{tid}", type="secondary"):
                        toggle_task_completion(tid, st.session_state.session_id)
                        st.rerun()
                with t_c3:
                    if st.button("Delete 🗑️", key=f"del_{tid}"):
                        delete_task(tid, st.session_state.session_id)
                        st.rerun()
                st.markdown("<hr style='border-color: #21262d; margin: 8px 0;'/>", unsafe_allow_html=True)

    with tab_completed:
        if not completed_tasks:
            st.info("No completed tasks yet. Finish a task in the pending tab to see it here!")
        else:
            for task in completed_tasks:
                tid = task["id"]
                comp_at = task["completed_at"] or "Recently"
                t_c1, t_c2, t_c3 = st.columns([6, 2, 2])
                with t_c1:
                    st.markdown(
                        f"""
                        <div style="padding: 6px 0;">
                            <span style="text-decoration: line-through; color: #8b949e; font-size: 15px;">{task['title']}</span>
                            <div style="font-size: 12px; color: #3fb950; margin-top: 4px;">
                                ✅ Completed: {comp_at}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with t_c2:
                    if st.button("Reopen ↩️", key=f"reopen_{tid}"):
                        toggle_task_completion(tid, st.session_state.session_id)
                        st.rerun()
                with t_c3:
                    if st.button("Delete 🗑️", key=f"del_c_{tid}"):
                        delete_task(tid, st.session_state.session_id)
                        st.rerun()
                st.markdown("<hr style='border-color: #21262d; margin: 8px 0;'/>", unsafe_allow_html=True)


# =============================================================
# FEATURE 3: COURSE PROGRESS TRACKER
# =============================================================
elif nav_choice == "📚 Course Progress Tracker":
    st.markdown("## 📚 Course & Curriculum Progress Tracker")
    st.markdown(
        "Track your progress through syllabus modules, check off completed chapters, "
        "and maintain steady coverage of all Digital Logic concepts."
    )

    c_summary = get_course_progress_summary(st.session_state.session_id)

    # Top summary metrics
    cp_col1, cp_col2, cp_col3, cp_col4 = st.columns(4)
    with cp_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Modules</div>
                <div class="metric-value">{c_summary['total_courses']}</div>
                <div class="metric-sub">Curriculum Units</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cp_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Finished Modules</div>
                <div class="metric-value" style="color: #3fb950;">{c_summary['completed_courses']} / {c_summary['total_courses']}</div>
                <div class="metric-sub">All Chapters Done</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cp_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Chapters Completed</div>
                <div class="metric-value" style="color: #58a6ff;">{c_summary['completed_chapters']} / {c_summary['total_chapters']}</div>
                <div class="metric-sub">{c_summary['total_chapters'] - c_summary['completed_chapters']} Remaining</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cp_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Completion</div>
                <div class="metric-value" style="color: #bc8cff;">{c_summary['overall_progress']}%</div>
                <div class="progress-track" style="margin-top: 8px;">
                    <div class="progress-bar-fill" style="width: {c_summary['overall_progress']}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📖 Individual Course Modules & Chapters")

    courses = c_summary["courses"]
    for course in courses:
        cid = course["course_id"]
        cname = course["course_name"]
        pct = course["progress_percent"]
        chapters = course["chapters"]

        with st.expander(f"📘 **{cname}** — {pct}% Completed ({course['completed_count']}/{course['total_count']} chapters)", expanded=True):
            st.markdown(
                f"""
                <div class="progress-track" style="margin-bottom: 16px;">
                    <div class="progress-bar-fill" style="width: {pct}%;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("##### 📝 Chapters Checklist:")
            for ch in chapters:
                ch_id = ch["chapter_id"]
                is_done = ch["is_completed"]
                ch_title = ch["title"]

                ch_col1, ch_col2 = st.columns([8, 2])
                with ch_col1:
                    checked = st.checkbox(
                        f"**{ch_title}**",
                        value=is_done,
                        key=f"ch_box_{cid}_{ch_id}"
                    )
                    if checked != is_done:
                        toggle_chapter_progress(st.session_state.session_id, cid, ch_id)
                        st.rerun()

                with ch_col2:
                    if is_done:
                        st.markdown("<span class='badge-mastered'>Completed</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-unassessed'>Incomplete</span>", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button(f"🎯 Practice {cname} in Adaptive Quiz", key=f"quiz_link_{cid}"):
                st.session_state.current_question = None
                st.session_state.last_attempt_result = None
                # Set next question matching this concept
                for q_candidate in quiz_service.get_all_questions():
                    if q_candidate["concept_id"] == cid:
                        st.session_state.current_question = q_candidate
                        st.session_state.adaptive_rationale = f"Focused review selected for {cname}."
                        break
                st.session_state.question_start_time = time.time()
                st.info(f"Steered to {cname}! Navigate to '🎯 Adaptive Practice & Quiz' to start.")


# =============================================================
# FEATURE 4: WEEKLY STUDY ANALYTICS
# =============================================================
elif nav_choice == "📈 Weekly Study Analytics":
    st.markdown("## 📈 Weekly Study Analytics")
    st.markdown(
        "Interactive visualizations of your study habits, daily focus hours, "
        "and task velocity powered by **Plotly** and actual SQLite application data."
    )

    analytics = get_weekly_analytics(st.session_state.session_id)
    daily_data = analytics["daily_data"]

    # Top KPI Metrics Row
    an_c1, an_c2, an_c3, an_c4 = st.columns(4)
    with an_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Weekly Study Time</div>
                <div class="metric-value">{analytics['total_weekly_hours']}h</div>
                <div class="metric-sub">Past 7 Days Combined</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with an_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Daily Average Time</div>
                <div class="metric-value" style="color: #3fb950;">{analytics['daily_average_hours']}h</div>
                <div class="metric-sub">Target: 2.0 hrs / day</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with an_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Most Productive Day</div>
                <div class="metric-value" style="color: #bc8cff; font-size: 24px;">{analytics['most_productive_day']}</div>
                <div class="metric-sub">Peak Focus Time</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with an_c4:
        tot_tasks = sum(d["tasks_completed"] for d in daily_data)
        tot_qs = sum(d["questions_attempted"] for d in daily_data)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Activities Finished</div>
                <div class="metric-value" style="color: #58a6ff;">{tot_tasks + tot_qs}</div>
                <div class="metric-sub">{tot_tasks} Tasks + {tot_qs} Quizzes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Plotly Visualizations Grid
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        fig_hours = create_daily_study_hours_chart(daily_data)
        st.plotly_chart(fig_hours, use_container_width=True)

    with ch_col2:
        fig_prod = create_weekly_productivity_chart(daily_data)
        st.plotly_chart(fig_prod, use_container_width=True)

    # Secondary Row: Tasks completed per day
    fig_tasks = create_tasks_completed_chart(daily_data)
    st.plotly_chart(fig_tasks, use_container_width=True)

    # Detailed Daily Breakdown Table
    with st.expander("📊 **View Raw Daily Log Table**"):
        df_daily = pd.DataFrame(daily_data)[["date", "day_name", "pomodoro_hours", "quiz_hours", "total_hours", "tasks_completed", "questions_attempted"]]
        df_daily.columns = ["Date", "Day", "Pomodoro Hours", "Quiz Hours", "Total Hours", "Tasks Completed", "Questions Attempted"]
        st.dataframe(df_daily, use_container_width=True)


# =============================================================
# FEATURE 5: POMODORO FOCUS TIMER
# =============================================================
elif nav_choice == "⏱️ Pomodoro Focus Timer":
    st.markdown("## ⏱️ Pomodoro Study Timer")
    st.markdown(
        "Boost retention and prevent cognitive burnout with timed **25-minute deep focus sessions** "
        "alternated with **5-minute deliberate breaks**."
    )

    pomo_stats = get_pomodoro_stats(st.session_state.session_id)

    # Metrics banner
    pm_c1, pm_c2, pm_c3 = st.columns(3)
    with pm_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Completed Focus Sessions</div>
                <div class="metric-value">{pomo_stats['total_sessions']}</div>
                <div class="metric-sub">25-Minute Blocks</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with pm_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Dedicated Focus Time</div>
                <div class="metric-value" style="color: #3fb950;">{pomo_stats['total_hours']}h</div>
                <div class="metric-sub">{pomo_stats['total_minutes']} Total Minutes</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with pm_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Active Focus State</div>
                <div class="metric-value" style="color: {'#3fb950' if st.session_state.pomo_is_running else '#8b949e'}; font-size: 24px;">
                    {'🟢 Running' if st.session_state.pomo_is_running else '⏸️ Paused'}
                </div>
                <div class="metric-sub">Mode: {st.session_state.pomo_mode.title()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Timer Card
    mode_label = "📚 Deep Study Focus (25m)" if st.session_state.pomo_mode == "study" else "☕ Rest & Recharge Break (5m)"
    mins = st.session_state.pomo_seconds_left // 60
    secs = st.session_state.pomo_seconds_left % 60
    time_str = f"{mins:02d}:{secs:02d}"

    st.markdown(
        f"""
        <div class="pomodoro-card">
            <span class="pomodoro-session-badge">{mode_label}</span>
            <div class="pomodoro-time">{time_str}</div>
            <div style="color: #8b949e; font-size: 14px; margin-bottom: 20px;">
                {'Focus on a single textbook concept or logic circuit without multitasking.' if st.session_state.pomo_mode == 'study' else 'Step away from your screen, stretch, or hydrate.'}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        if not st.session_state.pomo_is_running:
            if st.button("▶️ Start Focus", type="primary", use_container_width=True):
                st.session_state.pomo_is_running = True
                st.rerun()
        else:
            if st.button("⏸️ Pause Timer", type="secondary", use_container_width=True):
                st.session_state.pomo_is_running = False
                st.rerun()

    with ctrl_col2:
        if st.button("🔄 Reset Timer", use_container_width=True):
            st.session_state.pomo_is_running = False
            st.session_state.pomo_seconds_left = 25 * 60 if st.session_state.pomo_mode == "study" else 5 * 60
            st.rerun()

    with ctrl_col3:
        target_mode = "break" if st.session_state.pomo_mode == "study" else "study"
        btn_label = "☕ Switch to 5m Break" if target_mode == "break" else "📚 Switch to 25m Study"
        if st.button(btn_label, use_container_width=True):
            st.session_state.pomo_mode = target_mode
            st.session_state.pomo_is_running = False
            st.session_state.pomo_seconds_left = 5 * 60 if target_mode == "break" else 25 * 60
            st.rerun()

    with ctrl_col4:
        if st.button("✅ Log Completed Session", type="primary", use_container_width=True):
            duration = 25 if st.session_state.pomo_mode == "study" else 5
            log_pomodoro_session(
                session_id=st.session_state.session_id,
                duration_minutes=duration,
                session_type=st.session_state.pomo_mode,
                notes="Manual completion log"
            )
            st.session_state.pomo_is_running = False
            st.session_state.pomo_seconds_left = 25 * 60 if st.session_state.pomo_mode == "study" else 5 * 60
            st.success("🎉 Focus session successfully recorded to SQLite database!")
            st.rerun()

    # Active countdown simulation when timer is running
    if st.session_state.pomo_is_running:
        time.sleep(1)
        if st.session_state.pomo_seconds_left > 0:
            st.session_state.pomo_seconds_left -= 1
            st.rerun()
        else:
            # Session finished!
            st.session_state.pomo_is_running = False
            duration = 25 if st.session_state.pomo_mode == "study" else 5
            log_pomodoro_session(
                session_id=st.session_state.session_id,
                duration_minutes=duration,
                session_type=st.session_state.pomo_mode,
                notes="Auto-completed timer session"
            )
            # Switch modes automatically
            if st.session_state.pomo_mode == "study":
                st.session_state.pomo_mode = "break"
                st.session_state.pomo_seconds_left = 5 * 60
                st.balloons()
                st.success("🎉 Outstanding focus! 25 minutes completed. Enjoy a 5-minute break.")
            else:
                st.session_state.pomo_mode = "study"
                st.session_state.pomo_seconds_left = 25 * 60
                st.success("Break over! Ready for your next study sprint?")
            st.rerun()


# =============================================================
# FEATURE 6: STUDY STREAKS & ACHIEVEMENT BADGES
# =============================================================
elif nav_choice == "🏆 Study Streaks & Badges":
    st.markdown("## 🏆 Study Streaks & Gamification")
    st.markdown(
        "Celebrate consistency, build unstoppable momentum, and unlock achievements "
        "as you progress through your syllabus."
    )

    streak_data = get_study_streak(st.session_state.session_id)
    curr_streak = streak_data["current_streak"]
    long_streak = streak_data["longest_streak"]
    active_dates = streak_data["active_dates"]

    # Hero Streak Display
    st.markdown(
        f"""
        <div class="streak-container">
            <div style="font-size: 56px; margin-bottom: 8px;">🔥</div>
            <div class="streak-number">{curr_streak} <span style="font-size: 24px; font-weight: 500; color: #c9d1d9;">Days Active</span></div>
            <p style="color: #f0f6fc; font-size: 16px; margin-top: 8px;">
                {'🌟 You are on fire! Keep studying today to maintain your momentum.' if curr_streak > 0 else 'Start studying today with a quiz or task to initiate your streak!'}
            </p>
            <div style="font-size: 13px; color: #8b949e;">
                Longest Career Streak: <strong style="color: #58a6ff;">{long_streak} Days</strong> &nbsp;•&nbsp; 
                Total Active Days Logged: <strong style="color: #3fb950;">{len(active_dates)} Days</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 🏅 Achievement Badges")

    badges = get_user_badges(st.session_state.session_id)
    
    # 3x3 Grid of modern badge cards
    for row_idx in range(0, len(badges), 3):
        row_badges = badges[row_idx:row_idx+3]
        b_cols = st.columns(3)
        for col_idx, b in enumerate(row_badges):
            with b_cols[col_idx]:
                is_unlocked = b["is_unlocked"]
                card_class = "badge-card-unlocked" if is_unlocked else "badge-card-locked"
                status_html = f"<div class='badge-date'>✅ Unlocked: {b['unlocked_at'][:10]}</div>" if is_unlocked else "<div style='font-size: 11px; color: #8b949e; margin-top: 8px;'>🔒 Locked</div>"

                st.markdown(
                    f"""
                    <div class="badge-card {card_class}">
                        <div class="badge-icon">{b['icon']}</div>
                        <div class="badge-title">{b['title']}</div>
                        <div style="font-size: 11px; color: #58a6ff; text-transform: uppercase; margin-bottom: 6px;">{b['category']}</div>
                        <div class="badge-desc">{b['desc']}</div>
                        {status_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)


# =============================================================
# VIEW 7: ADAPTIVE PRACTICE & QUIZ (PRESERVED)
# =============================================================
elif nav_choice == "🎯 Adaptive Practice & Quiz":
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


# =============================================================
# VIEW 8: CONCEPT-GAP MASTERY DASHBOARD (PRESERVED)
# =============================================================
elif nav_choice == "📉 Concept-Gap Mastery":
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


# =============================================================
# VIEW 9: TRANSFORMER SEMANTIC CLUSTERING (ML) (PRESERVED)
# =============================================================
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


# =============================================================
# VIEW 10: SCIENTIFIC EVALUATION & EXPERIMENT (PRESERVED)
# =============================================================
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


# =============================================================
# VIEW 11: SESSION DATA & RESET (PRESERVED)
# =============================================================
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
    tab_att, tab_tasks_log, tab_pomo_log, tab_mast, tab_exp = st.tabs([
        "Quiz Attempts",
        "Study Tasks",
        "Pomodoro Focus",
        "Concept Mastery",
        "Explanation Logs"
    ])

    with tab_att:
        df_att = get_attempts(session_id=st.session_state.session_id)
        if not df_att.empty:
            st.dataframe(df_att, use_container_width=True)
        else:
            st.info("No quiz attempts in current session.")

    with tab_tasks_log:
        tasks_list = get_tasks(session_id=st.session_state.session_id)
        if tasks_list:
            st.dataframe(pd.DataFrame(tasks_list), use_container_width=True)
        else:
            st.info("No study tasks logged yet.")

    with tab_pomo_log:
        pomo_analytics = get_weekly_analytics(st.session_state.session_id)
        st.dataframe(pd.DataFrame(pomo_analytics["daily_data"]), use_container_width=True)

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
    st.markdown("Permanently clears all session records, tasks, focus logs, and restores initial state.")
    confirm_reset = st.checkbox("I understand that this will erase all quiz attempts, tasks, pomodoro sessions, and mastery records.")
    if st.button("Reset Entire Database", type="secondary", disabled=not confirm_reset):
        success = reset_database(confirmation=True)
        if success:
            st.session_state.current_question = None
            st.session_state.last_attempt_result = None
            st.session_state.pomo_is_running = False
            st.session_state.pomo_seconds_left = 25 * 60
            st.success("Database reset complete. All records cleared.")
            st.rerun()
