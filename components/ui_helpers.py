"""
UI Helper functions and Plotly chart builders for StudyPath.
Dark-themed, elegant, readable, and responsive.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


def render_custom_css():
    """Injects restrained dark-theme styling, typography, and card accents."""
    st.markdown(
        """
        <style>
        /* Base typography & dark theme styling */
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Metric cards */
        .metric-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: #58a6ff;
            transform: translateY(-2px);
        }
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            color: #58a6ff;
            line-height: 1.1;
            margin-top: 4px;
        }
        .metric-label {
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }
        .metric-sub {
            font-size: 13px;
            color: #8b949e;
            margin-top: 6px;
        }

        /* Status & Concept Badges */
        .badge-mastered {
            background-color: #238636;
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .badge-progress {
            background-color: #1f6feb;
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .badge-needs-practice {
            background-color: #da3633;
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .badge-unassessed {
            background-color: #30363d;
            color: #c9d1d9;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        /* Priority Badges */
        .priority-high {
            background-color: rgba(248, 81, 73, 0.15);
            color: #f85149;
            border: 1px solid rgba(248, 81, 73, 0.4);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .priority-medium {
            background-color: rgba(210, 153, 34, 0.15);
            color: #d29922;
            border: 1px solid rgba(210, 153, 34, 0.4);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .priority-low {
            background-color: rgba(88, 166, 255, 0.15);
            color: #58a6ff;
            border: 1px solid rgba(88, 166, 255, 0.4);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }

        /* Callout boxes */
        .callout-box {
            background: #161b22;
            border-left: 4px solid #58a6ff;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 16px;
        }
        .callout-warning {
            background: #211a1d;
            border-left: 4px solid #d29922;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 16px;
        }
        .callout-remedy {
            background: #1c2128;
            border-left: 4px solid #bc8cff;
            padding: 16px 20px;
            border-radius: 6px;
            margin-top: 14px;
            margin-bottom: 14px;
        }
        .callout-success {
            background: #122619;
            border-left: 4px solid #3fb950;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 16px;
        }

        /* Course Progress Cards */
        .course-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 14px;
        }
        .progress-track {
            background-color: #21262d;
            border-radius: 8px;
            height: 10px;
            width: 100%;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #1f6feb 0%, #3fb950 100%);
            height: 100%;
            border-radius: 8px;
            transition: width 0.4s ease;
        }

        /* Task Cards */
        .task-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .task-card-completed {
            background: #0d1117;
            border: 1px solid #21262d;
            opacity: 0.75;
        }

        /* Streak & Gamification */
        .streak-container {
            background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
            border: 1px solid #ff7b72;
            border-radius: 14px;
            padding: 22px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(255, 123, 114, 0.1);
        }
        .streak-number {
            font-size: 48px;
            font-weight: 800;
            color: #ff7b72;
            line-height: 1;
        }

        /* Badge Cards */
        .badge-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            height: 100%;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .badge-card:hover {
            transform: translateY(-2px);
        }
        .badge-card-unlocked {
            border-color: #d29922;
            background: linear-gradient(145deg, #161b22 0%, #1f1b16 100%);
        }
        .badge-card-locked {
            opacity: 0.55;
            filter: grayscale(0.8);
        }
        .badge-icon {
            font-size: 40px;
            margin-bottom: 8px;
        }
        .badge-title {
            font-weight: 700;
            font-size: 15px;
            color: #f0f6fc;
            margin-bottom: 4px;
        }
        .badge-desc {
            font-size: 12px;
            color: #8b949e;
            line-height: 1.4;
        }
        .badge-date {
            font-size: 11px;
            color: #3fb950;
            margin-top: 8px;
            font-weight: 600;
        }

        /* Pomodoro Timer Display */
        .pomodoro-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            max-width: 520px;
            margin: 0 auto 20px auto;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }
        .pomodoro-time {
            font-size: 64px;
            font-family: 'SF Mono', Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
            font-weight: 700;
            color: #58a6ff;
            letter-spacing: 2px;
            margin: 15px 0;
        }
        .pomodoro-session-badge {
            display: inline-block;
            background: #21262d;
            color: #79c0ff;
            border: 1px solid #388bfd;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def create_mastery_bar_chart(stats_list: List[Dict[str, Any]]) -> go.Figure:
    """Creates a grouped comparison of Observed Accuracy vs Laplace Smoothed Mastery."""
    if not stats_list:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            title="No Concept Attempts Yet"
        )
        return fig

    names = [s.get("concept_name", s.get("concept_id")) for s in stats_list]
    observed = [s.get("observed_accuracy", 0.0) * 100 for s in stats_list]
    smoothed = [s.get("smoothed_mastery", 0.5) * 100 for s in stats_list]
    attempts = [s.get("attempts_count", 0) for s in stats_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=observed,
        name="Observed Accuracy (%)",
        marker_color="#238636",
        text=[f"{v:.1f}% (N={n})" for v, n in zip(observed, attempts)],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=names,
        y=smoothed,
        name="Smoothed Mastery (%)",
        marker_color="#1f6feb",
        text=[f"{v:.1f}%" for v in smoothed],
        textposition="auto"
    ))

    # Add 80% mastery target line
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="#e3b341",
        annotation_text="Provisional Mastery Target (80%)",
        annotation_position="bottom right"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        barmode="group",
        title="Observed Accuracy vs. Bayesian Smoothed Mastery",
        yaxis=dict(title="Percentage (%)", range=[0, 105]),
        xaxis=dict(title="Concept"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_pca_scatter_plot(
    records: List[Dict[str, Any]],
    pca_variance_ratio: List[float]
) -> go.Figure:
    """
    Renders 2D PCA projection scatter plot with detailed hovercards.
    Clearly annotated as a 2D linear projection of high-dimensional embedding space.
    """
    if not records:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            title="No mistake embeddings available"
        )
        return fig

    df = pd.DataFrame(records)
    # Ensure columns exist
    df["cluster_name"] = df["cluster_label"].apply(lambda l: f"Cluster {l}" if l != -1 else "Unclustered / Outlier")
    total_var = sum(pca_variance_ratio) * 100

    # Custom color palette
    color_map = {
        "Unclustered / Outlier": "#8b949e"
    }

    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="cluster_name",
        hover_data={
            "pca_x": False,
            "pca_y": False,
            "cluster_name": True,
            "concept_id": True,
            "student_answer": True,
            "correct_answer": True,
            "misconception_tag": True
        },
        title=f"2D PCA Projection of Mistake Embeddings (Explains {total_var:.1f}% of 384-D Variance)",
        template="plotly_dark"
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="#30363d")))

    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        xaxis=dict(title=f"PC 1 ({pca_variance_ratio[0]*100:.1f}% var)", showgrid=True, gridcolor="#21262d"),
        yaxis=dict(title=f"PC 2 ({pca_variance_ratio[1]*100:.1f}% var)", showgrid=True, gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Footnote annotation
    fig.add_annotation(
        text="Note: 2D linear projection for visualization only. Clustering decisions operate in full 384D cosine space.",
        xref="paper", yref="paper",
        x=0.0, y=-0.18,
        showarrow=False,
        font=dict(size=11, color="#8b949e")
    )

    return fig


def create_similarity_heatmap(
    similarity_matrix: List[List[float]],
    labels: List[str]
) -> go.Figure:
    """Generates pairwise cosine similarity heatmap for clustered mistakes."""
    if not similarity_matrix:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            title="Similarity matrix empty"
        )
        return fig

    # Truncate labels for readability
    short_labels = [lbl[:24] + "..." if len(lbl) > 24 else lbl for lbl in labels]

    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=short_labels,
        y=short_labels,
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Cosine Sim")
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        title="Pairwise Mistake Cosine Similarity Matrix",
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(tickangle=-45)
    )
    return fig


def create_performance_timeline(attempts_df: pd.DataFrame) -> go.Figure:
    """Plots cumulative accuracy trajectory over question attempts."""
    if attempts_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            title="No attempts logged"
        )
        return fig

    df = attempts_df.copy().reset_index(drop=True)
    df["attempt_num"] = df.index + 1
    df["cumulative_correct"] = df["is_correct"].cumsum()
    df["cumulative_accuracy"] = (df["cumulative_correct"] / df["attempt_num"]) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["attempt_num"],
        y=df["cumulative_accuracy"],
        mode="lines+markers",
        name="Cumulative Accuracy (%)",
        line=dict(color="#58a6ff", width=2),
        marker=dict(
            size=8,
            color=["#238636" if c == 1 else "#da3633" for c in df["is_correct"]]
        )
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        title="Performance Trajectory Over Time",
        xaxis=dict(title="Question Attempt Number", dtick=1),
        yaxis=dict(title="Cumulative Accuracy (%)", range=[0, 105]),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


# ========================================================
# NEW PLOTLY CHARTS FOR STUDYPATH SAAS DASHBOARD
# ========================================================

def create_daily_study_hours_chart(daily_data: List[Dict[str, Any]]) -> go.Figure:
    """Renders a stacked bar chart of daily study hours (Pomodoro Focus vs Quiz Practice)."""
    if not daily_data:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            title="No Daily Study Time Recorded"
        )
        return fig

    df = pd.DataFrame(daily_data)
    fig = go.Figure()

    # Pomodoro Focus Blocks
    fig.add_trace(go.Bar(
        x=df["day_name"],
        y=df["pomodoro_hours"],
        name="Pomodoro Focus",
        marker=dict(color="#1f6feb", line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Pomodoro Focus: %{y:.2f} hrs<extra></extra>"
    ))

    # Quiz & Practice Time
    fig.add_trace(go.Bar(
        x=df["day_name"],
        y=df["quiz_hours"],
        name="Quiz & Practice",
        marker=dict(color="#bc8cff", line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Quiz Practice: %{y:.2f} hrs<extra></extra>"
    ))

    # Target Benchmark Line (e.g. 2 hrs/day)
    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="#3fb950",
        annotation_text="Daily Goal (2h)",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#3fb950")
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        barmode="stack",
        title=dict(text="<b>Daily Study Hours (Past 7 Days)</b>", font=dict(size=16, color="#f0f6fc")),
        xaxis=dict(title="Day of Week", showgrid=False),
        yaxis=dict(title="Hours", showgrid=True, gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320
    )
    return fig


def create_weekly_productivity_chart(daily_data: List[Dict[str, Any]]) -> go.Figure:
    """Renders dual-axis weekly productivity chart (Study Hours vs Activity Output)."""
    if not daily_data:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22")
        return fig

    df = pd.DataFrame(daily_data)
    df["tasks_and_questions"] = df["tasks_completed"] + df["questions_attempted"]

    fig = go.Figure()

    # Primary axis: Total Study Hours (Bar)
    fig.add_trace(go.Bar(
        x=df["day_name"],
        y=df["total_hours"],
        name="Study Hours",
        marker=dict(color="#238636"),
        hovertemplate="<b>%{x}</b><br>Total Time: %{y:.1f} hrs<extra></extra>",
        yaxis="y1"
    ))

    # Secondary axis: Completed Items (Line)
    fig.add_trace(go.Scatter(
        x=df["day_name"],
        y=df["tasks_and_questions"],
        name="Tasks & Questions",
        mode="lines+markers",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=8, color="#79c0ff"),
        hovertemplate="<b>%{x}</b><br>Activity Count: %{y}<extra></extra>",
        yaxis="y2"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        title=dict(text="<b>Weekly Productivity & Output</b>", font=dict(size=16, color="#f0f6fc")),
        xaxis=dict(title="Day", showgrid=False),
        yaxis=dict(
            title="Study Hours",
            side="left",
            showgrid=True,
            gridcolor="#21262d"
        ),
        yaxis2=dict(
            title="Activity Count (Tasks + Qs)",
            side="right",
            overlaying="y",
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320
    )
    return fig


def create_tasks_completed_chart(daily_data: List[Dict[str, Any]]) -> go.Figure:
    """Renders bar chart of daily completed study tasks."""
    if not daily_data:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22")
        return fig

    df = pd.DataFrame(daily_data)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["day_name"],
        y=df["tasks_completed"],
        name="Tasks Completed",
        marker=dict(color="#d29922"),
        hovertemplate="<b>%{x}</b><br>Tasks Done: %{y}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        title=dict(text="<b>Tasks Completed Per Day</b>", font=dict(size=16, color="#f0f6fc")),
        xaxis=dict(title="Day", showgrid=False),
        yaxis=dict(title="Completed Tasks", dtick=1, showgrid=True, gridcolor="#21262d"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=280
    )
    return fig


def create_course_progress_donut(completed_chapters: int, total_chapters: int) -> go.Figure:
    """Renders an attractive donut gauge of overall curriculum completion."""
    remaining = max(0, total_chapters - completed_chapters)
    pct = round((completed_chapters / total_chapters * 100), 1) if total_chapters > 0 else 0.0

    fig = go.Figure(data=[go.Pie(
        labels=["Completed", "Remaining"],
        values=[completed_chapters, remaining],
        hole=0.72,
        marker=dict(colors=["#3fb950", "#21262d"]),
        textinfo="none",
        hoverinfo="label+value"
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{pct}%</b><br><span style='font-size:12px;color:#8b949e;'>Curriculum</span>",
                x=0.5, y=0.5,
                font=dict(size=24, color="#f0f6fc"),
                showarrow=False
            )
        ],
        margin=dict(l=10, r=10, t=10, b=10),
        height=200
    )
    return fig
