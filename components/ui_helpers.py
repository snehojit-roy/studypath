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
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #58a6ff;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 13px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }

        /* Status badges */
        .badge-mastered {
            background-color: #238636;
            color: #ffffff;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-progress {
            background-color: #1f6feb;
            color: #ffffff;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-needs-practice {
            background-color: #da3633;
            color: #ffffff;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-unassessed {
            background-color: #484f58;
            color: #ffffff;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }

        /* Callout boxes */
        .callout-box {
            background: #161b22;
            border-left: 4px solid #58a6ff;
            padding: 14px 18px;
            border-radius: 4px;
            margin-bottom: 16px;
        }
        .callout-warning {
            background: #211a1d;
            border-left: 4px solid #d29922;
            padding: 14px 18px;
            border-radius: 4px;
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
