"""
UI Components and Visualizations for StudyPath.
"""
from .ui_helpers import (
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

__all__ = [
    "render_custom_css",
    "create_mastery_bar_chart",
    "create_pca_scatter_plot",
    "create_similarity_heatmap",
    "create_performance_timeline",
    "create_daily_study_hours_chart",
    "create_weekly_productivity_chart",
    "create_tasks_completed_chart",
    "create_course_progress_donut"
]
