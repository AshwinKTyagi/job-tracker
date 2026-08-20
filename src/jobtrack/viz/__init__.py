"""viz module — see CONTRACTS.md §8.

`charts.py` builds individual plotly figures from a DataFrame shaped like M4's
`build_dataframe` / `build_events_dataframe` output; `dashboard.py` composes them into
one self-contained HTML file. Nothing here touches SQLite — viz only ever sees a
DataFrame (I10).
"""

from __future__ import annotations

from jobtrack.viz.charts import (
    PIPELINE_STAGE_LABELS,
    SANKEY_NODE_ORDER,
    STATUS_COLORS,
    STATUS_PIPELINE_ORDER,
    applications_over_time,
    compute_stage_flows,
    funnel_sankey,
    response_time_histogram,
    status_bar_chart,
    top_companies_bar,
)
from jobtrack.viz.dashboard import build_dashboard

__all__ = [
    "PIPELINE_STAGE_LABELS",
    "SANKEY_NODE_ORDER",
    "STATUS_COLORS",
    "STATUS_PIPELINE_ORDER",
    "applications_over_time",
    "build_dashboard",
    "compute_stage_flows",
    "funnel_sankey",
    "response_time_histogram",
    "status_bar_chart",
    "top_companies_bar",
]
