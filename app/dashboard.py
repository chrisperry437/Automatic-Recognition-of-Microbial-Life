from pathlib import Path
import json
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh


DB_PATH = Path("./results/realtime/predictions.sqlite")
ACTIVE_MODEL_PATH = Path("./configs/active_model.json")
MAX_ROWS_TO_LOAD = 50_000

st.set_page_config(
    page_title="Rapid-E Bioaerosol Dashboard",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 Rapid-E Bioaerosol Dashboard")

refresh_seconds = st.sidebar.slider(
    "Refresh every",
    min_value=1,
    max_value=30,
    value=5,
    key="dashboard_refresh_seconds",
)

st_autorefresh(interval=refresh_seconds * 1000, key="dashboard_refresh")


@st.cache_data(ttl=2)
def load_predictions_from_db(
    db_path: Path,
    minutes: int = 30,
    limit: int = MAX_ROWS_TO_LOAD,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    query = """
        SELECT *
        FROM predictions
        WHERE timestamp >= (
            SELECT datetime(MAX(timestamp), ?)
            FROM predictions
        )
        ORDER BY timestamp DESC
        LIMIT ?
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(f"-{minutes} minutes", limit),
        )

    for col in ["processed_at", "event_time", "timestamp", "stored_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def load_active_model(path: Path) -> str:
    if not path.exists():
        return "No active model selected"

    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        return config.get("active_model", "No active model selected")
    except Exception:
        return "Could not read active model"


# Load the last 30 minutes from SQLite.
df = load_predictions_from_db(
    DB_PATH,
    minutes=30,
    limit=MAX_ROWS_TO_LOAD,
)

active_model = load_active_model(ACTIVE_MODEL_PATH)

if df.empty:
    st.warning(f"No predictions found in database at: {DB_PATH}")
    st.stop()


time_col = "timestamp" if "timestamp" in df.columns else "processed_at"
df = df.dropna(subset=[time_col]).copy()

if df.empty:
    st.warning("Predictions found, but no valid timestamps are available.")
    st.stop()


latest_time = df[time_col].max()
last_minute = df[df[time_col] >= latest_time - pd.Timedelta(minutes=1)]
last_hour = df[df[time_col] >= latest_time - pd.Timedelta(hours=1)]


# -----------------------------
# Top status cards
# -----------------------------

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Loaded particles", f"{len(df):,}")
col2.metric("Particles / minute", f"{len(last_minute):,}")
col3.metric("Particles / hour", f"{len(last_hour):,}")
col4.metric("Active model", active_model)
col5.metric("Current time", datetime.now().strftime("%H:%M:%S"))


# -----------------------------
# Live particle count by class
# -----------------------------

st.divider()

st.subheader("Live Particle Count by Predicted Species")

chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    particle_bin = st.selectbox(
        "Particle count time bin",
        ["10s", "30s", "1min", "5min"],
        index=2,
        key="dashboard_particle_count_bin",
    )

with chart_col2:
    chart_window_minutes = st.slider(
        "Show previous minutes",
        min_value=1,
        max_value=30,
        value=30,
        step=1,
        key="dashboard_chart_window_minutes",
    )

latest_chart_time = df[time_col].max()
chart_start_time = latest_chart_time - pd.Timedelta(minutes=chart_window_minutes)

count_df = df[
    (df[time_col] >= chart_start_time)
    & (df[time_col] <= latest_chart_time)
].copy()

if count_df.empty:
    st.info("No particles found in the selected chart time window.")
else:
    count_df["time_bin"] = count_df[time_col].dt.floor(particle_bin)

    particle_counts = (
        count_df
        .groupby(["time_bin", "predicted_label"])
        .size()
        .reset_index(name="particle_count")
    )

    fig = px.line(
        particle_counts,
        x="time_bin",
        y="particle_count",
        color="predicted_label",
        markers=True,
        title=f"Live particle count per {particle_bin}",
    )

    fig.update_xaxes(
        title="Time",
        range=[chart_start_time, latest_chart_time],
    )

    fig.update_yaxes(title="Particle count")

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Most recent raw files
# -----------------------------

st.divider()

st.subheader("Most Recent Raw Files")

raw_file_summary = (
    df
    .groupby("raw_file")
    .agg(
        particle_count=("raw_file", "size"),
        first_timestamp=(time_col, "min"),
        latest_timestamp=(time_col, "max"),
        mean_confidence=("prediction_confidence", "mean"),
        dominant_species=("predicted_label", lambda x: x.value_counts().idxmax()),
    )
    .reset_index()
    .sort_values("latest_timestamp", ascending=False)
    .head(10)
)

raw_file_summary["mean_confidence"] = (
    raw_file_summary["mean_confidence"]
    .mul(100)
    .round(1)
    .astype(str)
    + "%"
)

st.dataframe(
    raw_file_summary[
        [
            "raw_file",
            "latest_timestamp",
            "particle_count",
            "dominant_species",
            "mean_confidence",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Dashboard updates automatically every {refresh_seconds} seconds. "
    f"Loaded at most the most recent {MAX_ROWS_TO_LOAD:,} prediction rows "
    f"from SQLite: {DB_PATH}."
)