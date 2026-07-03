from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh


DB_PATH = Path("./results/realtime/predictions.sqlite")
MAX_ROWS_TO_LOAD = 50_000

st.set_page_config(
    page_title="Live Predictions",
    page_icon="🦠",
    layout="wide",
)

st.divider()
st.title("🦠 Live Predictions")

refresh_seconds = st.sidebar.slider(
    "Refresh every",
    min_value=1,
    max_value=30,
    value=5,
    key="refresh_seconds",
)

st_autorefresh(interval=refresh_seconds * 1000, key="refresh")


@st.cache_data(ttl=2)
def load_predictions_from_db(
    db_path: Path,
    limit: int = MAX_ROWS_TO_LOAD,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    query = """
        SELECT *
        FROM predictions
        ORDER BY rowid DESC
        LIMIT ?
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(limit,))

    df = df.iloc[::-1].reset_index(drop=True)

    for col in ["processed_at", "event_time", "timestamp", "stored_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


df = load_predictions_from_db(DB_PATH)

if df.empty:
    st.warning(f"No predictions found in SQLite database at: {DB_PATH}")
    st.stop()


latest = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Loaded particles", f"{len(df):,}")
col2.metric("Latest class", latest["predicted_label"])
col3.metric("Confidence", f"{latest['prediction_confidence']:.1%}")
col4.metric("Mean size", f"{df['size'].mean():.2f} µm")


st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Predicted Class Distribution")

    class_counts = df["predicted_label"].value_counts().reset_index()
    class_counts.columns = ["predicted_label", "count"]

    fig = px.bar(
        class_counts,
        x="predicted_label",
        y="count",
        title="Particle predictions by class",
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Prediction Confidence")

    fig = px.histogram(
        df,
        x="prediction_confidence",
        nbins=30,
        title="Confidence distribution",
    )

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Predictions Over Time")

timeline_window_minutes = st.slider(
    "Timeline window in minutes",
    min_value=1,
    max_value=30,
    value=30,
    step=1,
    key="live_predictions_timeline_window",
)

timeline_bin = st.selectbox(
    "Timeline bin",
    ["10s", "30s", "1min", "5min"],
    index=0,
    key="live_predictions_timeline_bin",
)

timeline = df.dropna(subset=["timestamp"]).copy()

if timeline.empty:
    st.info("No valid timestamps available for timeline.")
else:
    latest_time = timeline["timestamp"].max()
    start_time = latest_time - pd.Timedelta(minutes=timeline_window_minutes)

    timeline = timeline[
        (timeline["timestamp"] >= start_time)
        & (timeline["timestamp"] <= latest_time)
    ].copy()

    if timeline.empty:
        st.info("No predictions found in the selected timeline window.")
    else:
        timeline["time_bin"] = timeline["timestamp"].dt.floor(timeline_bin)

        timeline_counts = (
            timeline
            .groupby(["time_bin", "predicted_label"])
            .size()
            .reset_index(name="count")
        )

        fig = px.line(
            timeline_counts,
            x="time_bin",
            y="count",
            color="predicted_label",
            markers=True,
            title=f"Predictions per {timeline_bin}",
        )

        fig.update_xaxes(
            title="Time",
            range=[start_time, latest_time],
        )
        fig.update_yaxes(title="Particle count")

        st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Current Sample Composition")

max_window = min(len(df), 10000)
min_window = min(100, max_window)

if max_window < 1:
    st.info("Not enough particles for composition estimate.")
else:
    window_size = st.slider(
        "Particles used for composition estimate",
        min_value=min_window,
        max_value=max_window,
        value=min(max_window, 1000),
        step=100 if max_window >= 100 else 1,
        key="composition_window_size",
    )

    recent = df.tail(window_size).copy()

    composition = (
        recent["predicted_label"]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )

    composition.columns = ["predicted_label", "percentage"]

    composition_counts = (
        recent["predicted_label"]
        .value_counts()
        .reset_index()
    )

    composition_counts.columns = ["predicted_label", "count"]

    composition = composition.merge(
        composition_counts,
        on="predicted_label",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        dominant_class = composition.iloc[0]["predicted_label"]
        dominant_percentage = composition.iloc[0]["percentage"]

        st.metric(
            "Dominant predicted species",
            dominant_class,
            f"{dominant_percentage:.1f}%",
        )

        st.dataframe(
            composition.assign(
                percentage=composition["percentage"].round(2)
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        fig = px.bar(
            composition,
            x="predicted_label",
            y="percentage",
            text=composition["percentage"].round(1),
            title=f"Estimated composition from last {window_size:,} particles",
        )

        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_yaxes(title="Percentage", range=[0, 100])
        fig.update_xaxes(title="Predicted species")

        st.plotly_chart(fig, use_container_width=True)


st.subheader("Composition Trend Over Time")

composition_window_minutes = st.slider(
    "Composition trend window in minutes",
    min_value=1,
    max_value=30,
    value=30,
    step=1,
    key="composition_trend_window_minutes",
)

trend_bin = st.selectbox(
    "Time bin",
    ["10s", "30s", "1min", "5min"],
    index=2,
    key="composition_trend_bin",
)

composition_timeline = df.dropna(subset=["timestamp"]).copy()

if composition_timeline.empty:
    st.info("No valid timestamps available for composition trend.")
else:
    latest_time = composition_timeline["timestamp"].max()
    start_time = latest_time - pd.Timedelta(minutes=composition_window_minutes)

    composition_timeline = composition_timeline[
        (composition_timeline["timestamp"] >= start_time)
        & (composition_timeline["timestamp"] <= latest_time)
    ].copy()

    if composition_timeline.empty:
        st.info("No predictions found in the selected composition trend window.")
    else:
        composition_timeline["time_bin"] = composition_timeline[
            "timestamp"
        ].dt.floor(trend_bin)

        composition_timeline = (
            composition_timeline
            .groupby(["time_bin", "predicted_label"])
            .size()
            .reset_index(name="count")
        )

        composition_timeline["total_in_bin"] = (
            composition_timeline
            .groupby("time_bin")["count"]
            .transform("sum")
        )

        composition_timeline["percentage"] = (
            composition_timeline["count"]
            / composition_timeline["total_in_bin"]
            * 100
        )

        fig = px.area(
            composition_timeline,
            x="time_bin",
            y="percentage",
            color="predicted_label",
            title="Estimated bacterial composition over time",
        )

        fig.update_yaxes(title="Percentage", range=[0, 100])
        fig.update_xaxes(
            title="Time",
            range=[start_time, latest_time],
        )

        st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Particle Profile Statistics")

profile_max = min(len(df), MAX_ROWS_TO_LOAD)

if profile_max < 1000:
    profile_df = df.copy()
else:
    profile_sample_size = st.slider(
        "Maximum particles shown in profile plots",
        min_value=1_000,
        max_value=profile_max,
        value=min(profile_max, 10_000),
        step=1_000,
        key="profile_plot_sample_size",
    )

    profile_df = df.tail(profile_sample_size).copy()

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        profile_df,
        x="size",
        nbins=40,
        color="predicted_label",
        title="Particle size distribution",
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        profile_df,
        x="size",
        y="time_asymmetry",
        color="predicted_label",
        hover_data=[
            "raw_file",
            "particle_index",
            "prediction_confidence",
        ],
        title="Size vs time asymmetry",
    )

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Class Probabilities")

prob_cols = [
    "prob_B_cereus",
    "prob_B_endophyticus",
    "prob_K_salsicia",
    "prob_M_luteus",
    "prob_S_huminis",
]

available_prob_cols = [col for col in prob_cols if col in df.columns]

if available_prob_cols:
    prob_means = df[available_prob_cols].mean().reset_index()

    prob_means.columns = ["class", "mean_probability"]
    prob_means["class"] = (
        prob_means["class"]
        .str.replace("prob_", "", regex=False)
        .str.replace("_", " ")
    )

    fig = px.bar(
        prob_means,
        x="class",
        y="mean_probability",
        title="Mean predicted probability by class",
    )

    fig.update_yaxes(title="Mean probability", range=[0, 1])
    fig.update_xaxes(title="Class")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No probability columns found.")


st.divider()

st.subheader("Recent Predictions")

rows = st.slider(
    "Rows to show",
    min_value=10,
    max_value=500,
    value=50,
    key="recent_predictions_rows",
)

display_cols = [
    "timestamp",
    "raw_file",
    "particle_index",
    "size",
    "time_asymmetry",
    "predicted_label",
    "prediction_confidence",
]

available_display_cols = [col for col in display_cols if col in df.columns]

st.dataframe(
    df[available_display_cols].tail(rows).sort_index(ascending=False),
    use_container_width=True,
)

st.caption(
    f"Loaded at most the most recent {MAX_ROWS_TO_LOAD:,} rows from SQLite: "
    f"{DB_PATH}."
)