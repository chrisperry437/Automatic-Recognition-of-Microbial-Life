from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


DB_PATH = Path("./results/realtime/predictions.sqlite")
MAX_ROWS_TO_LOAD = 50_000

st.set_page_config(
    page_title="Raw File Explorer",
    page_icon="📁",
    layout="wide",
)

st.divider()
st.title("📁 Raw File Explorer")


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


required_cols = [
    "raw_file",
    "predicted_label",
    "prediction_confidence",
    "size",
    "time_asymmetry",
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()


st.subheader("Filters")

time_col = "timestamp" if "timestamp" in df.columns else "processed_at"
df = df.dropna(subset=[time_col]).copy()

if df.empty:
    st.warning("Predictions found, but no valid timestamps are available.")
    st.stop()

min_datetime = df[time_col].min()
max_datetime = df[time_col].max()

filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

with filter_col1:
    start_date = st.date_input(
        "Start date",
        value=min_datetime.date(),
        min_value=min_datetime.date(),
        max_value=max_datetime.date(),
        key="raw_file_start_date",
    )

    start_time = st.time_input(
        "Start time",
        value=min_datetime.time(),
        key="raw_file_start_time",
    )

with filter_col2:
    end_date = st.date_input(
        "End date",
        value=max_datetime.date(),
        min_value=min_datetime.date(),
        max_value=max_datetime.date(),
        key="raw_file_end_date",
    )

    end_time = st.time_input(
        "End time",
        value=max_datetime.time(),
        key="raw_file_end_time",
    )

start_datetime = pd.Timestamp.combine(start_date, start_time)
end_datetime = pd.Timestamp.combine(end_date, end_time)

if start_datetime > end_datetime:
    st.warning("Start datetime must be before end datetime.")
    st.stop()

time_filtered_df = df[
    (df[time_col] >= start_datetime)
    & (df[time_col] <= end_datetime)
].copy()

if time_filtered_df.empty:
    st.warning("No raw files found in the selected datetime range.")
    st.stop()


raw_summary = (
    time_filtered_df
    .groupby("raw_file")
    .agg(
        particle_count=("raw_file", "size"),
        mean_confidence=("prediction_confidence", "mean"),
        first_timestamp=(time_col, "min"),
        last_timestamp=(time_col, "max"),
        dominant_species=(
            "predicted_label",
            lambda x: x.value_counts().idxmax(),
        ),
    )
    .reset_index()
    .sort_values("last_timestamp", ascending=False)
)

with filter_col3:
    dominant_species_options = sorted(
        raw_summary["dominant_species"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_dominant_species = st.multiselect(
        "Dominant species in raw file",
        dominant_species_options,
        default=dominant_species_options,
        key="raw_file_dominant_species_filter",
    )

    min_file_confidence = st.slider(
        "Minimum file mean confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="raw_file_mean_confidence_filter",
    )

raw_summary = raw_summary[
    raw_summary["dominant_species"].isin(selected_dominant_species)
    & (raw_summary["mean_confidence"] >= min_file_confidence)
].copy()

if raw_summary.empty:
    st.warning("No raw files match the selected file-level filters.")
    st.stop()

selected_raw_file = st.selectbox(
    "Select raw file",
    raw_summary["raw_file"].tolist(),
    key="raw_file_explorer_selected_file",
)

selected_df = time_filtered_df[
    time_filtered_df["raw_file"] == selected_raw_file
].copy()


st.subheader("Particle-Level Filters Within Selected Raw File")

particle_filter_col1, particle_filter_col2 = st.columns([2, 1])

with particle_filter_col1:
    species_options = sorted(
        selected_df["predicted_label"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_particle_species = st.multiselect(
        "Predicted species within selected file",
        species_options,
        default=species_options,
        key="raw_file_particle_species_filter",
    )

with particle_filter_col2:
    min_particle_confidence = st.slider(
        "Minimum particle confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="raw_file_particle_confidence_filter",
    )

selected_df = selected_df[
    selected_df["predicted_label"].isin(selected_particle_species)
    & (selected_df["prediction_confidence"] >= min_particle_confidence)
].copy()

if selected_df.empty:
    st.warning("No particles in this raw file match the selected particle-level filters.")
    st.stop()

st.divider()


st.subheader("Raw File Overview")
st.code(selected_raw_file)

particle_count = len(selected_df)
mean_confidence = selected_df["prediction_confidence"].mean()
median_confidence = selected_df["prediction_confidence"].median()
mean_size = selected_df["size"].mean()

valid_time = selected_df.dropna(subset=[time_col])
start_time_value = valid_time[time_col].min() if not valid_time.empty else None
end_time_value = valid_time[time_col].max() if not valid_time.empty else None

col1, col2, col3, col4 = st.columns(4)

col1.metric("Particle count", f"{particle_count:,}")
col2.metric("Mean confidence", f"{mean_confidence:.1%}")
col3.metric("Median confidence", f"{median_confidence:.1%}")
col4.metric("Mean size", f"{mean_size:.2f} µm")

if start_time_value is not None and end_time_value is not None:
    col1, col2 = st.columns(2)
    col1.info(f"Start time: {start_time_value}")
    col2.info(f"End time: {end_time_value}")


st.divider()

st.subheader("Raw File Composition")

composition = (
    selected_df["predicted_label"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)

composition.columns = ["predicted_label", "percentage"]

composition_counts = (
    selected_df["predicted_label"]
    .value_counts()
    .reset_index()
)

composition_counts.columns = ["predicted_label", "count"]

composition = composition.merge(composition_counts, on="predicted_label")

col1, col2 = st.columns([1, 2])

with col1:
    dominant = composition.iloc[0]

    st.metric(
        "Dominant species",
        dominant["predicted_label"],
        f"{dominant['percentage']:.1f}%",
    )

    st.dataframe(
        composition.assign(percentage=composition["percentage"].round(2)),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    fig = px.bar(
        composition,
        x="predicted_label",
        y="percentage",
        text=composition["percentage"].round(1),
        title="Predicted composition for selected raw file",
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_yaxes(title="Percentage", range=[0, 100])
    fig.update_xaxes(title="Predicted species")

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Confidence Profile")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        selected_df,
        x="prediction_confidence",
        nbins=30,
        color="predicted_label",
        title="Confidence distribution by species",
    )

    fig.update_xaxes(title="Prediction confidence")
    fig.update_yaxes(title="Particle count")

    st.plotly_chart(fig, use_container_width=True)

with col2:
    species_conf = (
        selected_df
        .groupby("predicted_label")
        .agg(
            mean_confidence=("prediction_confidence", "mean"),
            median_confidence=("prediction_confidence", "median"),
            particle_count=("predicted_label", "size"),
        )
        .reset_index()
        .sort_values("mean_confidence", ascending=False)
    )

    fig = px.bar(
        species_conf,
        x="predicted_label",
        y="mean_confidence",
        text=species_conf["mean_confidence"].round(3),
        title="Mean confidence by predicted species",
    )

    fig.update_yaxes(title="Mean confidence", range=[0, 1])
    fig.update_xaxes(title="Predicted species")

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Raw File Timeline")

if selected_df[time_col].isna().all():
    st.info("No valid timestamp column available for this raw file.")
else:
    time_bin = st.selectbox(
        "Timeline bin",
        ["1s", "5s", "10s", "30s", "1min"],
        index=2,
        key="raw_file_timeline_bin",
    )

    timeline = selected_df.dropna(subset=[time_col]).copy()
    timeline["time_bin"] = timeline[time_col].dt.floor(time_bin)

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
        title="Particle predictions over time within raw file",
    )

    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Particle count")

    st.plotly_chart(fig, use_container_width=True)

    timeline_counts["total_in_bin"] = (
        timeline_counts
        .groupby("time_bin")["count"]
        .transform("sum")
    )

    timeline_counts["percentage"] = (
        timeline_counts["count"]
        / timeline_counts["total_in_bin"]
        * 100
    )

    fig = px.line(
        timeline_counts,
        x="time_bin",
        y="percentage",
        color="predicted_label",
        markers=True,
        title="Composition percentage over time within raw file",
    )

    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Percentage", range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Particle Profile Summary")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        selected_df,
        x="size",
        color="predicted_label",
        nbins=40,
        title="Particle size distribution",
    )

    fig.update_xaxes(title="Size (µm)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        selected_df,
        x="size",
        y="time_asymmetry",
        color="predicted_label",
        hover_data=[
            "particle_index",
            "prediction_confidence",
        ],
        title="Size vs time asymmetry",
    )

    fig.update_xaxes(title="Size (µm)")
    fig.update_yaxes(title="Time asymmetry")

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Raw File Table")

display_cols = [
    "timestamp",
    "raw_file",
    "particle_index",
    "size",
    "time_asymmetry",
    "predicted_label",
    "prediction_confidence",
]

available_display_cols = [col for col in display_cols if col in selected_df.columns]

st.dataframe(
    selected_df[available_display_cols].sort_values(
        "particle_index",
        ascending=True,
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Loaded at most the most recent {MAX_ROWS_TO_LOAD:,} rows from SQLite: "
    f"{DB_PATH}."
)