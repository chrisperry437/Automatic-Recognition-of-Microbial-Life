from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


DB_PATH = Path("./results/realtime/predictions.sqlite")
MAX_ROWS_TO_LOAD = 50_000

st.set_page_config(
    page_title="Particle Profiles",
    page_icon="🔬",
    layout="wide",
)

st.divider()
st.title("🔬 Particle Profiles")


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


prob_cols = [
    "prob_B_cereus",
    "prob_B_endophyticus",
    "prob_K_salsicia",
    "prob_M_luteus",
    "prob_S_huminis",
]

available_prob_cols = [col for col in prob_cols if col in df.columns]


st.subheader("Filters")

filter_col1, filter_col2 = st.columns([1, 2])

with filter_col1:
    selected_label = st.selectbox(
        "Predicted class",
        ["All"] + sorted(df["predicted_label"].dropna().unique().tolist()),
        key="particle_profiles_label_filter",
    )

with filter_col2:
    min_confidence = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="particle_profiles_min_confidence",
    )

filtered = df.copy()

if selected_label != "All":
    filtered = filtered[filtered["predicted_label"] == selected_label].copy()

filtered = filtered[filtered["prediction_confidence"] >= min_confidence].copy()

if filtered.empty:
    st.warning("No particles match the selected filters.")
    st.stop()


st.divider()

st.subheader("Select a Particle")

filtered = filtered.sort_values(
    ["timestamp", "raw_file", "particle_index"],
    ascending=[False, True, True],
).reset_index(drop=True)

raw_files = filtered["raw_file"].dropna().unique().tolist()

selected_raw_file = st.selectbox(
    "Raw file",
    raw_files,
    key="particle_profiles_raw_file_selector",
)

file_particles = filtered[
    filtered["raw_file"] == selected_raw_file
].copy()

available_classes = (
    file_particles["predicted_label"]
    .dropna()
    .sort_values()
    .unique()
    .tolist()
)

selected_particle_class = st.selectbox(
    "Predicted class within raw file",
    ["All"] + available_classes,
    key="particle_profiles_file_class_selector",
)

if selected_particle_class != "All":
    file_particles = file_particles[
        file_particles["predicted_label"] == selected_particle_class
    ].copy()

confidence_range = st.slider(
    "Confidence range within raw file",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01,
    key="particle_profiles_confidence_range",
)

file_particles = file_particles[
    file_particles["prediction_confidence"].between(
        confidence_range[0],
        confidence_range[1],
    )
].copy()

if file_particles.empty:
    st.warning("No particles match this file/class/confidence filter.")
    st.stop()

particle_indices = (
    file_particles["particle_index"]
    .astype(int)
    .sort_values()
    .unique()
    .tolist()
)

selected_particle_index = st.selectbox(
    "Particle index",
    particle_indices,
    key="particle_profiles_particle_index_selector",
)

particle_matches = file_particles[
    file_particles["particle_index"].astype(int) == selected_particle_index
]

particle = particle_matches.iloc[0]


col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted class", particle["predicted_label"])
col2.metric("Confidence", f"{particle['prediction_confidence']:.1%}")
col3.metric("Size", f"{particle['size']:.2f} µm")
col4.metric("Time asymmetry", f"{particle['time_asymmetry']:.3f}")


st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Particle Metadata")

    metadata_cols = [
        "source_file",
        "raw_file",
        "particle_index",
        "timestamp",
        "processed_at",
        "event_time",
        "stored_at",
        "predicted_class_index",
    ]

    metadata = {
        col: particle[col]
        for col in metadata_cols
        if col in particle.index
    }

    st.json(
        {
            key: str(value)
            for key, value in metadata.items()
        }
    )

with right:
    st.subheader("Class Probabilities")

    if available_prob_cols:
        probs = pd.DataFrame(
            {
                "class": [
                    col.replace("prob_", "").replace("_", " ")
                    for col in available_prob_cols
                ],
                "probability": [
                    float(particle[col])
                    for col in available_prob_cols
                ],
            }
        ).sort_values("probability", ascending=False)

        fig = px.bar(
            probs,
            x="class",
            y="probability",
            title="Prediction probability by class",
        )

        fig.update_yaxes(range=[0, 1], title="Probability")
        fig.update_xaxes(title="Class")

        st.plotly_chart(fig, use_container_width=True)

        top_two = probs.head(2).reset_index(drop=True)

        if len(top_two) >= 2:
            st.info(
                f"Top alternative: {top_two.loc[1, 'class']} "
                f"({top_two.loc[1, 'probability']:.1%})"
            )
    else:
        st.info("No probability columns found.")


st.divider()

st.subheader("Particle Context")

context_window = st.slider(
    "Nearby particles to show",
    min_value=5,
    max_value=100,
    value=25,
    key="particle_profiles_context_window",
)

same_file = df[df["raw_file"] == particle["raw_file"]].copy()
same_file = same_file.sort_values("particle_index").reset_index(drop=True)

matching_positions = same_file.index[
    same_file["particle_index"].astype(int) == int(particle["particle_index"])
].tolist()

if matching_positions:
    pos = matching_positions[0]

    start_pos = max(pos - context_window, 0)
    end_pos = min(pos + context_window + 1, len(same_file))

    context = same_file.iloc[start_pos:end_pos].copy()

    fig = px.scatter(
        context,
        x="particle_index",
        y="prediction_confidence",
        color="predicted_label",
        hover_data=[
            "size",
            "time_asymmetry",
            "prediction_confidence",
        ],
        title="Prediction confidence near selected particle",
    )

    fig.add_vline(
        x=particle["particle_index"],
        line_dash="dash",
        annotation_text="selected particle",
    )

    fig.update_yaxes(range=[0, 1], title="Confidence")
    fig.update_xaxes(title="Particle index")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Could not locate selected particle in its raw file.")


st.divider()

st.subheader("Profile Distributions for Similar Particles")

similar = df[df["predicted_label"] == particle["predicted_label"]].copy()

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        similar,
        x="size",
        nbins=40,
        title=f"Size distribution for {particle['predicted_label']}",
    )

    fig.add_vline(
        x=float(particle["size"]),
        line_dash="dash",
        annotation_text="selected particle",
    )

    fig.update_xaxes(title="Size (µm)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        similar,
        x="time_asymmetry",
        nbins=40,
        title=f"Time asymmetry distribution for {particle['predicted_label']}",
    )

    fig.add_vline(
        x=float(particle["time_asymmetry"]),
        line_dash="dash",
        annotation_text="selected particle",
    )

    fig.update_xaxes(title="Time asymmetry")
    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Recent Matching Particles")

display_cols = [
    "timestamp",
    "raw_file",
    "particle_index",
    "size",
    "time_asymmetry",
    "predicted_label",
    "prediction_confidence",
]

available_display_cols = [col for col in display_cols if col in filtered.columns]

st.dataframe(
    filtered[available_display_cols].head(100),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Loaded at most the most recent {MAX_ROWS_TO_LOAD:,} rows from SQLite: "
    f"{DB_PATH}."
)