import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


MODELS_DIR = Path("models/trained")
CONFIG_PATH = Path("configs/active_model.json")
DB_PATH = Path("./results/realtime/predictions.sqlite")

st.set_page_config(
    page_title="Model Selection",
    page_icon="🧠",
    layout="wide",
)

st.divider()
st.title("🧠 Model Selection")


def load_active_model() -> str | None:
    if not CONFIG_PATH.exists():
        return None

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("active_model")


def save_active_model(model_name: str) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump({"active_model": model_name}, f, indent=2)


@st.cache_data(ttl=2)
def load_recent_predictions_from_db(
    db_path: Path,
    limit: int = 1000,
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


if not MODELS_DIR.exists():
    st.warning(f"Models directory not found: {MODELS_DIR}")
    st.stop()


model_files = sorted(
    [
        p for p in MODELS_DIR.iterdir()
        if p.is_file() and p.suffix in [".pt", ".joblib", ".pkl"]
    ]
)

model_dirs = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()])
available_models = model_dirs + model_files

if not available_models:
    st.warning(f"No models found in {MODELS_DIR}")
    st.stop()


model_names = [p.name for p in available_models]
active_model = load_active_model()

default_index = model_names.index(active_model) if active_model in model_names else 0

selected_model = st.selectbox(
    "Select active model",
    model_names,
    index=default_index,
    key="active_model_selector",
)

if st.button("Set active model", key="set_active_model_button"):
    save_active_model(selected_model)
    st.success(f"Active model updated to: {selected_model}")

active_model = load_active_model()

st.subheader("Current Active Model")
st.code(active_model or "No active model selected")


st.divider()

st.subheader("Live Prediction Summary From SQLite")

recent_df = load_recent_predictions_from_db(DB_PATH, limit=1000)

if recent_df.empty:
    st.info(f"No recent predictions found in SQLite database: {DB_PATH}")
else:
    latest = recent_df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Recent rows loaded", f"{len(recent_df):,}")
    col2.metric("Latest prediction", latest.get("predicted_label", "N/A"))

    if "prediction_confidence" in recent_df.columns:
        col3.metric(
            "Mean confidence",
            f"{recent_df['prediction_confidence'].mean():.1%}",
        )
    else:
        col3.metric("Mean confidence", "N/A")

    if "timestamp" in recent_df.columns:
        latest_time = recent_df["timestamp"].dropna().max()
        col4.metric("Latest timestamp", str(latest_time))
    else:
        col4.metric("Latest timestamp", "N/A")

    if "predicted_label" in recent_df.columns:
        live_counts = recent_df["predicted_label"].value_counts().reset_index()
        live_counts.columns = ["predicted_label", "count"]

        fig = px.bar(
            live_counts,
            x="predicted_label",
            y="count",
            title="Recent prediction distribution from SQLite",
        )

        fig.update_xaxes(title="Predicted species")
        fig.update_yaxes(title="Particle count")

        st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Model Comparison")

comparison_df = pd.DataFrame(
    [
        {
            "Model": "Paper RF",
            "Experiment": "exp00",
            "Accuracy": 0.807,
            "Balanced Accuracy": 0.654,
            "Macro F1": 0.641,
        },
        {
            "Model": "Tuned RF",
            "Experiment": "exp03",
            "Accuracy": 0.853,
            "Balanced Accuracy": 0.716,
            "Macro F1": 0.721,
        },
        {
            "Model": "Baseline CNN",
            "Experiment": "exp04",
            "Accuracy": 0.552,
            "Balanced Accuracy": 0.385,
            "Macro F1": 0.363,
        },
        {
            "Model": "Multimodal Deep Learning",
            "Experiment": "exp05",
            "Accuracy": 0.876,
            "Balanced Accuracy": 0.769,
            "Macro F1": 0.769,
        },
    ]
)

st.dataframe(
    comparison_df.style.format(
        {
            "Accuracy": "{:.3f}",
            "Balanced Accuracy": "{:.3f}",
            "Macro F1": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

metrics_long = comparison_df.melt(
    id_vars=["Model", "Experiment"],
    value_vars=["Accuracy", "Balanced Accuracy", "Macro F1"],
    var_name="Metric",
    value_name="Score",
)

fig = px.bar(
    metrics_long,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Model performance comparison",
)

fig.update_yaxes(title="Score", range=[0, 1])
fig.update_xaxes(title="Model")

st.plotly_chart(fig, use_container_width=True)


best_row = comparison_df.sort_values(
    "Balanced Accuracy",
    ascending=False,
).iloc[0]

st.success(
    f"Best overall model by balanced accuracy: "
    f"{best_row['Model']} ({best_row['Balanced Accuracy']:.3f})"
)


st.divider()

st.subheader("Available Model Artifacts")

for path in available_models:
    st.write(f"**{path.name}**")
    st.caption(str(path))

    metadata_path = path / "metadata.json" if path.is_dir() else None

    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        with st.expander(f"Metadata: {path.name}"):
            st.json(metadata)

st.caption(f"Live prediction summary is loaded from SQLite: {DB_PATH}")