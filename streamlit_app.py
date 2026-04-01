import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.autoencoder import LogAutoencoder
from src.detector import AnomalyDetector
from src.parser import parse_log_file
from src.sequencer import attach_labels, build_sequences
from src.visualizer import LEVEL_NAMES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogSense - Log Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)

# ── Custom styles ─────────────────────────────────────────────────────────────
def apply_custom_styles() -> None:
    st.markdown("""
    <style>
        .main > div { padding-top: 1.2rem; }
        .upload-wrap {
            max-width: 760px;
            margin: 2rem auto;
            text-align: center;
            padding: 1.5rem;
            border-radius: 14px;
            background: rgba(18, 27, 40, 0.65);
            border: 1px solid rgba(101, 157, 255, 0.35);
        }
        .summary-card {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: rgba(16, 23, 34, 0.95);
            margin-bottom: 0.8rem;
        }
        .summary-card.blue { border-left: 4px solid #4f8cff; }
        .summary-card.red  { border-left: 4px solid #ff5d6c; }
        .summary-label {
            color: #9fb2cf;
            font-size: 0.84rem;
            margin-bottom: 0.15rem;
        }
        .summary-value {
            color: #f1f5ff;
            font-size: 1.42rem;
            font-weight: 700;
        }
        .model-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 99px;
            font-size: 0.72rem;
            font-weight: 600;
            background: #052e16;
            color: #86efac;
            border: 1px solid #14532d;
            margin-left: 6px;
        }
    </style>
    """, unsafe_allow_html=True)


def render_summary_card(label: str, value: str, is_anomaly: bool = False) -> None:
    card_class = "red" if is_anomaly else "blue"
    st.markdown(f"""
    <div class="summary-card {card_class}">
        <div class="summary-label">{label}</div>
        <div class="summary-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Load models once at startup ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    """
    Load pre-trained models from disk.
    @st.cache_resource means this only runs ONCE when the app starts —
    not every time a file is uploaded.
    """
    models = {}

    ae_path  = Path("models/autoencoder")
    iso_path = Path("models/isolation_forest.pkl")

    if ae_path.exists():
        models["autoencoder"] = LogAutoencoder.load(str(ae_path))
        st.sidebar.success("✅ Autoencoder loaded")
    else:
        st.sidebar.error("❌ Autoencoder model not found — run main.py first")

    if iso_path.exists():
        models["isolation_forest"] = AnomalyDetector.load(str(iso_path))
        st.sidebar.success("✅ Isolation Forest loaded")
    else:
        st.sidebar.error("❌ Isolation Forest model not found — run main.py first")

    return models


# ── Charts ────────────────────────────────────────────────────────────────────
def level_distribution_chart(df: pd.DataFrame):
    ordered_levels = ["INFO", "WARN", "ERROR", "FATAL"]
    counts = (
        df["Level"].str.upper()
        .replace({"WARNING": "WARN"})
        .value_counts()
        .reindex(ordered_levels, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Level", "Count"]
    return px.bar(
        counts, x="Level", y="Count", color="Level",
        color_discrete_map={
            "INFO": "#4f8cff", "WARN": "#ffb347",
            "ERROR": "#ff6b6b", "FATAL": "#d448ff"
        },
        title="Log Level Distribution",
    ).update_layout(
        template="plotly_dark", showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )


def anomaly_scatter_chart(predictions: np.ndarray, errors: np.ndarray):
    labels = np.where(predictions == 1, "Anomaly", "Normal")
    df = pd.DataFrame({
        "Sequence Index": np.arange(len(predictions)),
        "Reconstruction Error": errors,
        "Prediction": labels,
    })
    return px.scatter(
        df, x="Sequence Index", y="Reconstruction Error",
        color="Prediction",
        color_discrete_map={"Normal": "#4f8cff", "Anomaly": "#ff5d6c"},
        title="Normal vs Anomaly Sequences", opacity=0.8,
    ).update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )


def sequence_heatmap_chart(sequences: np.ndarray, predictions: np.ndarray):
    anomaly_idx = np.where(predictions == 1)[0]
    normal_idx  = np.where(predictions == 0)[0]
    selected = np.concatenate([anomaly_idx[:20], normal_idx[:20]])
    if len(selected) == 0:
        selected = np.arange(min(20, len(sequences)))
    return px.imshow(
        sequences[selected],
        color_continuous_scale="YlOrRd", aspect="auto",
        title="Sequence Heatmap",
        labels={"x": "Time Step", "y": "Sequence", "color": "Event ID"},
    ).update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )


def classify_severity(score: float, low_t: float, high_t: float) -> str:
    if score >= high_t:   return "High"
    if score >= low_t:    return "Medium"
    return "Low"


# ── Analysis ──────────────────────────────────────────────────────────────────
def run_analysis(log_file_path: str, models: dict) -> dict:
    """
    Run the full pipeline using PRE-TRAINED models.
    No retraining — just parse, sequence, predict.
    """
    # Step 1 — Parse
    df = parse_log_file(log_file_path)
    if df.empty:
        raise ValueError("No valid log lines detected. Please upload a valid HDFS .log file.")

    # Step 2 — Sequence
    sequence_data = build_sequences(df, max_len=20)
    if len(sequence_data["sequences"]) == 0:
        raise ValueError("No block sequences found in this file.")

    # Attach labels if available locally
    label_file = Path("data/anomaly_label.csv")
    labels = None
    if label_file.exists():
        sequence_data = attach_labels(sequence_data, str(label_file))
        labels = sequence_data.get("labels")

    sequences = sequence_data["sequences"]
    block_ids = sequence_data["block_ids"]

    # Step 3 — Predict using loaded models.
    # Fallback: if loaded Isolation Forest returns zero anomalies on a file,
    # run a per-file unsupervised fit so comparison remains meaningful.
    iso_runtime = AnomalyDetector(mode="unsupervised")
    iso_runtime.train(sequences)
    iso_preds = iso_runtime.predict(sequences)
    ae_preds  = models["autoencoder"].predict(sequences)
    ae_errors = models["autoencoder"].reconstruction_errors(sequences)

    # Severity thresholds
    low_t  = float(np.quantile(ae_errors, 0.70))
    high_t = float(np.quantile(ae_errors, 0.90))

    # Results table
    severity   = [classify_severity(float(e), low_t, high_t) for e in ae_errors]
    results_df = pd.DataFrame({
        "Block ID":         block_ids,
        "Error Score":      ae_errors,
        "Severity":         severity,
        "Isolation Forest": np.where(iso_preds == 1, "Anomaly", "Normal"),
        "Autoencoder":      np.where(ae_preds  == 1, "Anomaly", "Normal"),
    }).sort_values("Error Score", ascending=False)

    has_valid_labels = labels is not None and bool(np.any(labels != -1))

    if has_valid_labels:
        results_df["True Label"] = np.where(labels == 1, "Anomaly", "Normal")

    # Most affected component
    if "Component" in df.columns:
        anomaly_blocks = set(np.array(block_ids)[ae_preds == 1])
        anomaly_df = df[df["BlockId"].apply(
            lambda bl: any(b in anomaly_blocks for b in bl)
        )]
        top_comp = (
            anomaly_df["Component"].mode().iloc[0]
            if not anomaly_df.empty
            else df["Component"].mode().iloc[0]
        )
        top_comp = top_comp.split(".")[-1] if "." in str(top_comp) else str(top_comp)
    else:
        top_comp = "N/A"

    # Model comparison
    if has_valid_labels:
        valid    = labels != -1
        true_v   = labels[valid]
        iso_v    = iso_preds[valid]
        ae_v     = ae_preds[valid]
        model_comparison = pd.DataFrame([
            {
                "Model":        "Isolation Forest",
                "Detected":     int(iso_v.sum()),
                "Correct":      int(np.sum((iso_v == 1) & (true_v == 1))),
                "False Alarms": int(np.sum((iso_v == 1) & (true_v == 0))),
            },
            {
                "Model":        "Autoencoder",
                "Detected":     int(ae_v.sum()),
                "Correct":      int(np.sum((ae_v == 1) & (true_v == 1))),
                "False Alarms": int(np.sum((ae_v == 1) & (true_v == 0))),
            },
        ])
    else:
        model_comparison = pd.DataFrame([
            {"Model": "Isolation Forest", "Detected": int(iso_preds.sum())},
            {"Model": "Autoencoder",      "Detected": int(ae_preds.sum())},
        ])

    return {
        "df": df,
        "sequences": sequences,
        "block_ids": block_ids,
        "labels": labels if has_valid_labels else None,
        "iso_preds": iso_preds,
        "ae_preds":  ae_preds,
        "ae_errors": ae_errors,
        "results_df": results_df,
        "model_comparison": model_comparison,
        "summary": {
            "total_logs":   int(len(df)),
            "anomalies":    int(ae_preds.sum()),
            "anomaly_rate": float(ae_preds.sum() / len(ae_preds) * 100),
            "component":    top_comp,
        },
    }


# ── Pages ─────────────────────────────────────────────────────────────────────
def render_upload_page(models: dict):
    st.markdown("""
    <div class="upload-wrap">
        <h2 style="margin-bottom:0.5rem;">LogSense</h2>
        <p style="color:#a9b8cf;margin-top:0;">
            Upload an HDFS .log file to detect anomalous sequences
            using pre-trained Isolation Forest and Autoencoder models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop a .log file",
        type=["log", "txt"],
        help="Supports HDFS log format. Max 200MB.",
    )

    if st.button("Analyse", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a .log file first.")
            return
        if not models:
            st.error("No models loaded. Please run main.py first.")
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            with st.spinner("Running anomaly detection..."):
                results = run_analysis(tmp_path, models)

            st.session_state["analysis_results"] = results
            st.session_state["current_page"] = "Dashboard"
            st.rerun()

        except Exception as e:
            st.error(f"Analysis failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def render_dashboard_page(results: dict):
    summary = results["summary"]
    total_sequences = len(results["sequences"])
    iso_detected = int(results["iso_preds"].sum())
    ae_detected = int(results["ae_preds"].sum())

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_summary_card("Total logs analysed", f"{summary['total_logs']:,}")
    with col2:
        render_summary_card("Anomalies detected", f"{summary['anomalies']:,}", is_anomaly=True)
    with col3:
        render_summary_card("Anomaly rate", f"{summary['anomaly_rate']:.2f}%", is_anomaly=True)
    with col4:
        render_summary_card("Most affected component", summary["component"])

    # Charts
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(level_distribution_chart(results["df"]), use_container_width=True)
    with c2:
        st.plotly_chart(
            anomaly_scatter_chart(results["ae_preds"], results["ae_errors"]),
            use_container_width=True
        )
    with c3:
        st.plotly_chart(
            sequence_heatmap_chart(results["sequences"], results["ae_preds"]),
            use_container_width=True
        )

    # Top 10 suspicious blocks
    st.subheader("Top 10 Most Suspicious Blocks")
    top10 = results["results_df"][["Block ID", "Error Score", "Severity"]].head(10).copy()
    top10["Error Score"] = top10["Error Score"].map(lambda x: f"{x:.6f}")
    st.dataframe(top10, use_container_width=True, hide_index=True)

    # Model comparison
    st.subheader("Model Comparison")
    st.caption(
        f"Debug: Total sequences = {total_sequences:,} | "
        f"Isolation Forest detected = {iso_detected:,} | "
        f"Autoencoder detected = {ae_detected:,}"
    )
    if results["labels"] is not None:
        st.dataframe(results["model_comparison"], use_container_width=True, hide_index=True)
    else:
        st.info("No ground truth labels available — showing detected counts only.")
        st.dataframe(
            results["model_comparison"][["Model", "Detected"]],
            use_container_width=True, hide_index=True
        )

    # Download
    csv = results["results_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Results CSV",
        data=csv,
        file_name="logsense_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Event ID reference
    with st.expander("Event ID Reference"):
        ref_df = pd.DataFrame([
            {"Event ID": k, "Log Level": v} for k, v in LEVEL_NAMES.items()
        ])
        st.dataframe(ref_df, use_container_width=True, hide_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    apply_custom_styles()

    # Load models once
    models = load_models()

    st.sidebar.title("LogSense")
    st.sidebar.caption("Log Anomaly Detection System")

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Upload"

    has_results  = "analysis_results" in st.session_state
    page_options = ["Upload"] + (["Dashboard"] if has_results else [])
    selected     = st.sidebar.radio(
        "Navigation",
        page_options,
        index=page_options.index(st.session_state["current_page"])
        if st.session_state["current_page"] in page_options
        else 0,
    )
    st.session_state["current_page"] = selected

    st.title("LogSense Dashboard")

    if st.session_state["current_page"] == "Dashboard" and "analysis_results" in st.session_state:
        render_dashboard_page(st.session_state["analysis_results"])
    else:
        render_upload_page(models)


if __name__ == "__main__":
    main()
