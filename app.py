from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src.config import (
    CLEAN_DATA_PATH,
    CLUSTER_MODEL_PATH,
    FEATURE_COLUMNS,
    EFFORT_MODEL_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    PRIORITY_MODEL_PATH,
    RAW_DIR,
)
from src.data_prep import (
    clean_dataset,
    ensure_feature_frame,
    load_csv,
    prepare_classification_data,
    prepare_cluster_data,
    prepare_regression_data,
)
from src.history_store import append_predictions, read_history
from src.inference import predict_bulk, predict_single_startup
from src.model_store import load_joblib, load_json, save_joblib, save_json
from src.modeling import train_cluster_model, train_effort_model, train_priority_model
from src.ui import inject_global_css, kpi_card, title_block


st.set_page_config(
    page_title="Startup Predictor",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


@st.cache_data
def load_clean_data(file_or_path) -> pd.DataFrame:
    frame = load_csv(file_or_path)
    return clean_dataset(frame, keep_status_filter=True)


def load_model_bundles() -> dict | None:
    if not (PRIORITY_MODEL_PATH.exists() and EFFORT_MODEL_PATH.exists() and CLUSTER_MODEL_PATH.exists()):
        return None

    return {
        "priority": load_joblib(PRIORITY_MODEL_PATH),
        "effort": load_joblib(EFFORT_MODEL_PATH),
        "cluster": load_joblib(CLUSTER_MODEL_PATH),
    }


def train_and_persist_models(df: pd.DataFrame) -> dict:
    x_cls, y_cls, _ = prepare_classification_data(df)
    x_reg, y_reg, _ = prepare_regression_data(df)
    x_cluster, _ = prepare_cluster_data(df)

    priority_bundle = train_priority_model(x_cls, y_cls)
    effort_bundle = train_effort_model(x_reg, y_reg)
    cluster_bundle = train_cluster_model(x_cluster)

    save_joblib(PRIORITY_MODEL_PATH, priority_bundle)
    save_joblib(EFFORT_MODEL_PATH, effort_bundle)
    save_joblib(CLUSTER_MODEL_PATH, cluster_bundle)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    metrics = {
        "priority": priority_bundle["metrics"],
        "effort": effort_bundle["metrics"],
        "cluster": cluster_bundle["metrics"],
        "rows_used": int(len(df)),
    }
    save_json(METRICS_PATH, metrics)

    return {
        "priority": priority_bundle,
        "effort": effort_bundle,
        "cluster": cluster_bundle,
    }


def status_badge(label: str) -> str:
    if label.lower() == "acquired":
        return "badge-good"
    return "badge-risk"


def feature_label(name: str) -> str:
    return name.replace("_", " ").title()


def artifact_status_table() -> pd.DataFrame:
    items = [
        ("cleaned_dataset", CLEAN_DATA_PATH.exists()),
        ("priority_model", PRIORITY_MODEL_PATH.exists()),
        ("effort_model", EFFORT_MODEL_PATH.exists()),
        ("cluster_model", CLUSTER_MODEL_PATH.exists()),
        ("metrics_json", METRICS_PATH.exists()),
        ("prediction_history", HISTORY_PATH.exists()),
    ]
    return pd.DataFrame(
        {
            "artifact": [name for name, _ in items],
            "status": ["Available" if exists else "Missing" for _, exists in items],
        }
    )


def build_csv_template() -> bytes:
    template = pd.DataFrame([{**{c: 0 for c in FEATURE_COLUMNS}, "status": "acquired"}])
    return template.to_csv(index=False).encode("utf-8")


with st.sidebar:
    st.markdown("## Startup Predictor")
    st.caption("Predict startup outcomes with ML insights")

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Startup Analyzer",
            "Bulk Upload",
            "Analytics",
            "Model Center",
            "Clusters",
            "Prediction History",
        ],
    )

    st.markdown("---")
    st.markdown("### Training Source")
    uploaded = st.file_uploader("Upload startup CSV", type=["csv"])

    st.download_button(
        "Download CSV Template",
        data=build_csv_template(),
        file_name="startup_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if uploaded is not None:
        st.success("Dataset uploaded.")


data = None
data_source = None
if uploaded is not None:
    try:
        data = load_clean_data(uploaded)
        data_source = "uploaded"
    except Exception as exc:
        st.error(f"Failed to read uploaded dataset: {exc}")
elif CLEAN_DATA_PATH.exists():
    data = load_clean_data(CLEAN_DATA_PATH)
    data_source = "processed"
elif (RAW_DIR / "demo_startups.csv").exists():
    data = load_clean_data(RAW_DIR / "demo_startups.csv")
    data_source = "demo"

bundles = load_model_bundles()

with st.sidebar:
    st.markdown("---")
    st.markdown("### Model Status")
    if bundles is None:
        st.warning("Models not trained yet")
    else:
        st.success("Models available")

    if data is not None:
        st.caption(f"Training rows: {len(data):,}")

    if data_source == "demo":
        st.info("Using bundled demo data. Upload your own CSV for custom training.")

    if data is not None:
        if st.button("Train / Retrain Models", use_container_width=True):
            try:
                with st.spinner("Training all models and persisting artifacts..."):
                    bundles = train_and_persist_models(data)
                st.success("Training complete. Models saved in models/.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    else:
        st.info("Upload data or keep data/processed/startup_clean.csv to train.")


if page == "Dashboard":
    title_block("Dashboard", "Overview of data health and startup outcome signals")

    if data is None:
        st.warning("No dataset available yet. Upload a CSV from the sidebar to begin.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Rows", f"{len(data):,}")
    with c2:
        acquired_rate = (data["status"] == "acquired").mean() * 100
        kpi_card("Acquired Rate", f"{acquired_rate:.1f}%")
    with c3:
        med_funding = data["funding_total_usd"].median()
        kpi_card("Median Funding", f"${med_funding:,.0f}")
    with c4:
        model_state = "Ready" if bundles is not None else "Not Trained"
        kpi_card("ML Pipeline", model_state)

    st.markdown("### Outcome Distribution")
    status_counts = data["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig_status = px.bar(
        status_counts,
        x="status",
        y="count",
        color="status",
        color_discrete_map={"acquired": "#16a34a", "closed": "#dc2626"},
    )
    fig_status.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_status, use_container_width=True)

    st.markdown("### Funding vs Milestones")
    fig_scatter = px.scatter(
        data,
        x="funding_rounds",
        y="funding_total_usd",
        color="status",
        size="milestones",
        hover_data=["relationships", "avg_participants"],
        color_discrete_map={"acquired": "#16a34a", "closed": "#dc2626"},
    )
    fig_scatter.update_layout(height=440, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)


elif page == "Startup Analyzer":
    title_block("Startup Analyzer", "Single startup prediction with confidence, risk and segment intelligence")

    if bundles is None:
        st.warning("No trained models found. Upload data and click 'Train / Retrain Models' first.")
        st.stop()

    features = bundles["priority"]["features"]
    defaults = {}
    if data is not None:
        for col in features:
            defaults[col] = float(data[col].median()) if col in data.columns else 0.0
    else:
        defaults = {col: 0.0 for col in features}

    st.markdown("### Input Startup Profile")
    cols = st.columns(3)
    values = {}

    for idx, col in enumerate(features):
        with cols[idx % 3]:
            values[col] = st.number_input(
                label=feature_label(col),
                value=float(defaults[col]),
                step=1.0,
                format="%.4f",
            )

    if st.button("Analyze Startup", use_container_width=True):
        row = pd.DataFrame([values])
        result = predict_single_startup(
            row,
            bundles["priority"],
            bundles["effort"],
            bundles["cluster"],
        )

        badge_class = status_badge(result["predicted_status"])
        st.markdown(
            f"<div class='{badge_class}'>{result['predicted_status']}</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
        c2.metric("Risk Level", result["risk_level"])
        c3.metric("Segment", f"Cluster {result['segment']}")

        st.metric("Estimated Resource Intensity (USD)", f"${result['effort_estimate']:,.2f}")

        record = {**values, **result}
        append_predictions([record])
        st.success("Prediction saved to history.")


elif page == "Bulk Upload":
    title_block("Bulk Upload", "Batch startup scoring with downloadable results and history persistence")

    if bundles is None:
        st.warning("No trained models found. Train models first.")
        st.stop()

    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")

    if batch_file is not None:
        raw_batch = load_csv(batch_file)
        batch = clean_dataset(raw_batch, keep_status_filter=False)

        st.write(f"Rows detected: {len(batch)}")
        st.dataframe(batch.head(10), use_container_width=True)

        if st.button("Process Batch", use_container_width=True):
            results = predict_bulk(
                batch,
                bundles["priority"],
                bundles["effort"],
                bundles["cluster"],
            )
            st.success("Batch scoring complete")
            st.dataframe(results.head(20), use_container_width=True)

            append_predictions(results.to_dict(orient="records"))

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv,
                file_name="startup_bulk_predictions.csv",
                mime="text/csv",
            )


elif page == "Analytics":
    title_block("Analytics", "Correlations, trend analysis and outcome diagnostics")

    if data is None:
        st.warning("No data available for analytics.")
        st.stop()

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("### Correlation Matrix")
    if len(num_cols) >= 2:
        corr = data[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        fig_corr.update_layout(height=580)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Outcome Trend by First Funding Year")
    if "age_first_funding_year" in data.columns:
        yearly = (
            data.groupby("age_first_funding_year")["status"]
            .apply(lambda s: (s == "acquired").mean() * 100)
            .reset_index(name="acquired_rate")
            .sort_values("age_first_funding_year")
        )
        fig_line = px.line(yearly, x="age_first_funding_year", y="acquired_rate", markers=True)
        fig_line.update_layout(height=380)
        st.plotly_chart(fig_line, use_container_width=True)


elif page == "Model Center":
    title_block("Model Center", "Model metrics and system artifact health")

    st.markdown("### Artifact Status")
    st.dataframe(artifact_status_table(), use_container_width=True)

    if METRICS_PATH.exists():
        metrics = load_json(METRICS_PATH)

        st.markdown("### Priority Classifier (SVC)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['priority']['accuracy']:.3f}")
        c2.metric("F1", f"{metrics['priority']['f1']:.3f}")
        c3.metric("ROC AUC", f"{metrics['priority']['roc_auc']:.3f}")

        st.markdown("### Effort Regressor (Polynomial)")
        c4, c5, c6 = st.columns(3)
        c4.metric("R2", f"{metrics['effort']['r2']:.3f}")
        c5.metric("RMSE", f"{metrics['effort']['rmse']:.2f}")
        c6.metric("MAE", f"{metrics['effort']['mae']:.2f}")

        st.markdown("### Clustering (K-Means)")
        c7, c8 = st.columns(2)
        c7.metric("Clusters", str(metrics["cluster"]["clusters"]))
        c8.metric("Silhouette", f"{metrics['cluster']['silhouette']:.3f}")

        cm = metrics["priority"]["confusion_matrix"]
        cm_df = pd.DataFrame(cm, columns=["Pred Closed", "Pred Acquired"], index=["Actual Closed", "Actual Acquired"])
        st.markdown("### Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

        with st.expander("Raw Metrics JSON"):
            st.json(metrics)
    else:
        st.warning("No saved model metrics found yet.")


elif page == "Clusters":
    title_block("Clusters", "Customer-style segmentation view for startup cohorts")

    if bundles is None or data is None:
        st.warning("Need trained clustering model and dataset to render this page.")
        st.stop()

    c_bundle = bundles["cluster"]
    x = ensure_feature_frame(data, c_bundle["features"])
    x_scaled = c_bundle["scaler"].transform(x)
    labels = c_bundle["model"].predict(x_scaled)

    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(x_scaled)

    chart_df = pd.DataFrame({
        "pc1": points[:, 0],
        "pc2": points[:, 1],
        "segment": labels.astype(str),
        "status": data["status"].values,
    })

    fig = px.scatter(
        chart_df,
        x="pc1",
        y="pc2",
        color="segment",
        symbol="status",
        opacity=0.8,
        title="PCA Projection of Startup Segments",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Profiles")
    profile = x.copy()
    profile["segment"] = labels
    st.dataframe(profile.groupby("segment").mean(numeric_only=True).round(2), use_container_width=True)


elif page == "Prediction History":
    title_block("Prediction History", "Audit trail of every single and batch prediction")

    hist = read_history()
    if hist.empty:
        st.info("No prediction history yet.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        status_filter = st.selectbox("Filter by Status", ["All"] + sorted(hist["predicted_status"].dropna().unique().tolist()))
    with col_b:
        risk_filter = st.selectbox("Filter by Risk", ["All"] + sorted(hist["risk_level"].dropna().unique().tolist()))

    filtered = hist.copy()
    if status_filter != "All":
        filtered = filtered[filtered["predicted_status"] == status_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]

    st.write(f"Records: {len(filtered)}")
    st.dataframe(filtered.sort_values("predicted_at", ascending=False), use_container_width=True)
