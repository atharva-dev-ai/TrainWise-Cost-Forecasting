"""
TrainWise AI - AI Infrastructure Cost Forecasting Platform
A production-quality Streamlit application that predicts AI training costs
across AWS, Azure, and GCP using a pre-trained XGBoost model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TrainWise AI | Cost Forecasting",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# CUSTOM CSS - Dark mode, glassmorphism, gradient accents
# ---------------------------------------------------------------------------
def load_custom_css():
    """Inject custom CSS for a premium dark SaaS dashboard aesthetic."""
    st.markdown(
        """
        <style>
        /* Global background */
        .stApp {
            background: radial-gradient(circle at 20% 20%, #1a1c2e 0%, #0e0f1a 50%, #0a0b12 100%);
            color: #e6e6f0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* Hide default Streamlit chrome */
        #MainMenu, header, footer {visibility: hidden;}

        /* Hero section */
        .hero-container {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem 1rem;
        }
        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7F5AF0, #2CB1FF, #00E0C6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            color: #b3b3c6;
            font-weight: 400;
        }

        /* Glass card */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }

        /* Metric cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.2rem;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, border 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            border: 1px solid rgba(127, 90, 240, 0.5);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #7F5AF0, #00E0C6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #9a9ab0;
            margin-top: 0.3rem;
        }

        /* Section headers */
        .section-header {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid #7F5AF0;
            padding-left: 0.7rem;
        }

        /* Result card */
        .result-card {
            background: linear-gradient(135deg, rgba(127,90,240,0.18), rgba(0,224,198,0.12));
            border: 1px solid rgba(127, 90, 240, 0.4);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px rgba(127, 90, 240, 0.25);
        }
        .result-cost {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7F5AF0, #2CB1FF, #00E0C6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .result-label {
            font-size: 1rem;
            color: #b3b3c6;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        /* Category badges */
        .badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            margin-top: 0.8rem;
        }
        .badge-low { background: rgba(0, 224, 198, 0.2); color: #00E0C6; border: 1px solid #00E0C6; }
        .badge-medium { background: rgba(44, 177, 255, 0.2); color: #2CB1FF; border: 1px solid #2CB1FF; }
        .badge-high { background: rgba(255, 184, 0, 0.2); color: #FFB800; border: 1px solid #FFB800; }
        .badge-enterprise { background: rgba(255, 90, 122, 0.2); color: #FF5A7A; border: 1px solid #FF5A7A; }

        /* Insight items */
        .insight-item {
            background: rgba(255, 255, 255, 0.04);
            border-left: 3px solid #00E0C6;
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 0.6rem;
            color: #d8d8e8;
        }

        /* Primary button */
        div.stButton > button {
            background: linear-gradient(90deg, #7F5AF0, #2CB1FF);
            color: white;
            font-weight: 700;
            font-size: 1.1rem;
            padding: 0.8rem 2rem;
            border-radius: 14px;
            border: none;
            width: 100%;
            transition: all 0.25s ease;
            box-shadow: 0 4px 20px rgba(127, 90, 240, 0.4);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 28px rgba(127, 90, 240, 0.6);
        }

        /* Footer */
        .footer-container {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #6f6f85;
            font-size: 0.9rem;
            border-top: 1px solid rgba(255,255,255,0.06);
            margin-top: 2rem;
        }
        .footer-pill {
            display: inline-block;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 999px;
            padding: 0.3rem 1rem;
            margin: 0.2rem;
            font-size: 0.85rem;
            color: #b3b3c6;
        }

        /* Inputs */
        .stSelectbox label, .stSlider label, .stNumberInput label {
            color: #c9c9dc !important;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load the trained XGBoost model and feature columns list."""
    model = joblib.load("trainwise_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_input(
    cloud_provider,
    gpu_type,
    gpu_count,
    training_hours,
    model_parameters_billion,
    dataset_size_gb,
    epochs,
    feature_columns,
):
    """
    Build a single-row dataframe from raw user inputs, apply one-hot
    encoding for categorical fields (matching training-time encoding with
    drop_first=True, where 'AWS' and 'A100' are the implicit reference
    categories represented as all-zero dummy columns), and align columns
    to match the feature set the model was trained on.
    """
    # Numeric features
    raw_input = {
        "gpu_count": gpu_count,
        "training_hours": training_hours,
        "model_parameters_billion": model_parameters_billion,
        "dataset_size_gb": dataset_size_gb,
        "epochs": epochs,
    }

    # Manually replicate pd.get_dummies(..., drop_first=True)
    # Reference categories (dropped at training time): AWS, A100
    for provider in ["Azure", "GCP"]:
        raw_input[f"cloud_provider_{provider}"] = 1 if cloud_provider == provider else 0

    for gpu in ["A10G", "H100", "T4"]:
        raw_input[f"gpu_type_{gpu}"] = 1 if gpu_type == gpu else 0

    input_df = pd.DataFrame([raw_input])

    # Align with the model's expected feature columns
    # - add any missing columns with 0
    # - drop any extra columns
    # - reorder to match training feature order
    aligned_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Ensure numeric dtype
    aligned_df = aligned_df.fillna(0).astype(float)

    return aligned_df


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def predict_cost(model, input_df):
    """Run the model prediction and return the estimated cost (float)."""
    prediction = model.predict(input_df)
    return float(prediction[0])


def categorize_cost(cost):
    """Classify the predicted cost into a category bucket."""
    if cost < 1000:
        return "Low", "badge-low"
    elif cost < 10000:
        return "Medium", "badge-medium"
    elif cost < 50000:
        return "High", "badge-high"
    else:
        return "Enterprise", "badge-enterprise"


def get_business_interpretation(category):
    """Return a short business-oriented interpretation for a cost category."""
    interpretations = {
        "Low": "Suitable for experimentation and research workloads.",
        "Medium": "Suitable for production pilot projects.",
        "High": "Suitable for large-scale enterprise workloads.",
        "Enterprise": "Recommended for dedicated AI infrastructure planning.",
    }
    return interpretations.get(category, "")


# ---------------------------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------------------------
def generate_insights(cloud_provider, gpu_type, gpu_count, training_hours,
                       model_parameters_billion, dataset_size_gb, epochs, cost):
    """Generate dynamic, human-readable insights based on the input values."""
    insights = []

    # Provider-related insight
    provider_notes = {
        "AWS": "AWS often offers competitive spot pricing for large-scale GPU clusters.",
        "Azure": "Azure provides strong enterprise discounts for sustained AI workloads.",
        "GCP": "GCP's TPU and preemptible VM options can reduce costs for this workload.",
    }
    insights.append(f"💡 {provider_notes.get(cloud_provider, 'Cloud provider choice impacts overall cost efficiency.')}")

    # GPU count insight
    if gpu_count >= 8:
        insights.append("⚡ High GPU count is a major driver of total training cost — consider distributed training optimizations.")
    elif gpu_count >= 4:
        insights.append("⚡ Moderate GPU count detected — scaling efficiency will impact cost-per-epoch.")
    else:
        insights.append("⚡ Low GPU count keeps infrastructure costs lean, but may increase total training time.")

    # Training hours insight
    if training_hours > 200:
        insights.append("⏱️ Training duration significantly impacts expenses — long runs amplify compute costs exponentially.")
    elif training_hours > 50:
        insights.append("⏱️ Training duration is a notable cost factor for this configuration.")
    else:
        insights.append("⏱️ Short training duration helps keep this estimate cost-efficient.")

    # Model size insight
    if model_parameters_billion >= 50:
        insights.append("🧠 Large model size (50B+ parameters) substantially increases memory and compute requirements.")
    elif model_parameters_billion >= 10:
        insights.append("🧠 Mid-sized model — balances performance with manageable infrastructure costs.")

    # Dataset size insight
    if dataset_size_gb >= 500:
        insights.append("📦 Large dataset size increases I/O overhead and storage costs across cloud providers.")

    # Epochs insight
    if epochs >= 10:
        insights.append("🔁 High epoch count multiplies total compute time — consider early stopping to control costs.")

    # Overall cost insight
    category, _ = categorize_cost(cost)
    if category == "Enterprise":
        insights.append("🚨 This configuration falls into the Enterprise cost tier — negotiate committed-use discounts.")
    elif category == "Low":
        insights.append("✅ This configuration is highly cost-efficient — great for prototyping and experimentation.")

    return insights


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_columns):
    """Create an interactive Plotly bar chart of model feature importances."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#2CB1FF", "#7F5AF0", "#00E0C6"],
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e6f0", family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )

    return fig


# ---------------------------------------------------------------------------
# UI SECTIONS
# ---------------------------------------------------------------------------
def render_hero():
    """Render the hero / title section."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🚀 TrainWise AI</div>
            <div class="hero-subtitle">Predict AI Training Costs Across AWS, Azure & GCP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics():
    """Render the top metrics row with key model/platform stats."""
    metrics = [
        ("XGBoost", "Model Type"),
        ("0.914", "Cross Validation R²"),
        ("3", "Cloud Providers"),
        ("100,000", "Dataset Size (Records)"),
    ]

    cols = st.columns(4)
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_form():
    """Render the two-column prediction input form and return user inputs."""
    st.markdown('<div class="section-header">🧮 Configure Your Training Job</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        cloud_provider = st.selectbox("☁️ Cloud Provider", ["AWS", "Azure", "GCP"])
        gpu_type = st.selectbox("🎮 GPU Type", ["A100", "H100", "A10G", "T4"])
        gpu_count = st.slider("🔢 GPU Count", min_value=1, max_value=64, value=8, step=1)

    with col2:
        training_hours = st.number_input("⏱️ Training Hours", min_value=1.0, max_value=2000.0, value=100.0, step=1.0)
        model_parameters_billion = st.number_input("🧠 Model Parameters (Billions)", min_value=0.1, max_value=500.0, value=7.0, step=0.1)
        dataset_size_gb = st.number_input("📦 Dataset Size (GB)", min_value=1.0, max_value=10000.0, value=200.0, step=1.0)
        epochs = st.number_input("🔁 Epochs", min_value=1, max_value=100, value=3, step=1)

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "cloud_provider": cloud_provider,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "training_hours": training_hours,
        "model_parameters_billion": model_parameters_billion,
        "dataset_size_gb": dataset_size_gb,
        "epochs": epochs,
    }


def render_results(cost):
    """Render the prediction result card with cost, category badge, and a
    short business interpretation to guide decision-making."""
    category, badge_class = categorize_cost(cost)
    interpretation = get_business_interpretation(category)

    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Training Cost</div>
            <div class="result-cost">${cost:,.2f}</div>
            <div class="badge {badge_class}">{category} Cost</div>
            <div style="margin-top: 1rem; color: #c9c9dc; font-size: 0.95rem;">{interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insights(insights):
    """Render the dynamic insights section."""
    st.markdown('<div class="section-header">🔍 AI-Generated Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_feature_importance(model, feature_columns):
    """Render the feature importance chart section."""
    st.markdown('<div class="section-header">📈 Model Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    fig = plot_feature_importance(model, feature_columns)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model type.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard_stats():
    """Render premium metric cards summarizing session prediction statistics."""
    history = st.session_state.get("prediction_history", [])

    total_predictions = len(history)
    if total_predictions > 0:
        costs = [row["Predicted Cost"] for row in history]
        highest_cost = max(costs)
        average_cost = sum(costs) / len(costs)
    else:
        highest_cost = 0.0
        average_cost = 0.0

    st.markdown('<div class="section-header">📊 Session Dashboard</div>', unsafe_allow_html=True)

    stats = [
        (f"{total_predictions}", "Total Predictions"),
        (f"${highest_cost:,.2f}", "Highest Cost Predicted"),
        (f"${average_cost:,.2f}", "Average Predicted Cost"),
    ]

    cols = st.columns(3)
    for col, (value, label) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_history():
    """Render the prediction history table, clear button, and CSV export."""
    history = st.session_state.get("prediction_history", [])

    st.markdown('<div class="section-header">🕘 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if not history:
        st.info("No predictions yet. Run an estimate above to start building your session history.")
    else:
        history_df = pd.DataFrame(history)

        # Format the cost column for display
        display_df = history_df.copy()
        display_df["Predicted Cost"] = display_df["Predicted Cost"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Action buttons: clear history / export to CSV
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Prediction History"):
                st.session_state.prediction_history = []
                st.rerun()
        with col2:
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Prediction History",
                data=csv_data,
                file_name="trainwise_prediction_history.csv",
                mime="text/csv",
            )

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    """Render the footer section with tech stack pills."""
    st.markdown(
        """
        <div class="footer-container">
            <div>Built with</div>
            <div style="margin-top: 0.6rem;">
                <span class="footer-pill">🐍 Python</span>
                <span class="footer-pill">🎈 Streamlit</span>
                <span class="footer-pill">⚡ XGBoost</span>
                <span class="footer-pill">☁️ AWS</span>
            </div>
            <div style="margin-top: 0.8rem;">Built by Atharva | TrainWise AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""
    load_custom_css()
    render_hero()
    render_metrics()

    # Initialize session state for prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # Load model artifacts
    try:
        model, feature_columns = load_model_artifacts()
    except FileNotFoundError:
        st.error("⚠️ Model artifacts not found. Please ensure 'trainwise_model.pkl' and 'feature_columns.pkl' are in the app directory.")
        return

    # Prediction form
    inputs = render_prediction_form()

    # Center the prediction button
    btn_col = st.columns([1, 1, 1])
    with btn_col[1]:
        predict_clicked = st.button("🚀 Estimate Training Cost")

    if predict_clicked:
        # Preprocess inputs and run prediction
        input_df = preprocess_input(
            inputs["cloud_provider"],
            inputs["gpu_type"],
            inputs["gpu_count"],
            inputs["training_hours"],
            inputs["model_parameters_billion"],
            inputs["dataset_size_gb"],
            inputs["epochs"],
            feature_columns,
        )

        cost = predict_cost(model, input_df)
        cost = max(cost, 0.0)  # cost cannot be negative

        # Results
        render_results(cost)

        # Insights
        insights = generate_insights(
            inputs["cloud_provider"],
            inputs["gpu_type"],
            inputs["gpu_count"],
            inputs["training_hours"],
            inputs["model_parameters_billion"],
            inputs["dataset_size_gb"],
            inputs["epochs"],
            cost,
        )
        render_insights(insights)

        # Record this prediction in session history
        st.session_state.prediction_history.append({
            "Cloud Provider": inputs["cloud_provider"],
            "GPU Type": inputs["gpu_type"],
            "GPU Count": inputs["gpu_count"],
            "Predicted Cost": cost,
        })

    # Session dashboard stats (total / highest / average)
    render_dashboard_stats()

    # Prediction history table with clear & export options
    render_prediction_history()

    # Feature importance (always shown)
    render_feature_importance(model, feature_columns)

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
