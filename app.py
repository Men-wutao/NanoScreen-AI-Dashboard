import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

PLOT_CONFIG = {"displayModeBar": False}

def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource
def load_model_payload():
    model_path = MODEL_DIR / "best_cat.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    return joblib.load(model_path)


def transform_input_dataframe(df, payload):
    preprocessor = payload["preprocessor"]
    base_features = payload["base_features"]

    df_use = df[base_features].copy()
    X = preprocessor.transform(df_use)

    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X)


def predict_high_delivery(input_df, payload):
    model = payload["estimator"]
    threshold = float(payload["best_threshold"])

    X = transform_input_dataframe(input_df, payload)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)

    if prob >= 0.90:
        level = "Top-priority"
    elif prob >= 0.75:
        level = "High-potential"
    elif prob >= 0.50:
        level = "Moderate"
    else:
        level = "Not prioritized"

    return prob, pred, threshold, level

def predict_batch_high_delivery(df, payload):
    model = payload["estimator"]
    threshold = float(payload["best_threshold"])

    X = transform_input_dataframe(df, payload)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result_df = df.copy()
    result_df["predicted_high_delivery_probability"] = probs
    result_df["decision_threshold"] = threshold
    result_df["predicted_class"] = [
        "High-delivery" if p == 1 else "Low-delivery" for p in preds
    ]

    def assign_recommendation(prob):
        if prob >= 0.90:
            return "Top-priority"
        elif prob >= 0.75:
            return "High-potential"
        elif prob >= 0.50:
            return "Moderate"
        else:
            return "Not prioritized"

    result_df["recommendation"] = [
        assign_recommendation(prob) for prob in probs
    ]

    result_df = result_df.sort_values(
        "predicted_high_delivery_probability",
        ascending=False
    ).reset_index(drop=True)

    result_df.insert(0, "rank", range(1, len(result_df) + 1))

    return result_df


st.set_page_config(
    page_title="NanoScreen-AI",
    page_icon="🧬",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
        color: #111827;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #e5e7eb;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 10px 10px 0 0;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        border-radius: 24px;
        padding: 34px;
        margin: 28px 0 34px 0;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.1fr 1fr;
        gap: 36px;
        align-items: center;
    }

    .hero-label {
        color: #ef4444;
        font-weight: 800;
        letter-spacing: 0.12em;
        font-size: 0.82rem;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2rem;
        line-height: 1.2;
        margin-bottom: 16px;
        color: #111827;
    }

    .hero-text {
        color: #475569;
        font-size: 1rem;
        line-height: 1.7;
    }

    .hero-tags {
        margin-top: 20px;
    }

    .hero-tags span {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border-radius: 999px;
        padding: 7px 12px;
        margin: 4px 6px 4px 0;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .pipeline-box {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
    }

    .pipe-step {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 13px 16px;
        margin: 8px auto;
        font-weight: 700;
        color: #1e293b;
    }

    .pipe-arrow {
        font-size: 1.3rem;
        color: #ef4444;
        font-weight: 800;
        margin: 2px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


try:
    payload = load_model_payload()
    model_loaded = True
except Exception as e:
    payload = None
    model_loaded = False
    model_error = str(e)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🧬 NanoScreen-AI")
st.sidebar.markdown("High tumor delivery nanoparticle screening")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Prediction",
        "Candidate Screening",
        "Top Candidates",
        "Local Working Range",
        "SHAP Explanation",
        "Model Evaluation"
    ]
)

# ----------------------------
# Page 1: Overview
# ----------------------------
if page == "Overview":
    st.title("NanoScreen-AI Dashboard")
    st.subheader(
        "A screening-oriented machine-learning system for high tumor delivery nanoparticle prioritization"
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="hero-label">NANOSCREEN-AI</div>
                    <h2 class="hero-title">Machine-learning-guided nanoparticle candidate screening</h2>
                    <p class="hero-text">
                        NanoScreen-AI is a screening-oriented dashboard for prioritizing nanoparticle 
                        formulations with high tumor delivery potential. The system integrates dataset curation, 
                        q0.75 high-delivery task construction, CatBoost-based prediction, candidate ranking, 
                        model interpretation, and local working range recommendation.
                    </p>
                    <div class="hero-tags">
                        <span>CatBoost</span>
                        <span>High-delivery classification</span>
                        <span>Candidate ranking</span>
                        <span>Batch screening</span>
                        <span>Model interpretation</span>
                    </div>
                </div>
                <div class="pipeline-box">
                    <div class="pipe-step">Nano-Tumor Dataset</div>
                    <div class="pipe-arrow">↓</div>
                    <div class="pipe-step">q0.75 Binary Task</div>
                    <div class="pipe-arrow">↓</div>
                    <div class="pipe-step">CatBoost Screening Model</div>
                    <div class="pipe-arrow">↓</div>
                    <div class="pipe-step">Top-ranked Candidates</div>
                    <div class="pipe-arrow">↓</div>
                    <div class="pipe-step">Local Working Range</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "Model-prioritized candidates should be interpreted as hypotheses for experimental validation, "
        "not as experimentally validated optimal formulations."
    )

    st.divider()
    st.header("Study framework")

    framework_path = BASE_DIR / "figures" / "study_framework.svg"

    if framework_path.exists():
        st.image(str(framework_path), use_container_width=True)
    else:
        st.warning("未找到 figures/study_framework.svg，请确认流程图文件是否已经放入 figures 文件夹。")

    st.divider()

    # ----------------------------
    # Core metrics
    # ----------------------------
    metrics_path = RESULTS_DIR / "best_cat_metrics_test.csv"

    if metrics_path.exists():
        metrics_df_overview = read_csv_safely(metrics_path)
        metric_row = metrics_df_overview.iloc[0].to_dict()

        pr_auc_value = float(metric_row.get("PR-AUC", 0.8447))
        roc_auc_value = float(metric_row.get("ROC-AUC", 0.9182))
        precision_value = float(metric_row.get("Precision", 0.6136))
        recall_value = float(metric_row.get("Recall", 0.9000))
        f1_value = float(metric_row.get("F1", 0.7297))
    else:
        pr_auc_value = 0.8447
        roc_auc_value = 0.9182
        precision_value = 0.6136
        recall_value = 0.9000
        f1_value = 0.7297

    st.header("Core results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analytical records", "534")
    c2.metric("Training samples", "429")
    c3.metric("Independent test samples", "105")
    c4.metric("High-delivery cutoff", "1.7728 %ID")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Primary model", "CatBoost")
    c6.metric("Test PR-AUC", f"{pr_auc_value:.4f}")
    c7.metric("Test ROC-AUC", f"{roc_auc_value:.4f}")
    c8.metric("Test F1", f"{f1_value:.4f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Precision", f"{precision_value:.4f}")
    c10.metric("Recall", f"{recall_value:.4f}")
    c11.metric("Generated candidates", "50,000")
    c12.metric("Reported shortlist", "Top 200")

    st.divider()

    # ----------------------------
    # Workflow
    # ----------------------------
    st.header("Workflow")

    w1, w2, w3, w4, w5 = st.columns(5)

    with w1:
        st.markdown(
            """
            **1. Dataset**

            534 analytical records with nanoparticle, tumor model, and delivery-related variables.
            """
        )

    with w2:
        st.markdown(
            """
            **2. Task definition**

            DETumor at 24 h was converted into a q0.75 high-delivery classification task.
            """
        )

    with w3:
        st.markdown(
            """
            **3. Model training**

            CatBoost was retained as the primary screening model after PR-AUC-centered evaluation.
            """
        )

    with w4:
        st.markdown(
            """
            **4. Candidate screening**

            Virtual candidates are scored, ranked, and filtered for high-delivery potential.
            """
        )

    with w5:
        st.markdown(
            """
            **5. Interpretation**

            Feature importance and local working ranges support mechanistic interpretation.
            """
        )

    st.divider()

    # ----------------------------
    # Dashboard modules
    # ----------------------------
    st.header("Dashboard modules")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            """
            ### Model Prediction

            Input a single nanoparticle formulation and obtain:

            - high-delivery probability
            - predicted class
            - decision threshold
            - recommendation level
            """
        )

    with m2:
        st.markdown(
            """
            ### Candidate Screening

            Upload CSV or Excel files for batch screening:

            - batch probability prediction
            - ranked candidate list
            - recommendation distribution
            - downloadable results
            """
        )

    with m3:
        st.markdown(
            """
            ### Top Candidates

            Explore model-prioritized candidates:

            - paper Top 10
            - paper Top 200
            - generated Top 200
            - candidate formulation details
            """
        )

    m4, m5, m6 = st.columns(3)

    with m4:
        st.markdown(
            """
            ### Local Working Range

            Review suggested local parameter windows for:

            - Size
            - Zeta Potential
            - Admin dose
            - Breast-specific recommendations
            """
        )

    with m5:
        st.markdown(
            """
            ### SHAP Explanation

            Interpret model behavior using:

            - raw feature importance
            - original-feature-level importance
            - preprocessed one-hot feature importance
            """
        )

    with m6:
        st.markdown(
            """
            ### Model Evaluation

            Inspect independent test-set performance:

            - ROC curve
            - PR curve
            - score distribution
            - confusion matrix
            - ranking metrics
            """
        )

    st.divider()

    # ----------------------------
    # Usage note
    # ----------------------------
    st.info(
        "Recommended use: first inspect the Overview and Model Evaluation pages, then use Model Prediction "
        "for individual formulations and Candidate Screening for batch prioritization. Top Candidates and "
        "Local Working Range provide supporting evidence for formulation design and experimental planning."
    )



# ----------------------------
# Page 2: Model Prediction
# ----------------------------
elif page == "Model Prediction":
    st.title("Model Prediction")
    st.markdown(
    """
    该页面用于输入单个纳米颗粒候选配方，并基于最终 CatBoost 模型预测其属于 
    **high-delivery candidate** 的概率。
    """
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        type_value = st.selectbox("Type", ["ONM", "INM", "Hybrid"])
        ts_value = st.selectbox("Targeting strategy (TS)", ["Passive", "Active"])
        tm_value = st.selectbox(
            "Tumor model (TM)",
            [
                "Xenograft Heterotopic",
                "Xenograft Orthotopic",
                "Allograft Heterotopic",
                "Allograft Orthotopic"
            ]
        )
        size_value = st.slider("Size, log10(nm)", 0.4314, 3.0792, 2.0000, 0.0001)

    with col2:
        mat_value = st.selectbox(
            "Core material (MAT)",
            [
                "Gold",
                "Polymeric",
                "Other OM",
                "Other IM",
                "Liposome",
                "Silica",
                "Dendrimer",
                "Hydrogel",
                "Hybrid",
                "Iron",
                "Oxide"
            ]
        )
        ct_value = st.selectbox(
            "Cancer type (CT)",
            [
                "Breast",
                "Liver",
                "Lung",
                "Cervix",
                "Colon",
                "Brain",
                "Glioma",
                "Ovary",
                "Pancreas",
                "Prostate",
                "Sarcoma",
                "Skin",
                "Other"
            ]
        )
        shape_value = st.selectbox("Shape", ["Spherical", "Rod", "Plate", "Other"])
        zeta_value = st.slider("Zeta Potential (mV)", -65.12, 71.30, 0.0, 0.01)

    admin_value = st.slider("Admin dose (mg/kg)", 0.0, 1292.0, 5.0, 0.1)

    if st.button("Predict", type="primary"):
        st.subheader("Prediction Results")

        if not model_loaded:
            st.error(f"模型加载失败：{model_error}")
        else:
            input_df = pd.DataFrame(
                [{
                    "Type": type_value,
                    "MAT": mat_value,
                    "TS": ts_value,
                    "CT": ct_value,
                    "TM": tm_value,
                    "Shape": shape_value,
                    "Size": size_value,
                    "Zeta Potential": zeta_value,
                    "Admin": admin_value
                }]
            )

            try:
                prob, pred, threshold, level = predict_high_delivery(input_df, payload)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted high-delivery probability", f"{prob:.4f}")
                col2.metric("Decision threshold", f"{threshold:.3f}")
                col3.metric(
                    "Predicted class",
                    "High-delivery" if pred == 1 else "Low-delivery"
                )
                col4.metric("Recommendation", level)

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Variable": [
                                "Type", "MAT", "TS", "CT", "TM", "Shape",
                                "Size", "Zeta Potential", "Admin"
                            ],
                            "Input value": [
                                type_value, mat_value, ts_value, ct_value, tm_value,
                                shape_value, size_value, zeta_value, admin_value
                            ]
                        }
                    ),
                    use_container_width=True
                )

                if prob >= 0.90:
                    st.success("This formulation is a top-priority candidate for experimental follow-up.")
                elif prob >= 0.75:
                    st.info("This formulation has high screening potential.")
                elif prob >= 0.50:
                    st.warning("This formulation has moderate potential and may require context-specific review.")
                else:
                    st.error("This formulation is not prioritized by the current screening model.")

            except Exception as e:
                st.error("预测失败，请检查输入变量是否与训练模型一致。")
                st.exception(e)

# ----------------------------
# Page 3: Candidate Screening
# ----------------------------
elif page == "Candidate Screening":
    st.title("Candidate Screening")
    st.markdown(
        """
        该页面用于上传 CSV 或 Excel 文件，对多个候选纳米颗粒配方进行批量预测和排序。
        """
    )

    required_cols = [
        "Type", "MAT", "TS", "CT", "TM", "Shape",
        "Size", "Zeta Potential", "Admin"
    ]

    st.code(
        "Required columns: Type, MAT, TS, CT, TM, Shape, Size, Zeta Potential, Admin",
        language="text"
    )

    uploaded_file = st.file_uploader("Upload candidate file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Uploaded data preview")
        st.dataframe(df.head(), use_container_width=True)

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        elif not model_loaded:
            st.error(f"模型加载失败：{model_error}")
        else:
            if st.button("Run Batch Screening", type="primary"):
                try:
                    result_df = predict_batch_high_delivery(
                        df[required_cols],
                        payload
                    )

                    st.subheader("Screening Summary")

                    total_n = len(result_df)
                    high_n = int((result_df["predicted_class"] == "High-delivery").sum())
                    low_n = int((result_df["predicted_class"] == "Low-delivery").sum())
                    top_priority_n = int((result_df["recommendation"] == "Top-priority").sum())
                    high_potential_n = int((result_df["recommendation"] == "High-potential").sum())
                    max_prob = result_df["predicted_high_delivery_probability"].max()
                    mean_prob = result_df["predicted_high_delivery_probability"].mean()

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total candidates", total_n)
                    c2.metric("Predicted high-delivery", high_n)
                    c3.metric("Predicted low-delivery", low_n)
                    c4.metric("Top-priority", top_priority_n)

                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("High-potential", high_potential_n)
                    c6.metric("Max probability", f"{max_prob:.4f}")
                    c7.metric("Mean probability", f"{mean_prob:.4f}")
                    c8.metric("Decision threshold", f"{float(payload['best_threshold']):.3f}")

                    st.divider()

                    st.subheader("Recommendation distribution")

                    recommendation_count = (
                        result_df["recommendation"]
                        .value_counts()
                        .reset_index()
                    )
                    recommendation_count.columns = ["Recommendation", "Count"]

                    st.bar_chart(
                        recommendation_count.set_index("Recommendation")
                    )

                    st.divider()

                    st.subheader("Ranked screening results")
                    st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="Download screening results",
                        data=csv,
                        file_name="nanoscreen_batch_screening_results.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error("批量预测失败，请检查上传文件中的类别名称和数值范围是否与训练数据一致。")
                    st.exception(e)

# ----------------------------
# Page 4: Top Candidates
# ----------------------------
elif page == "Top Candidates":
    st.title("Top-ranked Candidate Formulations")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generated candidates", "50,000")
    col2.metric("Unique candidates", "49,969")
    col3.metric("After filtering", "46,136")
    col4.metric("Reported shortlist", "Top 200")

    st.divider()

    top10_path = DATA_DIR / "paper_candidate_table_top10.csv"
    top200_path = DATA_DIR / "paper_candidate_table_top200.csv"
    generated_top200_path = DATA_DIR / "generated_top200.csv"
    all_scored_path = DATA_DIR / "generated_candidates_scored.csv"

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Paper Top 10", "Paper Top 200", "Generated Top 200", "All scored candidates"]
    )

    with tab1:
        st.subheader("Paper candidate table: Top 10")
        if top10_path.exists():
            top10_df = read_csv_safely(top10_path)
            st.dataframe(top10_df, use_container_width=True, height=420)
        else:
            st.error("未找到 data/paper_candidate_table_top10.csv")

    with tab2:
        st.subheader("Paper candidate table: Top 200")
        if top200_path.exists():
            top200_df = read_csv_safely(top200_path)
            st.dataframe(top200_df, use_container_width=True, height=520)
        else:
            st.error("未找到 data/paper_candidate_table_top200.csv")

    with tab3:
        st.subheader("Generated Top 200 candidates")
        if generated_top200_path.exists():
            generated_top200_df = read_csv_safely(generated_top200_path)
            st.dataframe(generated_top200_df, use_container_width=True, height=520)
        else:
            st.error("未找到 data/generated_top200.csv")

    with tab4:
        st.subheader("All generated candidates with model scores")
        if all_scored_path.exists():
            all_scored_df = read_csv_safely(all_scored_path)
            st.dataframe(all_scored_df, use_container_width=True, height=520)
        else:
            st.error("未找到 data/generated_candidates_scored.csv")

    st.warning(
        "Top candidates are model-prioritized hypotheses for future experimental validation, "
        "not experimentally validated optimal formulations."
    )


# ----------------------------
# Page 5: Local Working Range
# ----------------------------
elif page == "Local Working Range":
    st.title("Local Working Range")

    st.markdown(
        """
        This page displays model-prioritized local working ranges for key continuous variables,
        including **Size**, **Zeta Potential**, and **Admin**.
        """
    )

    pretty_path = DATA_DIR / "range_recommendation_pretty.csv"
    detail_path = DATA_DIR / "range_recommendation_detail.csv"
    breast_path = DATA_DIR / "range_recommendation_Breast_pretty.csv"

    tab1, tab2, tab3 = st.tabs(
        ["Pretty Table", "Detailed Table", "Breast-specific"]
    )

    with tab1:
        st.subheader("Local working range: pretty table")
        if pretty_path.exists():
            pretty_df = read_csv_safely(pretty_path)
            st.dataframe(pretty_df, use_container_width=True, height=260)
        else:
            st.error("未找到 data/range_recommendation_pretty.csv")

    with tab2:
        st.subheader("Local working range: detailed table")
        if detail_path.exists():
            detail_df = read_csv_safely(detail_path)
            st.dataframe(detail_df, use_container_width=True, height=520)
        else:
            st.error("未找到 data/range_recommendation_detail.csv")

    with tab3:
        st.subheader("Breast-specific local working range")
        if breast_path.exists():
            breast_df = read_csv_safely(breast_path)
            st.dataframe(breast_df, use_container_width=True, height=420)
        else:
            st.warning("未找到 data/range_recommendation_Breast_pretty.csv")

    st.info(
        "Local working ranges should be interpreted as model-prioritized parameter windows, "
        "not experimentally validated optimal conditions."
    )


# ----------------------------
# Page 6: SHAP Explanation
# ----------------------------
elif page == "SHAP Explanation":
    st.title("Model Interpretation")

    st.markdown(
        """
        This page displays feature-importance results from the retained CatBoost model.
        Preprocessed one-hot features are additionally aggregated back to the original predictor level
        for clearer interpretation.
        """
    )

    fi_path = RESULTS_DIR / "best_cat_feature_importances.csv"

    if not fi_path.exists():
        st.error("未找到 results/best_cat_feature_importances.csv")
    else:
        fi_df = read_csv_safely(fi_path)

        st.subheader("Raw feature-importance table")
        st.dataframe(fi_df, use_container_width=True)

        st.divider()

        feature_col = "feature"
        importance_col = "importance"

        if feature_col not in fi_df.columns or importance_col not in fi_df.columns:
            st.warning("无法识别 feature / importance 列，请检查 CSV 表头。")
            st.write("Current columns:", list(fi_df.columns))
        else:
            fi_plot_df = fi_df[[feature_col, importance_col]].copy()
            fi_plot_df[importance_col] = pd.to_numeric(
                fi_plot_df[importance_col],
                errors="coerce"
            )
            fi_plot_df = fi_plot_df.dropna(subset=[importance_col])

            def map_to_original_feature(name):
                name = str(name)

                if "Admin" in name:
                    return "Admin"
                elif "Size" in name:
                    return "Size"
                elif "Zeta" in name:
                    return "Zeta Potential"
                elif "TS" in name:
                    return "TS"
                elif "MAT" in name:
                    return "MAT"
                elif "CT" in name:
                    return "CT"
                elif "TM" in name:
                    return "TM"
                elif "Shape" in name:
                    return "Shape"
                elif "Type" in name:
                    return "Type"
                else:
                    return "Other"

            fi_plot_df["Original feature"] = fi_plot_df[feature_col].apply(
                map_to_original_feature
            )

            grouped_fi = (
                fi_plot_df
                .groupby("Original feature", as_index=False)[importance_col]
                .sum()
                .sort_values(importance_col, ascending=False)
                .reset_index(drop=True)
            )

            st.subheader("Original-feature-level importance")
            st.bar_chart(
                grouped_fi.set_index("Original feature")[importance_col]
            )
            st.dataframe(grouped_fi, use_container_width=True)

            st.divider()

            top_n = st.slider(
                "Number of preprocessed features to display",
                min_value=5,
                max_value=min(50, len(fi_plot_df)),
                value=min(20, len(fi_plot_df)),
                step=1
            )

            top_fi_df = (
                fi_plot_df
                .sort_values(importance_col, ascending=False)
                .head(top_n)
            )

            st.subheader(f"Top {top_n} preprocessed feature importances")
            st.bar_chart(
                top_fi_df.set_index(feature_col)[importance_col]
            )
            st.dataframe(top_fi_df, use_container_width=True)

            st.info(
                "The original-feature-level chart is easier to interpret for reporting. "
                "The preprocessed-feature table shows the detailed one-hot encoded model inputs."
            )


# ----------------------------
# Page 7: Model Evaluation
# ----------------------------
elif page == "Model Evaluation":
    st.title("Model Evaluation")

    st.markdown(
        """
        This page summarizes the independent test-set performance of the retained CatBoost model,
        including threshold-based metrics, ranking-oriented enrichment metrics, ROC/PR curves,
        score distribution, and confusion matrix.
        """
    )

    metrics_path = RESULTS_DIR / "best_cat_metrics_test.csv"
    pred_path = RESULTS_DIR / "best_cat_test_predictions.csv"

    if not metrics_path.exists():
        st.error("未找到 results/best_cat_metrics_test.csv")
    elif not pred_path.exists():
        st.error("未找到 results/best_cat_test_predictions.csv")
    else:
        metrics_df = read_csv_safely(metrics_path)
        pred_df = read_csv_safely(pred_path)

        st.subheader("Independent test-set metrics")

        st.dataframe(metrics_df, use_container_width=True, height=120)

        metric_row = metrics_df.iloc[0].to_dict()

        pr_auc_value = metric_row.get("PR-AUC", np.nan)
        roc_auc_value = metric_row.get("ROC-AUC", np.nan)
        precision_value = metric_row.get("Precision", np.nan)
        recall_value = metric_row.get("Recall", np.nan)
        f1_value = metric_row.get("F1", np.nan)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("PR-AUC", f"{float(pr_auc_value):.4f}")
        col2.metric("ROC-AUC", f"{float(roc_auc_value):.4f}")
        col3.metric("Precision", f"{float(precision_value):.4f}")
        col4.metric("Recall", f"{float(recall_value):.4f}")
        col5.metric("F1", f"{float(f1_value):.4f}")

        st.divider()

        st.subheader("Ranking-oriented screening metrics")

        ranking_cols = [
            col for col in metrics_df.columns
            if ("Precision@" in col)
            or ("Recall@" in col)
            or ("EF@" in col)
            or ("EF5%" in col)
            or ("EF10%" in col)
        ]

        if ranking_cols:
            ranking_df = metrics_df[ranking_cols].T.reset_index()
            ranking_df.columns = ["Metric", "Value"]
            st.dataframe(ranking_df, use_container_width=True, height=240)
        else:
            st.warning("未在 best_cat_metrics_test.csv 中找到 Precision@K / Recall@K / EF 指标。")

        st.divider()

        required_pred_cols = ["y_true_test", "y_prob_test", "y_pred_test"]
        missing_pred_cols = [col for col in required_pred_cols if col not in pred_df.columns]

        if missing_pred_cols:
            st.error(f"Prediction file missing required columns: {missing_pred_cols}")
        else:
            pred_df["y_true_test"] = pd.to_numeric(pred_df["y_true_test"], errors="coerce")
            pred_df["y_prob_test"] = pd.to_numeric(pred_df["y_prob_test"], errors="coerce")
            pred_df["y_pred_test"] = pd.to_numeric(pred_df["y_pred_test"], errors="coerce")

            pred_df = pred_df.dropna(subset=required_pred_cols)

            y_true = pred_df["y_true_test"].astype(int).to_numpy()
            y_prob = pred_df["y_prob_test"].astype(float).to_numpy()
            y_pred = pred_df["y_pred_test"].astype(int).to_numpy()

            tab1, tab2, tab3, tab4 = st.tabs(
                ["ROC Curve", "PR Curve", "Score Distribution", "Confusion Matrix"]
            )

            with tab1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc_calc = auc(fpr, tpr)

                roc_fig = go.Figure()
                roc_fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"ROC curve, AUC={roc_auc_calc:.4f}"
                    )
                )
                roc_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random baseline",
                        line=dict(dash="dash")
                    )
                )
                roc_fig.update_layout(
                    title="Receiver Operating Characteristic Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=420,
                    margin=dict(l=40, r=30, t=60, b=40),
                    template="plotly_white"
                )
                st.plotly_chart(roc_fig, use_container_width=True, config=PLOT_CONFIG)

            with tab2:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                pr_auc_calc = auc(recall_curve, precision_curve)

                pr_fig = go.Figure()
                pr_fig.add_trace(
                    go.Scatter(
                        x=recall_curve,
                        y=precision_curve,
                        mode="lines",
                        name=f"PR curve, AUC={pr_auc_calc:.4f}"
                    )
                )
                pr_fig.update_layout(
                    title="Precision–Recall Curve",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=420,
                    margin=dict(l=40, r=30, t=60, b=40),
                    template="plotly_white"
                )
                st.plotly_chart(pr_fig, use_container_width=True, config=PLOT_CONFIG)

            with tab3:
                score_fig = px.histogram(
                    pred_df,
                    x="y_prob_test",
                    color=pred_df["y_true_test"].astype(str),
                    nbins=30,
                    labels={
                        "y_prob_test": "Predicted high-delivery probability",
                        "color": "True class"
                    },
                    title="Predicted-score distribution by true class"
                )
                score_fig.update_layout(
                    height=420,
                    margin=dict(l=40, r=30, t=60, b=40),
                    template="plotly_white",
                    bargap=0.08
                )
                st.plotly_chart(score_fig, use_container_width=True, config=PLOT_CONFIG)

            with tab4:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                cm_df = pd.DataFrame(
                    cm,
                    index=["True low-delivery", "True high-delivery"],
                    columns=["Predicted low-delivery", "Predicted high-delivery"]
                )

                st.dataframe(cm_df, use_container_width=True, height=120)

                cm_fig = px.imshow(
                    cm_df,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion matrix"
                )
                cm_fig.update_layout(
                    height=340,
                    margin=dict(l=40, r=30, t=50, b=40),
                    template="plotly_white"
                )
                st.plotly_chart(cm_fig, use_container_width=True, config=PLOT_CONFIG)

            st.success(
                "The retained CatBoost model achieved strong independent test-set discrimination "
                "and screening-oriented enrichment performance, supporting its use for prioritizing "
                "high-delivery nanoparticle candidates before experimental validation."
            )

            st.divider()

            with st.expander("View and download test prediction table", expanded=False):
                st.dataframe(pred_df, use_container_width=True, height=320)

                csv = pred_df.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    label="Download test prediction table",
                    data=csv,
                    file_name="best_cat_test_predictions_view.csv",
                    mime="text/csv"
                )