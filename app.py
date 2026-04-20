import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, f1_score,
                              roc_auc_score, roc_curve, r2_score,
                              mean_squared_error)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VentureGuard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 6px 0;
}

/* Main background */
.main { background: #fafafa; }

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e8e8e8;
    border-left: 4px solid #8B0000;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f0f0f;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #8B0000;
    border-bottom: 2px solid #8B0000;
    padding-bottom: 6px;
    margin: 28px 0 18px 0;
}

/* Page title */
.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #0f0f0f;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 1rem;
    color: #666;
    margin-bottom: 28px;
}

/* Result badge */
.badge-acquired {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 6px 18px;
    font-weight: 700;
    font-size: 1rem;
    display: inline-block;
}
.badge-closed {
    background: #ffebee;
    color: #c62828;
    border: 1px solid #ef9a9a;
    border-radius: 20px;
    padding: 6px 18px;
    font-weight: 700;
    font-size: 1rem;
    display: inline-block;
}

/* Accuracy pill */
.acc-pill {
    background: #8B0000;
    color: white;
    border-radius: 6px;
    padding: 4px 14px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}

div[data-testid="stButton"] button {
    background: #8B0000;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
div[data-testid="stButton"] button:hover {
    background: #6a0000;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading & caching ────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Drop columns that are mostly empty or irrelevant
    drop_cols = ['Unnamed: 0', 'name', 'permalink', 'homepage_url',
                 'twitter_username', 'logo_url', 'short_description',
                 'description', 'founded_at', 'closed_at',
                 'first_funding_at', 'last_funding_at',
                 'age_first_milestone_year', 'age_last_milestone_year']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Normalize status labels to avoid case/spacing issues.
    if 'status' in df.columns:
        df['status'] = df['status'].astype(str).str.strip().str.lower()

    # Parse numeric-looking columns that may contain commas or currency symbols.
    numeric_cols = [
        'funding_total_usd', 'funding_rounds', 'milestones', 'relationships',
        'avg_participants', 'age_first_funding_year', 'age_last_funding_year',
        'has_VC', 'has_angel', 'is_software', 'is_web', 'is_mobile'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[$,]', '', regex=True).str.strip(),
                errors='coerce'
            )

    # Keep only acquired / closed
    if 'status' in df.columns:
        df = df[df['status'].isin(['acquired', 'closed'])].copy()

    # Drop rows with nulls in key columns
    key_cols = ['funding_total_usd', 'funding_rounds', 'milestones',
                'relationships', 'status']
    df.dropna(subset=[c for c in key_cols if c in df.columns], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


@st.cache_data
def prepare_ml(df):
    feature_cols = [
        'funding_rounds', 'funding_total_usd', 'milestones',
        'relationships', 'avg_participants',
        'age_first_funding_year', 'age_last_funding_year',
        'has_VC', 'has_angel', 'is_software', 'is_web', 'is_mobile'
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    y = (df['status'] == 'acquired').astype(int)   # 1=acquired, 0=closed
    return X, y, available


def get_model_catalog():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel='rbf', random_state=42, probability=True)
    }


@st.cache_resource
def train_prediction_model(X_train, y_train, model_name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model_catalog = get_model_catalog()
    clf = model_catalog[model_name]
    clf.fit(X_scaled, y_train)
    return scaler, clf


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 VentureGuard")
    st.markdown("*Predictive Analytics App*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home",
        "📊  EDA & Visualizations",
        "🤖  Classification",
        "🧩  Clustering",
        "📈  Regression",
        "🔮  Predict a Startup"
    ])
    st.markdown("---")
    st.markdown("**Product:** VentureGuard")
    st.markdown("**Dataset:** Startup Success Prediction")
    st.markdown("**Version:** v1.0")
    st.markdown("---")
    st.markdown("**About VentureGuard**")
    st.markdown(
        "VentureGuard helps founders and investors assess startup risk using "
        "data-driven insights, model comparisons, and an early warning prediction."
    )

# ── File uploader (global) ────────────────────────────────────────────────────
st.markdown('<div class="page-title">Upload Dataset to Begin</div>',
            unsafe_allow_html=True)
uploaded = st.file_uploader("Upload startup data.csv", type=["csv"],
                             label_visibility="collapsed")

if uploaded is None:
    st.info("👆 Please upload **startup data.csv** to continue.")
    st.stop()

df = load_and_clean(uploaded)

if 'status' not in df.columns:
    st.error("Dataset error: Missing required column 'status'.")
    st.stop()

if df.empty:
    st.error("Dataset error: No rows left after cleaning. Please upload a valid dataset.")
    st.stop()

if df['status'].nunique() < 2:
    st.error("Dataset error: Need both 'acquired' and 'closed' rows for classification.")
    st.stop()

X, y, feature_cols = prepare_ml(df)

if len(feature_cols) == 0:
    st.error("Dataset error: No usable ML feature columns were found.")
    st.stop()

if len(df) < 30:
    st.warning("Small dataset warning: Results may be unstable with fewer than 30 rows.")

class_counts = y.value_counts(normalize=True)
minority_ratio = class_counts.min()
if minority_ratio < 0.10:
    st.warning("Class imbalance warning: Minority class is under 10%. Consider using F1/ROC-AUC over accuracy.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown('<div class="page-title">VentureGuard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">VentureGuard: An Early Warning System for Founders and Investors</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Startups</div>
            <div class="value">{len(df):,}</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Features</div>
            <div class="value">{df.shape[1]}</div></div>""",
            unsafe_allow_html=True)
    with c3:
        acq = (df['status'] == 'acquired').sum()
        st.markdown(f"""<div class="metric-card">
            <div class="label">Acquired</div>
            <div class="value">{acq}</div></div>""",
            unsafe_allow_html=True)
    with c4:
        clo = (df['status'] == 'closed').sum()
        st.markdown(f"""<div class="metric-card">
            <div class="label">Closed</div>
            <div class="value">{clo}</div></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-header">PROJECT OVERVIEW</div>',
                unsafe_allow_html=True)
    st.markdown("""
This application uses real startup data sourced from **Crunchbase** to analyse
patterns behind startup failure and success. Across the pipeline we load, clean,
visualise, classify, regress, and predict — building toward a complete early
warning system.

**Target variable:** Startup status — *Acquired* or *Closed*

**Key question:** Can we predict whether a startup will succeed or fail
based on its funding structure, team relationships, and technology category?
    """)

    st.markdown('<div class="section-header">DATASET PREVIEW</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-header">PIPELINE SUMMARY</div>',
                unsafe_allow_html=True)
    steps = {
        "Step 1": "Data Loading, Inspection & Cleaning",
        "Step 2": "EDA & Visualizations (Matplotlib / Seaborn)",
        "Step 3": "Classification — KNN, Decision Tree, Naive Bayes, SVM",
        "Step 4": "Regression — Linear & Polynomial",
        "Step 5": "Clustering — K-Means with cluster profiling",
        "Step 6": "Interactive Startup Prediction",
    }
    for k, v in steps.items():
        st.markdown(f"**{k}** &nbsp;→&nbsp; {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA & Visualizations":
    st.markdown('<div class="page-title">EDA & Visualizations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Exploratory analysis of the startup dataset</div>',
                unsafe_allow_html=True)

    chart = st.selectbox("Select chart", [
        "Status Distribution",
        "Funding Distribution",
        "Top 8 Categories by Count",
        "Avg Funding by Category",
        "Funding Rounds vs Funding Total",
        "Correlation Heatmap"
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    if chart == "Status Distribution":
        counts = df['status'].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=['#8B0000', '#cccccc'], edgecolor='white', width=0.5)
        ax.set_title("Startup Status Distribution", fontweight='bold', fontsize=13)
        ax.set_ylabel("Count")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                    str(int(b.get_height())), ha='center', fontweight='bold')

    elif chart == "Funding Distribution":
        data = np.log1p(df['funding_total_usd'])
        ax.hist(data, bins=40, color='#8B0000', edgecolor='white', alpha=0.85)
        ax.set_title("Funding Distribution (log scale)", fontweight='bold', fontsize=13)
        ax.set_xlabel("log(1 + Funding USD)")
        ax.set_ylabel("Count")

    elif chart == "Top 8 Categories by Count":
        if 'category_code' in df.columns:
            top = df['category_code'].value_counts().head(8)
            bars = ax.barh(top.index[::-1], top.values[::-1],
                           color='#8B0000', edgecolor='white')
            ax.set_title("Top 8 Categories by Startup Count",
                         fontweight='bold', fontsize=13)
            ax.set_xlabel("Count")

    elif chart == "Avg Funding by Category":
        if 'category_code' in df.columns:
            avg = (df.groupby('category_code')['funding_total_usd']
                   .mean().sort_values(ascending=False).head(8))
            bars = ax.bar(avg.index, avg.values / 1e6,
                          color='#8B0000', edgecolor='white')
            ax.set_title("Avg Funding by Category (Top 8, $M)",
                         fontweight='bold', fontsize=13)
            ax.set_ylabel("Avg Funding ($M)")
            plt.xticks(rotation=30, ha='right')

    elif chart == "Funding Rounds vs Funding Total":
        colors = ['#8B0000' if s == 'acquired' else '#aaaaaa'
                  for s in df['status']]
        ax.scatter(df['funding_rounds'],
                   np.log1p(df['funding_total_usd']),
                   c=colors, alpha=0.5, s=20)
        ax.set_title("Funding Rounds vs log(Funding Total)",
                     fontweight='bold', fontsize=13)
        ax.set_xlabel("Funding Rounds")
        ax.set_ylabel("log(Funding Total USD)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#8B0000', label='Acquired'),
                            Patch(color='#aaaaaa', label='Closed')])

    elif chart == "Correlation Heatmap":
        num_cols = [c for c in feature_cols if c in df.columns]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#fafafa')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Reds',
                    linewidths=0.4, ax=ax, annot_kws={'size': 8})
        ax.set_title("Feature Correlation Heatmap",
                     fontweight='bold', fontsize=13)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">BASIC STATISTICS</div>',
                unsafe_allow_html=True)
    st.dataframe(df[feature_cols].describe().round(3), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Classification":
    st.markdown('<div class="page-title">Classification Models</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">KNN · Decision Tree · Naive Bayes · SVM — predicting acquired vs closed</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">HOW TO READ THESE RESULTS</div>',
                unsafe_allow_html=True)
    st.markdown("""
- **Accuracy**: overall correct predictions.
- **F1 score**: better when classes are imbalanced.
- **ROC-AUC**: ability to separate acquired vs closed (closer to 1 is better).
- **CV mean ± std**: reliability across 5 different data splits (higher mean, lower std is ideal).
    """)

    test_size = st.slider("Test set size", 0.15, 0.35, 0.20, 0.05)

    if st.button("▶  Run All 4 Models"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = get_model_catalog()

        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            results[name] = {
                'model' : model,
                'acc'   : accuracy_score(y_test, y_pred),
                'f1'    : f1_score(y_test, y_pred),
                'auc'   : roc_auc_score(y_test, y_proba),
                'proba' : y_proba,
                'cm'    : confusion_matrix(y_test, y_pred),
                'report': classification_report(y_test, y_pred,
                                                target_names=['Closed','Acquired'])
            }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {}
        for name, model in models.items():
            cv_estimator = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            cv_scores = cross_validate(
                cv_estimator,
                X,
                y,
                cv=cv,
                scoring=['accuracy', 'f1', 'roc_auc'],
                n_jobs=None
            )
            cv_results[name] = {
                'acc_mean': cv_scores['test_accuracy'].mean(),
                'acc_std': cv_scores['test_accuracy'].std(),
                'f1_mean': cv_scores['test_f1'].mean(),
                'f1_std': cv_scores['test_f1'].std(),
                'auc_mean': cv_scores['test_roc_auc'].mean(),
                'auc_std': cv_scores['test_roc_auc'].std(),
            }

        # Accuracy comparison
        st.markdown('<div class="section-header">ACCURACY COMPARISON</div>',
                    unsafe_allow_html=True)
        cols = st.columns(4)
        best = max(results, key=lambda k: results[k]['acc'])
        for i, (name, res) in enumerate(results.items()):
            with cols[i]:
                border = "border: 2px solid #8B0000;" if name == best else ""
                st.markdown(f"""<div class="metric-card" style="{border}">
                    <div class="label">{name}</div>
                    <div class="value">{res['acc']*100:.1f}%</div>
                    {'<div style="color:#8B0000;font-size:0.75rem;font-weight:700">★ BEST</div>' if name == best else ''}
                    </div>""", unsafe_allow_html=True)

        # Confusion matrices
        st.markdown('<div class="section-header">CONFUSION MATRICES</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.patch.set_facecolor('#fafafa')
        for ax, (name, res) in zip(axes, results.items()):
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Reds',
                        ax=ax, cbar=False,
                        xticklabels=['Closed','Acquired'],
                        yticklabels=['Closed','Acquired'])
            ax.set_title(f"{name}\n{res['acc']*100:.1f}%",
                         fontweight='bold', fontsize=11)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-header">F1 / ROC-AUC METRICS</div>',
                    unsafe_allow_html=True)
        metrics_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': res['acc'],
                'F1 Score': res['f1'],
                'ROC-AUC': res['auc']
            }
            for name, res in results.items()
        ]).set_index('Model')
        st.dataframe(metrics_df.style.format('{:.4f}'), use_container_width=True)

        st.markdown('<div class="section-header">5-FOLD CROSS-VALIDATION (MEAN ± STD)</div>',
                    unsafe_allow_html=True)
        cv_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy (CV)': f"{res['acc_mean']:.4f} ± {res['acc_std']:.4f}",
                'F1 (CV)': f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}",
                'ROC-AUC (CV)': f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f}",
            }
            for name, res in cv_results.items()
        ]).set_index('Model')
        st.dataframe(cv_df, use_container_width=True)

        best_cv = max(cv_results, key=lambda k: cv_results[k]['auc_mean'])
        st.markdown('<div class="section-header">BEST MODEL RECOMMENDATION</div>',
                    unsafe_allow_html=True)
        st.success(
            f"Recommended model: {best_cv} (best 5-fold ROC-AUC = {cv_results[best_cv]['auc_mean']:.4f})."
        )

        report_df = pd.DataFrame([
            {
                'Model': name,
                'Holdout Accuracy': results[name]['acc'],
                'Holdout F1': results[name]['f1'],
                'Holdout ROC-AUC': results[name]['auc'],
                'CV Accuracy Mean': cv_results[name]['acc_mean'],
                'CV Accuracy Std': cv_results[name]['acc_std'],
                'CV F1 Mean': cv_results[name]['f1_mean'],
                'CV F1 Std': cv_results[name]['f1_std'],
                'CV ROC-AUC Mean': cv_results[name]['auc_mean'],
                'CV ROC-AUC Std': cv_results[name]['auc_std'],
            }
            for name in models.keys()
        ])
        st.session_state['classification_summary_csv'] = report_df.to_csv(index=False).encode('utf-8')
        st.session_state['best_cv_model'] = best_cv

        st.markdown('<div class="section-header">ROC CURVES (HOLD-OUT TEST SET)</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res['proba'])
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={res['auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison', fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-header">DECISION TREE FEATURE IMPORTANCE</div>',
                    unsafe_allow_html=True)
        dt_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': results['Decision Tree']['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        st.dataframe(dt_importance, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        top_imp = dt_importance.head(8).iloc[::-1]
        ax.barh(top_imp['Feature'], top_imp['Importance'], color='#8B0000', edgecolor='white')
        ax.set_xlabel('Importance')
        ax.set_title('Top 8 Important Features (Decision Tree)', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Classification report for best model
        st.markdown(f'<div class="section-header">CLASSIFICATION REPORT — {best}</div>',
                    unsafe_allow_html=True)
        st.text(results[best]['report'])

        st.session_state['clf_results'] = results
        st.session_state['clf_scaler']  = scaler
        st.session_state['clf_features'] = feature_cols

    if 'classification_summary_csv' in st.session_state:
        st.markdown('<div class="section-header">DOWNLOAD CLASSIFICATION REPORT</div>',
                    unsafe_allow_html=True)
        st.download_button(
            label='⬇ Download Metrics CSV',
            data=st.session_state['classification_summary_csv'],
            file_name='classification_report_summary.csv',
            mime='text/csv'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧩  Clustering":
    st.markdown('<div class="page-title">Clustering Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">K-Means segmentation of startups by profile similarity</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">CONFIGURATION</div>',
                unsafe_allow_html=True)
    n_clusters = st.slider("Number of clusters (K)", 2, 8, 3, 1)

    if st.button("▶  Run Clustering"):
        X_cluster = X.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # Elbow values for K=2..8
        k_range = list(range(2, 9))
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        st.markdown('<div class="section-header">ELBOW METHOD</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        ax.plot(k_range, inertias, marker='o', color='#8B0000', linewidth=2)
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Curve', fontweight='bold')
        ax.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-header">CLUSTER VISUALIZATION (PCA 2D)</div>',
                    unsafe_allow_html=True)
        pca = PCA(n_components=2, random_state=42)
        points_2d = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1],
                             c=labels, cmap='Reds', alpha=0.75, s=26)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Startup Clusters in PCA Space', fontweight='bold')
        legend = ax.legend(*scatter.legend_elements(), title='Cluster')
        ax.add_artist(legend)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-header">CLUSTER SUMMARY</div>',
                    unsafe_allow_html=True)
        cluster_df = X_cluster.copy()
        cluster_df['Cluster'] = labels
        cluster_df['Status'] = y.values

        summary = cluster_df.groupby('Cluster').agg(
            Count=('Cluster', 'size'),
            Avg_Funding_USD=('funding_total_usd', 'mean') if 'funding_total_usd' in cluster_df.columns else ('Cluster', 'size'),
            Avg_Funding_Rounds=('funding_rounds', 'mean') if 'funding_rounds' in cluster_df.columns else ('Cluster', 'size'),
            Avg_Milestones=('milestones', 'mean') if 'milestones' in cluster_df.columns else ('Cluster', 'size'),
            Acquired_Rate=('Status', 'mean')
        ).reset_index()

        if 'Avg_Funding_USD' in summary.columns:
            summary['Avg_Funding_USD'] = summary['Avg_Funding_USD'].round(2)
        if 'Avg_Funding_Rounds' in summary.columns:
            summary['Avg_Funding_Rounds'] = summary['Avg_Funding_Rounds'].round(2)
        if 'Avg_Milestones' in summary.columns:
            summary['Avg_Milestones'] = summary['Avg_Milestones'].round(2)
        summary['Acquired_Rate'] = (summary['Acquired_Rate'] * 100).round(1)

        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption('Acquired_Rate is the percentage of startups in each cluster labeled as acquired.')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Regression":
    st.markdown('<div class="page-title">Regression Models</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Linear & Polynomial Regression — predicting log(funding total)</div>',
                unsafe_allow_html=True)

    if st.button("▶  Run Regression"):
        y_reg = np.log1p(df['funding_total_usd'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42)

        reg_results = {}

        # Linear
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        reg_results['Linear'] = {
            'r2'  : r2_score(y_test, y_pred_lr),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            'pred': y_pred_lr
        }

        # Polynomial Deg 2
        poly2 = Pipeline([
            ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler',StandardScaler()),
            ('model', LinearRegression())
        ])
        poly2.fit(X_train, y_train)
        y_pred_p2 = poly2.predict(X_test)
        reg_results['Polynomial Deg-2'] = {
            'r2'  : r2_score(y_test, y_pred_p2),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_p2)),
            'pred': y_pred_p2
        }

        # Metrics
        st.markdown('<div class="section-header">MODEL PERFORMANCE</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Linear R²",        f"{reg_results['Linear']['r2']:.4f}"),
            ("Linear RMSE",      f"{reg_results['Linear']['rmse']:.4f}"),
            ("Poly Deg-2 R²",    f"{reg_results['Polynomial Deg-2']['r2']:.4f}"),
            ("Poly Deg-2 RMSE",  f"{reg_results['Polynomial Deg-2']['rmse']:.4f}"),
        ]
        for col, (label, val) in zip([c1,c2,c3,c4], metrics):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{val}</div></div>""",
                    unsafe_allow_html=True)

        # Actual vs Predicted plots
        st.markdown('<div class="section-header">ACTUAL vs PREDICTED</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#fafafa')
        for ax, (name, res) in zip(axes, reg_results.items()):
            ax.scatter(y_test, res['pred'], alpha=0.4, s=18, color='#8B0000')
            mn = min(y_test.min(), res['pred'].min())
            mx = max(y_test.max(), res['pred'].max())
            ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.2)
            ax.set_title(f"{name}  |  R²={res['r2']:.4f}",
                         fontweight='bold', fontsize=11)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_facecolor('#fafafa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Residuals
        st.markdown('<div class="section-header">RESIDUALS</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor('#fafafa')
        for ax, (name, res) in zip(axes, reg_results.items()):
            residuals = y_test.values - res['pred']
            ax.scatter(res['pred'], residuals, alpha=0.4, s=18, color='#8B0000')
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f"Residuals — {name}", fontweight='bold', fontsize=11)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")
            ax.set_facecolor('#fafafa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict a Startup":
    st.markdown('<div class="page-title">Predict a Startup</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter startup characteristics — get an instant prediction</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">STARTUP CHARACTERISTICS</div>',
                unsafe_allow_html=True)

    auto_label = "Auto (Best CV Model)"
    model_option = st.selectbox(
        "Prediction model",
        [auto_label, "Decision Tree", "KNN", "Naive Bayes", "SVM"],
        help="Run Classification once to enable automatic best-model selection from CV ROC-AUC."
    )

    if model_option == auto_label:
        selected_model_name = st.session_state.get('best_cv_model', 'Decision Tree')
        if 'best_cv_model' in st.session_state:
            st.caption(f"Using best CV model: {selected_model_name}")
        else:
            st.caption("Run Classification first to auto-pick best model. Using Decision Tree for now.")
    else:
        selected_model_name = model_option

    col1, col2 = st.columns(2)

    with col1:
        funding_rounds          = st.slider("Funding Rounds", 1, 10, 2)
        funding_total_usd       = st.number_input("Total Funding (USD)", 0,
                                                   500_000_000, 5_000_000,
                                                   step=500_000)
        milestones              = st.slider("Milestones Achieved", 0, 10, 2)
        relationships           = st.slider("Relationships (network size)", 0, 50, 8)
        avg_participants        = st.slider("Avg Participants per Round", 1.0, 10.0, 2.5)
        age_first_funding_year  = st.slider("Age at First Funding (years)", 0.0, 10.0, 1.5)

    with col2:
        age_last_funding_year   = st.slider("Age at Last Funding (years)", 0.0, 15.0, 3.0)
        has_VC                  = st.selectbox("VC Backed?", [0, 1],
                                               format_func=lambda x: "Yes" if x else "No")
        has_angel               = st.selectbox("Angel Backed?", [0, 1],
                                               format_func=lambda x: "Yes" if x else "No")
        is_software             = st.selectbox("Software Company?", [0, 1],
                                               format_func=lambda x: "Yes" if x else "No")
        is_web                  = st.selectbox("Web Company?", [0, 1],
                                               format_func=lambda x: "Yes" if x else "No")
        is_mobile               = st.selectbox("Mobile Company?", [0, 1],
                                               format_func=lambda x: "Yes" if x else "No")

    if st.button("🔮  Predict Now"):
        input_values = {
            'funding_rounds': funding_rounds,
            'funding_total_usd': funding_total_usd,
            'milestones': milestones,
            'relationships': relationships,
            'avg_participants': avg_participants,
            'age_first_funding_year': age_first_funding_year,
            'age_last_funding_year': age_last_funding_year,
            'has_VC': has_VC,
            'has_angel': has_angel,
            'is_software': is_software,
            'is_web': is_web,
            'is_mobile': is_mobile
        }

        model_X = X[feature_cols].copy()
        scaler_pred, clf_pred = train_prediction_model(model_X, y, selected_model_name)

        input_df_model = pd.DataFrame([
            {col: input_values.get(col, 0) for col in feature_cols}
        ])

        input_scaled = scaler_pred.transform(input_df_model)
        prediction   = clf_pred.predict(input_scaled)[0]
        proba        = clf_pred.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.markdown('<div class="section-header">PREDICTION RESULT</div>',
                    unsafe_allow_html=True)

        result_label = "ACQUIRED ✅" if prediction == 1 else "CLOSED ❌"
        badge_class  = "badge-acquired" if prediction == 1 else "badge-closed"
        confidence   = proba[prediction] * 100

        st.markdown(f'<div class="{badge_class}">{result_label}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.markdown(f"**Model used:** {selected_model_name}")

        # Probability bar
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        ax.barh(['Closed', 'Acquired'], [proba[0]*100, proba[1]*100],
                color=['#aaaaaa', '#8B0000'], edgecolor='white')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Prediction Probability", fontweight='bold')
        for i, v in enumerate([proba[0]*100, proba[1]*100]):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("**Input Summary:**")
        input_df = pd.DataFrame([
            {'Feature': col, 'Value': input_values.get(col, 0)}
            for col in feature_cols
        ])
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        if hasattr(clf_pred, 'feature_importances_'):
            st.markdown("**Top feature signals used by the model:**")
            pred_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': clf_pred.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            st.dataframe(pred_importance, use_container_width=True, hide_index=True)
        else:
            st.info("Feature-importance explanation is available for Decision Tree model.")
