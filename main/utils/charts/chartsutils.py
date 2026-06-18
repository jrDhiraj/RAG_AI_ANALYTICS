import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from helper.data_utils import categorical_df, select_target
from load_df import load_df
from documents import documents

# ==========================================
# 1. INITIALIZE ALL SESSION STATE KEYS FIRST
# ==========================================
if "chart_summary" not in st.session_state:
    st.session_state.chart_summary = {
        "categorical_summary": {},
        "numerical_summary": {},
        "correlation_summary": {},
        "statistical_tests": {}
    }

if "data_insights" not in st.session_state:
    st.session_state.data_insights = {}

if "chart_docs" not in st.session_state:
    st.session_state.chart_docs = []

# Global page setup
st.set_page_config(layout="wide")

# ==========================================
# 2. DEFINE TOP-LEVEL MODULAR FUNCTIONS
# ==========================================

def categorical_chart(X, cat_cols, y, target_col):
    st.subheader("📊 Categorical Feature Distribution")
    
    temp_df = X.copy()
    temp_df[target_col] = y

    selected_cat = st.selectbox(
        "Select Categorical Column to Plot",
        cat_cols,
        key="select_cat_col"
    )

    col = selected_cat

    # Handle columns safely if they are entirely empty
    if temp_df[col].dropna().empty:
        st.warning(f"Column '{col}' contains no valid data.")
        return

    # Filter to top 10 categories to avoid cluttering the chart
    top = temp_df[col].value_counts().nlargest(10).index
    df_plot = temp_df[temp_df[col].isin(top)]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df_plot,
        x=col,
        hue=target_col,
        ax=ax
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Perform calculations safely
    if pd.api.types.is_numeric_dtype(df_plot[target_col]):
        grouped = df_plot.groupby(col)[target_col].mean()
        metric_label = "Mean Target"
    else:
        grouped = df_plot[col].value_counts()
        metric_label = "Count"

    # Display Streamlit native bar chart
    st.bar_chart(grouped)

    if not grouped.empty:
        top_cat, top_val = grouped.idxmax(), grouped.max()
        low_cat, low_val = grouped.idxmin(), grouped.min()

        # Update global session state structure safely
        st.session_state.chart_summary["categorical_summary"][col] = {
            "highest_category": str(top_cat),
            "highest_value": float(top_val),
            "lowest_category": str(low_cat),
            "lowest_value": float(low_val),
            "categories_analyzed": int(len(grouped))
        }
        
        # Display Summary Cards UI
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Top {metric_label} ({col})", f"{top_cat}", f"{top_val:.2f}")
        c2.metric(f"Lowest {metric_label} ({col})", f"{low_cat}", f"{low_val:.2f}")
        c3.metric("Total Categories Analysed", f"{len(grouped)}")


def numerical_chart(X, num_cols, y, target_col):
    st.subheader("📈 Numerical Feature Distribution & Dispersal")
    
    temp_df = X.copy()
    temp_df[target_col] = y

    selected_num = st.selectbox(
        "Select Numerical Column to Plot",
        num_cols,
        key="select_num_col"
    )

    col = selected_num
    clean_series = temp_df[col].dropna()

    if clean_series.empty:
        st.warning(f"Column '{col}' is empty.")
        return

    # Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(clean_series, kde=True, ax=ax)
    st.pyplot(fig)

    # Box Plot relative to target
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=temp_df, x=target_col, y=col, ax=ax)
    st.pyplot(fig)

    # Scatter Chart (If target column is also numerical)
    is_target_numeric = pd.api.types.is_numeric_dtype(temp_df[target_col])
    if is_target_numeric:
        st.scatter_chart(temp_df, x=col, y=target_col)

    # --- ADVANCED STATISTICAL CALCULATIONS ---
    mean_val = clean_series.mean()
    median_val = clean_series.median()
    std_val = clean_series.std()
    skew_val = clean_series.skew()
    kurt_val = clean_series.kurtosis()

    st.session_state.chart_summary["numerical_summary"][col] = {
        "mean": float(mean_val),
        "median": float(median_val),
        "std": float(std_val),
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "skewness": float(skew_val) if not pd.isna(skew_val) else 0.0,
        "kurtosis": float(kurt_val) if not pd.isna(kurt_val) else 0.0
    }

    # Statistical Hypotheses Testing (Normality Check via Shapiro-Wilk)
    # Shapiro-Wilk test works ideally for sample sizes N <= 5000
    sample_data = clean_series.sample(min(100, len(clean_series))) if len(clean_series) > 100 else clean_series
    if len(sample_data) >= 3:
        stat, p_val = stats.shapiro(sample_data)
        is_normal = "Yes (Normal)" if p_val > 0.05 else "No (Skewed)"
    else:
        p_val, is_normal = 1.0, "Insufficient Data"

    # UI Statistical Cards Layout
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean", f"{mean_val:.2f}")
    m2.metric("Median", f"{median_val:.2f}")
    m3.metric("Std Dev", f"{std_val:.2f}")
    m4.metric("Skewness", f"{skew_val:.2f}")
    m5.metric("Is Normally Distributed?", is_normal)


def correlation_chart(X, num_cols):
    st.subheader("🔥 Correlation Analysis Matrix")
    
    if len(num_cols) < 2:
        st.warning("Not enough numerical columns to execute a correlation matrix.")
        return

    num_df = X[num_cols].select_dtypes(include="number").dropna(how="all")
    if len(num_df.columns) < 2:
        st.warning("Need at least 2 clean numerical columns for correlation matrix analysis.")
        return

    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    strongest_corr = 0.0
    feature1, feature2 = None, None

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = abs(corr.iloc[i, j])
            if value > strongest_corr:
                strongest_corr = value
                feature1 = corr.columns[i]
                feature2 = corr.columns[j]

    if feature1 and feature2:
        st.session_state.chart_summary["correlation_summary"] = {
            "feature_1": feature1,
            "feature_2": feature2,
            "correlation": float(strongest_corr)
        }
        st.info(f"💡 **Strongest Linear Relationship Found:** `{feature1}` ↔ `{feature2}` (Absolute R Value: **{strongest_corr:.2f}**)")


def advanced_statistical_tests(X, cat_cols, num_cols, y, target_col):
    """
    Performs critical structural inferences (ANOVA/T-Test) automatically 
    between categorical variables and a numerical target feature.
    """
    st.subheader("🔬 Automated Hypothesis Testing Inference")
    temp_df = X.copy()
    temp_df[target_col] = y

    # Only run inference if target feature is numerical
    if not pd.api.types.is_numeric_dtype(temp_df[target_col]) or temp_df[target_col].dropna().empty:
        st.info("Continuous automated testing (ANOVA/T-test) is skipped because the selected Target Variable is Categorical.")
        return

    if not cat_cols:
        st.warning("No categorical variables available to compute group-variance relationships against target.")
        return

    test_results = {}
    
    for cat in cat_cols:
        # Create groups based on category levels
        groups = [df_target.dropna() for _, df_target in temp_df.groupby(cat)[target_col] if len(df_target.dropna()) > 1]
        
        if len(groups) == 2:
            stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            test_name = "Independent T-Test"
        elif len(groups) > 2:
            stat, p_val = stats.f_oneway(*groups)
            test_name = "One-Way ANOVA"
        else:
            continue

        significant = "Yes (p < 0.05)" if p_val < 0.05 else "No"
        test_results[cat] = {
            "test_type": test_name,
            "f_or_t_statistic": float(stat) if not pd.isna(stat) else 0.0,
            "p_value": float(p_val) if not pd.isna(p_val) else 1.0,
            "statistically_significant": significant
        }

    if test_results:
        st.session_state.chart_summary["statistical_tests"] = test_results
        st.write("Below are the feature interactions with your target feature (`" + target_col + "`):")
        st.dataframe(pd.DataFrame(test_results).T)
    else:
        st.info("No categorical variables containing sufficient group variances were found to run statistical testing loops.")


def run_charts_pipeline(df):
    st.session_state.chart_docs = []

    # Get features/targets safely
    X, y, target_col = select_target(df)
    cat_cols, num_cols = categorical_df(X)

    # Execute visualization modules
    if cat_cols:
        categorical_chart(X, cat_cols, y, target_col)
    else:
        st.info("No explicit categorical features found in dataset.")

    if num_cols:
        numerical_chart(X, num_cols, y, target_col)
    else:
        st.info("No explicit numerical columns found in dataset.")

    # Execute global multi-column relationships 
    correlation_chart(X, num_cols)
    advanced_statistical_tests(X, cat_cols, num_cols, y, target_col)

    # Sync safely with document stores without risking variable reference collision
    documents["chart_summary"] = st.session_state.chart_summary
    st.session_state.data_insights["charts_analysis"] = documents

# ==========================================
# 3. INTERACTIVE DASHBOARD VIEW BUILDER
# ==========================================
with st.expander("📊 Data Visualization Dashboard", expanded=True):
    if "df" in st.session_state:
        current_df = st.session_state.df
        run_charts_pipeline(current_df)
       
