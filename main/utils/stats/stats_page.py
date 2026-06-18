import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from load_df import load_df 
from main.utils.insights.insights_generator import generate_insights

# ==========================================
# 1. STRUCTURAL REGISTRATION FOR RAG PIPELINES
# ==========================================
if "data_insights" not in st.session_state:
    st.session_state.data_insights = {}

# Dictionary initialized with multi-level depth to prevent KeyErrors
documents = {
    "missing_pattern_narratives": {},
    "outlier_risk_profiles": {},
    "feature_redundancy_matrix": {},
    "distribution_modality_insights": {},
    "rag_metadata_summary": ""
}

st.set_page_config(layout="wide")

# ==========================================
# 2. SEPARATION OF MODULE FUNCTIONS
# ==========================================

def select_target(df):
    """Safely extracts target metrics and updates root RAG elements."""
    target_col = st.selectbox("Select the target column:", options=df.columns, key="rag_stats_target_select")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y, target_col


def get_feature_types(X):
    """Splits dataframe features into explicit structural partitions."""
    cat_cols = X.select_dtypes(exclude="number").columns.to_list()
    num_cols = X.select_dtypes(include="number").columns.to_list()
    st.write(cat_cols)
    st.write(num_cols)
    return cat_cols, num_cols


# ==========================================
# 3. ADVANCED ANALYSIS MODULES (NO REPETITION)
# ==========================================

def missing_pattern_analysis(X):
    """
    ANALYSIS 1: Missingness & RAG Data Completeness.
    Teaches the RAG system which columns cannot be trusted due to high sparsity.
    """
    st.subheader("Data Completeness & Missing Pattern Analysis")
    total_rows = len(X)
    missing_data = X.isna().sum()
    pct_missing = (missing_data / total_rows) * 100
    
    sparse_cols = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if sparse_cols.empty:
        st.success("Perfect Data Completeness! No missing records detected anywhere across features.")
        documents["missing_pattern_narratives"]["global"] = "The dataset features 100% data integrity with no missing fields."
        return

    st.write("Variables with Missing Records:")
    for col, count in sparse_cols.items():
        pct = pct_missing[col]
        severity = "High" if pct > 20 else "Moderate" if pct > 5 else "Low"
        
        st.warning(f"`{col}` is missing **{count}** records ({pct:.2f}%). Severity: **{severity}**")
        
        # Build clean semantic context for RAG text indexing
        documents["missing_pattern_narratives"][col] = (
            f"Feature '{col}' contains {count} missing records, representing {pct:.2f}% of total data observations. "
            f"The severity of this structural gap is deemed {severity}. Any RAG queries or predictive pipelines referencing "
            f"this column must accommodate missing entry values using appropriate strategies."
        )


def outlier_and_extremes_profiling(X, num_cols):
    """
    ANALYSIS 2: Outliers & Distribution Skew.
    Identifies anomalies using the Interquartile Range (IQR) technique.
    """
    st.subheader("Extreme Outliers & Data Bias Diagnostics")
    
    if not num_cols:
        st.info("No numerical fields found to evaluate for data outlier spikes.")
        return

    outlier_found = False
    
    for col in num_cols:
        series = X[col].dropna()
        if len(series) < 4:
            continue
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        if not outliers.empty:
            outlier_found = True
            outlier_pct = (len(outliers) / len(series)) * 100
            
            st.error(f"`{col}` contains **{len(outliers)}** severe outliers ({outlier_pct:.2f}% of rows).")
            
            # Record semantic structure for downstream processing
            documents["outlier_risk_profiles"][col] = {
                "outlier_count": int(len(outliers)),
                "outlier_percentage": float(outlier_pct),
                "lower_cutoff": float(lower_bound),
                "upper_cutoff": float(upper_bound),
                "narrative": (
                    f"Column '{col}' exhibits significant outlier anomalies. It has {len(outliers)} rows falling outside the "
                    f"mathematical 1.5x IQR boundary limits. Outliers comprise {outlier_pct:.2f}% of the feature's complete population. "
                    f"The values range below {lower_bound:.4f} and above {upper_bound:.4f}."
                )
            }
            
    if not outlier_found:
        st.success("Data Uniformity clear! No columns contain severe outlying value distributions.")


def multicollinearity_redundancy_check(X, num_cols):
    """
    ANALYSIS 3: Multicollinearity Detection (VIF Analysis).
    Tells the RAG system which features are redundant and overlap in information.
    """
    st.subheader("Feature Redundancy & Information Overlap (VIF Analysis)")
    
    if len(num_cols) < 2:
        st.info("Multicollinearity check skipped. A minimum of two numerical attributes are required.")
        return

    try:
        # Create a clean matrix filled with medians to prevent calculation failure
        numeric_matrix = X[num_cols].fillna(X[num_cols].median())
        
        # Inject a constant column required to calculate the VIF intercept properly
        numeric_matrix["_intercept"] = 1.0
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = num_cols
        vif_data["VIF"] = [
            variance_inflation_factor(numeric_matrix.values, i) 
            for i in range(len(num_cols))
        ]
        
        # Filter intercept out of visible data displays
        vif_data = vif_data.sort_values(by="VIF", ascending=False)
        
        high_vif = vif_data[vif_data["VIF"] > 5.0]
        
        if not high_vif.empty:
            st.warning("High Information Redundancy Found! The following features strongly mirror each other:")
            st.dataframe(high_vif)
            
            for _, row in high_vif.iterrows():
                f_name = row["feature"]
                v_val = row["VIF"]
                documents["feature_redundancy_matrix"][f_name] = (
                    f"Feature '{f_name}' exhibits severe structural variance redundancy with an inflation value of {v_val:.2f}. "
                    f"This means its informational value is heavily duplicated across other variables in the dataset."
                )
        else:
            st.success("Excellent! All features contain unique statistical signals (All Variance Inflation Factors < 5.0).")
            documents["feature_redundancy_matrix"]["global"] = "No multicollinearity or structural redundancy exists."
            
    except Exception as e:
        st.caption(f"Information redundancy matrix could not be evaluated: {e}")


def distribution_modality_check(X, num_cols):
    """
    ANALYSIS 4: Modality Tracking (Isolates multi-peak behaviors).
    Flags columns with two or more common centers, signaling distinct sub-populations.
    """
    st.subheader("Multi-Peak (Modality) Distribution Discovery")
    
    if not num_cols:
        return

    multimodal_cols = []
    
    for col in num_cols:
        try:
            series = X[col].dropna()
            if len(series) < 30:
                continue
                
            # Perform a smooth Kernel Density Estimate (KDE) over the series space
            kde = stats.gaussian_kde(series)
            sample_space = np.linspace(series.min(), series.max(), 200)
            evaluated_density = kde(sample_space)
            
            # Locate peaks (local maxima) in the density curve
            peaks = [
                i for i in range(1, len(evaluated_density) - 1)
                if evaluated_density[i] > evaluated_density[i-1] and evaluated_density[i] > evaluated_density[i+1]
            ]
            
            # Filter out minor noise peaks by requiring a minimum density threshold
            significant_peaks = [p for p in peaks if evaluated_density[p] > (evaluated_density.max() * 0.15)]
            
            if len(significant_peaks) >= 2:
                multimodal_cols.append(col)
                peak_values = sample_space[significant_peaks]
                formatted_peaks = ", ".join([f"{val:.2f}" in peak_values])
                
                st.info(f"`{col}` is **Multimodal** (Contains {len(significant_peaks)} distinct data peaks near: `{formatted_peaks}`).")
                
                documents["distribution_modality_insights"][col] = (
                    f"Feature '{col}' exhibits a complex, multi-modal probability density with {len(significant_peaks)} unique subgroups. "
                    f"This signifies that the column data represents mixed subpopulations instead of a single uniform audience group."
                )
        except Exception:
            continue
            
    if not multimodal_cols:
        st.success("All continuous fields show standard, single-peak (unimodal) behavioral trends.")


# ==========================================
# 4. MASTER CONTROLLER PIPELINE RUNNER
# ==========================================

def run_advanced_autoanalysis_pipeline(df):
    """Coordinates and executes the automated analysis tasks sequentially."""
    X, y, target_col = select_target(df)
    cat_cols, num_cols = get_feature_types(X)
    
    # Run the individual, non-overlapping analysis routines
    missing_pattern_analysis(X)



if "df" in st.session_state:
    current_df = st.session_state.df
    run_advanced_autoanalysis_pipeline(current_df)
    if st.session_state.data_insights:
        st.session_state.data_insights["insight"] = documents