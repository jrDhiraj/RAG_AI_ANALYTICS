from load_df import load_df 
import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from  documents import documents


with st.expander("📊 Statistical Insights", expanded=True):

    def select_target(df):
        target_col = st.selectbox("Select the target column:", options=df.columns)
        documents['target_col'] = target_col
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        documents['dataframe_X'] = X.head(5)
        documents['dataframe_y'] = y.head(5)
        return X, y, target_col


    def categorical_df(X):
        cat_df = X.select_dtypes(exclude="number").columns.to_list()
        num_df = X.select_dtypes(include="number").columns.to_list()
        return cat_df, num_df


    def categorical_stats(X, y, target_col):
        st.subheader("📌 Categorical Insights")

        st.write("Target distribution:")
        st.bar_chart(y.value_counts())

        cat_df, _ = categorical_df(X)

        temp_df = X.copy()
        temp_df[target_col] = y

        results = []

        for col in cat_df:
            try:
                cont = pd.crosstab(temp_df[col], temp_df[target_col])

                if cont.shape[0] > 1 and cont.shape[1] > 1:
                    p_val = chi2_contingency(cont)[1]

                    results.append((col, p_val))

                    if p_val < 0.05:
                        st.success(f"{col} → Significant (p={p_val:.4f})")
                    else:
                        st.info(f"{col} → Not significant")

                    st.write(temp_df.groupby(col)[target_col].mean())
                    st.divider()

                else:
                    st.warning(f"{col} skipped (not enough categories)")

            except Exception as e:
                st.error(f"{col} error: {e}")
        documents['categorical stats'] = results
        return results


    def numerical_stats(X, y, target_col):
        st.subheader("📌 Numerical Insights")

        _, num_df = categorical_df(X)

        temp_df = X.copy()
        temp_df[target_col] = y

        results = []

        # check if binary target
        unique_vals = y.dropna().unique()

        if len(unique_vals) != 2:
            st.warning("T-test works only for binary target (2 classes)")
            return results

        class_1, class_0 = unique_vals[0], unique_vals[1]

        for col in num_df:
            try:
                group1 = temp_df[temp_df[target_col] == class_1][col].dropna()
                group0 = temp_df[temp_df[target_col] == class_0][col].dropna()

                if len(group1) > 1 and len(group0) > 1:
                    _, p_val = ttest_ind(group1, group0)

                    results.append((col, p_val))

                    mean1 = group1.mean()
                    mean0 = group0.mean()

                    if p_val < 0.05:
                        direction = "higher" if mean1 > mean0 else "lower"
                        st.success(f"{col} → Significant ({direction} for {class_1}) p={p_val:.4f}")
                    else:
                        st.info(f"{col} → Not significant")

                    st.write(temp_df.groupby(target_col)[col].mean())
                    st.divider()

                else:
                    st.warning(f"{col} skipped (not enough data)")

            except Exception as e:
                st.error(f"{col} error: {e}")
        documents['categorical stats'] = results
        return results


    def show_top_features(cat_results, num_results):
        st.subheader("🏆 Top Important Features")

        all_results = cat_results + num_results

        if not all_results:
            st.warning("No valid features to rank")
            return

        sorted_feats = sorted(all_results, key=lambda x: x[1])

        for col, p in sorted_feats[:5]:
            st.write(f"{col} → p-value: {p:.5f}")
        
            documents['Top feature '] = [col,p]


    def stats_of_data(df):
        st.subheader("📊 Statistical Measurements")

        X, y, target_col = select_target(df)

        cat_results = categorical_stats(X, y, target_col)
        num_results = numerical_stats(X, y, target_col)

        show_top_features(cat_results, num_results)


    df = load_df()

    if df is not None and not df.empty:
        stats_of_data(df)
    else:
        st.error("❌ DataFrame not found")