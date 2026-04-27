from load_df import load_df 
import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

with st.expander("See data statistics"):

    def select_target(df):

        target_col = st.selectbox("Select the target column:", options=df.columns)
        # Separate features (X) and target (y)
        y = df[target_col]
        X = df.drop(columns=[target_col])

        return X, y, target_col

    def categorical_df(X):

        cat_df =  X.select_dtypes(exclude = "number").columns.to_list()

        num_df = X.select_dtypes(include = "number").columns.to_list()

        return cat_df , num_df


    def categorical_stats(X, y, target_col):
        st.write("Target distribution:")
        st.write(y.value_counts())
        cat_df, _ = categorical_df(X)

        temp_df = X.copy()
        temp_df['target'] = y

        for col in cat_df:
            cont = pd.crosstab(X[col], y).dropna()

            if cont.shape[0] > 1 and cont.shape[1] > 1:
                p_val = chi2_contingency(cont)[1]

                if p_val < 0.05:
                    st.write(f"{col} → Significant relationship (p={p_val:.4f})")
                else:
                    st.write(f"{col} → No significant relationship")
            else:
                st.write(f"{col} skipped (not enough categories)")
                continue
            st.write(f"group by {col} and {target_col}")
            st.write(temp_df.groupby(col)['target'].mean())
            st.write("--"*30)

    def numerical_stats(X, y, target_col):
        _, num_df = categorical_df(X)

        temp_df = X.copy()
        temp_df['target'] = y

        for col in num_df:

            positive_target = temp_df[temp_df['target'] == 1][col].dropna()
            negative_target = temp_df[temp_df['target'] == 0][col].dropna()


            _ , p_val = ttest_ind(positive_target, negative_target)

            if p_val > 0.05:
                st.write(f"there is significant relation between target and {col}")

            st.write(f"{p_val:.4f}")

            st.write(f"group by {col} and {target_col}")
            st.write(temp_df.groupby('target')[col].mean())
            st.write("--"*20)



    def stats_of_data(df):
        statistics_measurments = {}

        st.subheader("statistics measurments")

        X, y, target_col = select_target(df)

        categorical_stats(X, y, target_col)
        st.subheader("numerical stats")
        numerical_stats(X, y, target_col)



    df = load_df()

    if not df.empty:
        stats_of_data(df)
        
    else:
        st.write("df not found")
        

