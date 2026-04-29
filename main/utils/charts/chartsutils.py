import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from main.utils.stats.stats_page import categorical_df, select_target
from load_df import load_df


with st.expander(" Data Visualization Dashboard", expanded=True):

    col1, col2 = st.columns(2)

    with col1:


        def categorical_chart(X, cat_cols, y, target_col):
            try:
                temp_df = X.copy()
                temp_df[target_col] = y

                for col in cat_cols:
                    st.write(f"🔹 {col} vs {target_col}")

                    # limit categories (avoid clutter)
                    top = temp_df[col].value_counts().nlargest(10).index
                    df_plot = temp_df[temp_df[col].isin(top)]

                    # Count Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.countplot(data=df_plot, x=col, hue=target_col, ax=ax)

                    plt.xticks(rotation=45, ha='right')
                    ax.set_title(f"{col} vs {target_col}")
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Mean Target
                    grouped = df_plot.groupby(col)[target_col].mean()
                    st.write("Mean Target Value")
                    st.bar_chart(grouped)

                    st.divider()

            except Exception as e:
                st.error(f"Error in categorical chart: {e}")


    with col2:
        

        def numerical_chart(X, num_cols, y, target_col):
            try:
                temp_df = X.copy()
                temp_df[target_col] = y

                for col in num_cols:
                    st.write(f"🔹 {col} Distribution")

                    # Histogram
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(temp_df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(f"{col} Histogram")
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Boxplot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.boxplot(data=temp_df, x=target_col, y=col, ax=ax)
                    ax.set_title(f"{col} vs {target_col}")
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Scatter (NO matplotlib here)
                    st.write("Scatter Plot")
                    st.scatter_chart(temp_df, x=col, y=target_col)

                    st.divider()

            except Exception as e:
                st.error(f"Error in numerical chart: {e}")


    st.subheader("Correlation Heatmap")

    def correlation_chart(X, num_cols):
        try:
            if len(num_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    X[num_cols].corr(),
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    linewidths=0.5,
                    ax=ax
                )
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Not enough numerical columns for correlation")
        except Exception as e:
            st.error(f"Error in correlation chart: {e}")


    def charts(df):
        try:
            X, y, target_col = select_target(df)
            cat_cols, num_cols = categorical_df(X)

            if cat_cols:
                categorical_chart(X, cat_cols, y, target_col)
            else:
                st.warning("No categorical columns found")

            if num_cols:
                numerical_chart(X, num_cols, y, target_col)
                correlation_chart(X, num_cols)
            else:
                st.warning("No numerical columns found")

        except Exception as e:
            st.error(f"Error in charts function: {e}")

    df = load_df()

    if df is not None:
        st.dataframe(df.head())
        charts(df)
    else:
        st.warning("Please upload data from sidebar")