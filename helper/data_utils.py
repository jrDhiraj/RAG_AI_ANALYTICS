import streamlit as st

def select_target(df):

    target_col = st.selectbox(
        "Select Target",
        df.columns
    )

    y = df[target_col]

    X = df.drop(
        columns=[target_col]
    )

    return X, y, target_col


def categorical_df(X):

    cat_cols = (
        X
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    num_cols = (
        X
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    return cat_cols, num_cols