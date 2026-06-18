
import streamlit as st
import pandas as pd
from  load_df import load_df
import io

documents = {}

st.set_page_config(layout="wide")
with st.expander("See data description", True):

    def dataset_description(df):
        
        st.write("remove coluns from here")
        all_columns = df.columns.tolist()
        
        selected_col = st.multiselect("select col you want to keep", all_columns, default=all_columns)

        if selected_col:
            st.dataframe(df[selected_col])
            df = df[selected_col]

            st.session_state.df = df

        else:
            st.warning("Please select at least one column to display.")


        df_shape = df.shape

        st.write("Dataframe Description")
        numes = df.select_dtypes(include=["number"])

        if not numes.empty:
            st.write(numes.shape)
            st.write(numes.columns.tolist())
            df_description = numes.describe()
            documents['df_description'] = df_description
            st.dataframe(df_description, width="stretch")
        
       
        catg = df.select_dtypes(include=["object", "category"])
        if not catg.empty:
            st.divider()
            st.write("Categorical Description")
            categorical_description = catg.describe()
            documents['categorical_description'] = categorical_description
            st.dataframe(categorical_description, width="stretch")

        st.write("Df information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        documents['info_str'] = info_str
        st.text(info_str)
        st.space()
        

        st.write("is null")
        df_isnull = df.isnull().sum()
        null_row = df_isnull[df_isnull > 0]
        documents['null_row'] = null_row
        st.dataframe(null_row)
        

        st.write("Dataframe duplicate")
        df_isduplicats = df.duplicated().sum()
        documents['df_isduplicats'] = df_isduplicats
        st.write(df_isduplicats)
        st.space()

        if "data_insights" not in st.session_state:
            st.session_state.data_insights = {}
            st.session_state.data_insights["dataset_summary"] = documents

    df = load_df()

    if df is not None:
        st.dataframe(df.head())
        dataset_description(df)
    else:
        st.warning("Please upload data from sidebar")



