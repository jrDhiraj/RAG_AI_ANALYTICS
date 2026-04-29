
import streamlit as st
import pandas as pd
from  load_df import load_df
import io
from documents import documents

with st.expander("See data description"):

    def dataset_description(df):
        st.write("Dataframe shape")
        df_shape = df.shape
        st.write(df_shape)
        st.space()


        st.write("Dataframe Description")
        df_description = df.describe()
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
        documents['df_isnull'] = df_isnull
        st.write(df_isnull)
        st.space()

        st.write("Dataframe duplicate")
        df_isduplicats = df.duplicated().sum()
        documents['df_isduplicats'] = df_isduplicats
        st.write(df_isduplicats)
        st.space()

        return df_shape,df_description,categorical_description,info_str,df_isnull,df_isduplicats


    df = load_df()

    if df is not None:
        st.dataframe(df.head())
        dataset_description(df)
    else:
        st.warning("Please upload data from sidebar")



