
import streamlit as st
import pandas as pd
from  load_df import load_df
import io

with st.expander("See data description"):

    def dataset_description(df):
        st.write("Dataframe shape")
        st.write(df.shape)
        st.space()


        st.write("Dataframe Description")
        st.dataframe(df.describe(), width="stretch")
        
       
        catg = df.select_dtypes(include=["object", "category"])
        if not catg.empty:
            st.divider()
            st.write("Categorical Description")
            st.dataframe(catg.describe(), width="stretch")

        st.write("Df information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        st.space()
        

        st.write("is null")
        st.write(df.isnull().sum())
        st.space()

        st.write("Dataframe duplicate")
        st.write(df.duplicated().sum())
        st.space()


    try:

        df = load_df()
        if df is None:
            st.write("upload dataset")
        else:
            # df = pd.read_csv(df)
            st.toast("Dataset upload sucessfully")
            st.dataframe(df, width="stretch")
            dataset_description(df)
    except:
        ValueError


# with st.expander("see charts"):
#     st.write("chart is given")
