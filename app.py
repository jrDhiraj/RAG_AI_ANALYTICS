import streamlit as st
import pandas as pd
import chardet

st.title("AI ANALYTICS")




def upload_data():
    st.sidebar.title("Upload Data")

    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        try:
            rawdata = file.read()
            result = chardet.detect(rawdata)

            encoding = result['encoding']
            file.seek(0)

            df = pd.read_csv(file, encoding=encoding)

            st.session_state.df = df
            st.session_state.file_name = file.name

            st.sidebar.success(f" {file.name} ({encoding})")
            st.sidebar.write(f"Shape: {df.shape}")

        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    elif "df" in st.session_state:
        st.sidebar.success(f" Loaded: {st.session_state.file_name}")
        st.sidebar.write(f"Shape: {st.session_state.df.shape}")

    else:
        st.sidebar.info("Upload a CSV file")

upload_data()

pages = [
    st.Page("main/utils/home/home_page.py", title="Description", icon="🏠"),
    st.Page("main/utils/stats/stats_page.py", title="Statistics", icon="📏"),
    st.Page("main/utils/charts/chartsutils.py", title="Charts", icon="📊"),
    st.Page("main/AI/aiui.py", title="AI Chat", icon="🤖", ),
]

pg = st.navigation(pages, position="top")
pg.run()