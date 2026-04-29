import streamlit as st

def load_df():
    return st.session_state.get("df", None)