import streamlit as st

def load_df():
    return st.session_state.get("original_df", None)