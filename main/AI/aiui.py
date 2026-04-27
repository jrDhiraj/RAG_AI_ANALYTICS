import streamlit as st
from documents import documents
import datetime


def aiui():
    inp_query = st.chat_input("Write your query")

    if inp_query:   # only run when user inputs something
        n = len(documents)

        documents["metadata"] = {
            "time": datetime.datetime.now(),
            "len_of_docs": n
        }

        st.write("User Query:", inp_query)
        st.write("Documents:", documents)


aiui()