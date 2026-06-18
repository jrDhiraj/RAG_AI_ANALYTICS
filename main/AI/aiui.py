import streamlit as st
from documents import documents
from main.AI.chunking import chunking_docs
from main.AI.vector_db import create_vector_db
from main.AI.llm_model import model_llm

st.set_page_config(layout="wide")
def collect_all_insights():

    knowledge = []

    if "data_insights" in st.session_state:
        knowledge.append(
            str(st.session_state.data_insights)
        )

    if "chart_summary" in st.session_state:
        knowledge.append(
            str(st.session_state.chart_summary)
        )

    if "statistical_summary" in st.session_state:
        knowledge.append(
            str(st.session_state.statistical_summary)
        )

    if len(documents) > 0:
        knowledge.append(
            str(documents)
        )
    
    return knowledge

def build_vector_db():

    knowledge = collect_all_insights()

    chunks = []

    for text in knowledge:

        chunks.extend(
            chunking_docs(text)
        )

    if len(chunks) == 0:

        st.error(
            "No analysis available"
        )

        return

    st.session_state.vector_db = (
        create_vector_db(chunks)
    )

    st.success(
        f"Knowledge Base Created ({len(chunks)} chunks)"
    )


def aiui():

    st.title("Dataset AI Assistant")

    if st.button(
        "Build Knowledge Base"
    ):

        build_vector_db()

    query = st.chat_input(
        "Ask about dataset..."
    )

    if not query:
        return

    if "db" not in st.session_state:

        st.warning(
            "Build Knowledge Base First"
        )

        return

    docs = (
        st.session_state.vector_db
        .similarity_search(
            query,
            k=10
        )
    )

    context = "\n".join(
        [
            d.page_content
            for d in docs
        ]
    )

    answer = model_llm(
        query,
        context
    )

    st.chat_message(
        "AI"
    ).write(answer)


aiui()