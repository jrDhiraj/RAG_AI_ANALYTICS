import streamlit as st
from documents import documents
from main.AI.chunking import chunking_docs
from main.AI.vector_db import create_vector_db
from main.AI.llm_model import model_llm

st.set_page_config(layout="wide")

def aiui():
    st.title("AI Data Assistant")

    query = st.chat_input("Ask something...")

    if "db" not in st.session_state:
        st.session_state.db = None

    if st.session_state.db is None:

        if isinstance(documents, dict):
            texts = [str(v) for v in documents.values()]
        elif isinstance(documents, list):
            texts = documents
        else:
            texts = [str(documents)]

        # chunking
        chunks = []
        for t in texts:
            chunks.extend(chunking_docs(str(t)))

        st.write("Total chunks created:", len(chunks))

        if len(chunks) == 0:
            st.error(" No data available to create vector DB")
            return

        st.session_state.db = create_vector_db(chunks)
        st.success(" Vector DB created")

    if query and st.session_state.db:

        st.write("🔍 Searching...")

        docs = st.session_state.db.similarity_search(query, k=3)

        if not docs:
            st.warning("No relevant data found")
            return

        # LLM answer
        st.write(" Generating Answer...")

        answer = model_llm(query, docs)

        st.subheader("Answer:")
        st.write(answer)


aiui()