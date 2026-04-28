import streamlit as st
from documents import documents
from main.AI.chunking import chunking_docs
from main.AI.vector_db import create_vector_db
from main.AI.embedings import load_embedding_model

def aiui():
    query = st.chat_input("Ask something...")

    if "db" not in st.session_state:
        st.session_state.db = None

    # Create DB from documents (once)
    if st.session_state.db is None:
        if isinstance(documents, dict):
            texts = list(documents.values())
        else:
            texts = documents if isinstance(documents, list) else [str(documents)]

        # chunk each text
        chunks = []
        for t in texts:
            chunks.extend(chunking_docs(str(t)))
            
        st.session_state.db = create_vector_db(chunks)
        st.success("Vector DB created")

    if query and st.session_state.db:
        docs = st.session_state.db.similarity_search(query, k=3)

        st.write("🔍 Retrieved Docs:")
        for d in docs:
            st.write(d.page_content)


aiui()