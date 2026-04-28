from langchain_community.vectorstores import FAISS
import streamlit as st
from main.AI.embedings import load_embedding_model

def create_vector_db(chunks):
    embeddings = load_embedding_model()
    db = FAISS.from_texts(chunks, embeddings)
    return db