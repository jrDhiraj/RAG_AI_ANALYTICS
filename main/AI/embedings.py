from main.AI.chunking import chunking_docs
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )


    