from chunking import chunking_docs
from langchain_huggingface import HuggingFaceEmbeddings


def embeddings_pipeline(documents):
    # Step 1: chunk
    chunks = chunking_docs(documents)

    # Step 2: embeddings model
    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    # Step 3: embed
    embeddings = hf.embed_documents(chunks)

    return chunks, embeddings


    