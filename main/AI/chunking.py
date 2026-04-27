from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents import documents

def chunking_docs(documents):

    if documents is not None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_text(documents)

        return texts