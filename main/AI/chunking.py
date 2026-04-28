from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents import documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking_docs(text):
    if text:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        return splitter.split_text(text)
    return []