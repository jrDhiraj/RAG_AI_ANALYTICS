from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents import documents
import warnings
warnings.filterwarnings("ignore")

def chunking_docs(text):
    if text:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        return splitter.split_text(text)
    return []