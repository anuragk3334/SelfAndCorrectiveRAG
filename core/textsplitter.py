from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter     

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def chunk_documents(docs:List)->List:
    """Splits documents into smaller chunks.

    Args:
        docs (List): List of documents to be split.

    Returns:
        List: List of document chunks.
    """
    return text_splitter.split_documents(docs)

