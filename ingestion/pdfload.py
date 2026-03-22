from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path):
    print(f"Loading PDF document from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents=loader.load()
    return documents


    