from typing import List

from ingestion.pdfload import load_pdf
import os
from core.textsplitter import chunk_documents
from core.vectorstore import write_embeddings_to_pinecone    

def ingestPdf():
    print("Loading PDF document...")
    pdf_path = os.path.join(os.path.dirname(__file__)+"\ingestion","Diabetes_RAG_Practice.pdf")
    print("Loading PDF document ends...")
    print(f"PDF path: {pdf_path}")
    docs = load_pdf(pdf_path)
    #print(docs)
    #Chunking the documents into smaller pieces
    chunks = chunk_documents(docs)
    
    #update MetaData of the Chunks
    chunks=updateChunkMetadata(chunks)
    #print(chunks[0].metadata)
    #Write the chunks and their embeddings to Pinecone vector store
    print("Writing embeddings to Pinecone vector store...")
    write_embeddings_to_pinecone(chunks)
    print("Ingestion completed successfully!")
    
    
    
    
def updateChunkMetadata(chunks:List)->List:
    """Updates the metadata of the document chunks.

    Args:
        chunks (List): List of document chunks.
        

    Returns:
        List: List of document chunks with updated metadata.
    """
    file_name=chunks[0].metadata['source']
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['source']=os.path.basename(file_name)
        chunk.metadata['chunk_id']=i
        chunk.metadata['subject']="cancer"
        
    return chunks
 
if __name__=="__main__":
    ingestPdf()
    