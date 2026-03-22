

import json
from multiprocessing import context

from langchain.messages import HumanMessage, SystemMessage

from core.llm import llm


def is_rel(query:str ,chunks:list[dict])->list[dict]:
    """Determines if retrieved chunks are relevant to the query.

    Args:
        query (str): The user query.
        chunks (list[dict]): List of retrieved document chunks.

    Returns:
        list[dict]: List of relevant chunks with content, source, page, and relevance score.
    """
    parts=[]
    for index,chunk in enumerate(chunks):
        if hasattr(chunk, 'page_content'):
            # It's a Document
            label=f"[Chunk{index+1}] Page{chunk.metadata.get('page', '?')}"
            content=chunk.page_content[0:400]
        else:
            # It's a dict
            label=f"[Chunk{index+1}] Page{chunk.get('page', '?')}"
            content=chunk['content'][0:400]
        
        chunk_string=label +"\n\n"+content
        parts.append(chunk_string)
        
    chunks_text="\n\n---\n\n".join(parts)
    # print(chunks_text)
    
    result = llm.invoke([
        SystemMessage(content="""You are a medical document relevance judge.
For each numbered chunk, judge if it contains information that
directly answers the user query about diabetes.

Return JSON array — one entry per chunk in order:
[
  {"chunk": 1, "relevant": true,  "score": 0.0-1.0, "reason": "brief"},
  {"chunk": 2, "relevant": false, "score": 0.0-1.0, "reason": "brief"}
]
Be strict. Score above 0.5 = relevant."""),
        HumanMessage(content=f"Query: {query}\n\nChunks:\n{chunks_text}")
    ])
    
    judgements=json.loads(result.content)
    #print(f"Relevance judgements: {judgements}")
    
    rel_chunks=[]
    
    for i,chunk  in enumerate(chunks):
        j=judgements[i]
        if j["relevant"]:
            if hasattr(chunk, 'page_content'):
                content = chunk.page_content
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", "?")
            else:
                content = chunk['content']
                source = chunk.get("source", "unknown")
                page = chunk.get("page", "?")
            rel_chunk={
                "content": content,
                "source": source,
                "page": page,
                "score": j["score"],
                "reason": j["reason"]
            }
            rel_chunks.append(rel_chunk)
            #print(rel_chunks)
            
    return rel_chunks

def generate_response(query:str, relevant_chunks:list[dict])->str:
    """Generates a response to the user query based on relevant chunks.

    Args:
        query (str): The user query.
        relevant_chunks (list[dict]): List of relevant chunks with content, source, page, and relevance score.

    Returns:
        str: The generated response to the user query.
    """
    
    context_list=[]
    
    
    
    context = "\n\n".join([
        f"[Source: {chunk['source']} | Page {chunk['page']} | Score {chunk['score']:.2f}]\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""Answer the question based ONLY on the context below. If the answer is not in the context, say "I don't have that information in the documents."
          Context:
        {context}
        Question: {query}
         Answer:"""

    return llm.invoke(prompt).content.strip()

def sup_check(response:str,chunks: list[dict])->dict:
    """Checks if the generated response is supported by the retrieved chunks.

    Args:
        response (str): The generated response to the user query.
        chunks (list[dict]): List of retrieved document chunks.

    Returns:
        str: "supported" if the response is supported by the chunks, otherwise "not supported".
    """
    
    top_chunks=chunks[:3]
    
    context_part=[]
    
    for chunk in top_chunks:
        context_part.append(chunk['content'][:400])

        
    context_text="\n\n---\n\n".join(context_part)
    
    systemMessage=SystemMessage(content="""You are a helpful and precise assistant for checking if the answer is supported by the retrieved document chunks.
    Return JSON only:
{
  "supported": "fully" | "partially" | "not_supported",
  "unsupported_claims": ["list claims NOT found in sources"],
  "reason": "one sentence"
}""")
    
    humanMessage=HumanMessage(content=(
            "Source documents:\n"
            + context_text
            + "\n\nGenerated response:\n"
            + response
        ))
    
    result = llm.invoke([systemMessage,humanMessage])
    
    try:
        judgment=json.loads(result.content)
    except json.JSONDecodeError:
        print("Error decoding JSON:", result.content)
        judgment = {
            "supported":          "fully",
            "unsupported_claims": [],
            "reason":             "could not parse LLM response"
        }
    
    print(f"Support check result: {judgment}")
    return judgment
 
        
         