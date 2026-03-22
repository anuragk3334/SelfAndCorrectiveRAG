
from core.llm import llm
from rag_adaptive import router
from no_rag_retrieval import answer as no_rag_answer
from core.vectorstore import query_pinecone
from rag_fusion import retrieve 
from rag_self import is_rel,generate_response,sup_check
from rag_corrective import corrective_isrel, corrective_issup

def answer(query:str)->dict:
    
    # Get Routing Decision using Adaptive RAG Router 
    routing=router(query)
    strategy=routing.get("strategy","single")
    
    if strategy=="no_retrieval":
        response=no_rag_answer(query)
        return {
            "query":    query,
            "strategy": "no_retrieval",
            "response": response,
            "chunks_used": 0
        }
        
    
    elif strategy=="single":
        chunks= query_pinecone(query=query, k=5)
        
    elif strategy=="rag_fusion":
        chunks=retrieve(query, top_k=5)
        
    else:
        print(f"Unknown strategy '{strategy}' returned by router. Defaulting to single retrieval.")
        chunks= query_pinecone(query=query, k=5)
   
   # Self RAG relevance check
    rel_chunks=[]
    current_chunks=chunks
    isrel_retries=0
    
    while isrel_retries<=2 :
        rel_chunks=is_rel(query,current_chunks)
        if rel_chunks:
            break
        
        print(f"No relevant chunks found.Go for corrective RAG . Retrying relevance check (Attempt {isrel_retries + 1})...")
        current_chunks=corrective_isrel(query, isrel_retries, index=None, embeddings=None)
        isrel_retries+=1
            
    if not rel_chunks:
        print("No relevant information found for the query after corrective attempts. Returning fallback answer.")
        # Exhausted all corrective attempts
        return {
                "query":    query,
                "strategy": strategy,
                "response": "I could not find relevant information in the document after multiple attempts.",
                "chunks_used": 0,
                "escalated": True
            }
    
    #Generate answer using relevant chunks:
    print(f"\n[Generate] Generating response from {len(rel_chunks)} chunks")
    response = generate_response(query, rel_chunks)
    print(f"\n[Generate] Generated response: {response}")
    
    # Check if the response is supported by the chunks
    issup_retries = 0
    judgment      = {}
    
    while issup_retries <= 2:
        judgment =sup_check(response, rel_chunks)
        if(judgment["supported"].lower()== "fully"):
            print("Response is supported by the retrieved chunks.")
            break
        
        unsupported_claims=judgment.get("unsupported_claims", [])
        response=corrective_issup(query, response, rel_chunks,unsupported_claims,issup_retries)
        issup_retries=issup_retries+1
        
    # ── Return final result ────────────────────────────────────
    print(f"\n{'─'*60}")
    print("FINAL RESPONSE:")
    print(response)
    print(f"\nStrategy used:   {strategy}")
    print(f"Chunks used:     {len(rel_chunks)}")
    print(f"IsRel retries:   {isrel_retries}")
    print(f"IsSup retries:   {issup_retries}")

    return {
        "query":          query,
        "strategy":       strategy,
        "response":       response,
        "chunks_used":    len(rel_chunks),
        "isrel_retries":  isrel_retries,
        "issup_retries":  issup_retries,
        "groundedness":   judgment["supported"]
    }
    
if __name__ == "__main__":

    # Tests each strategy
    test_cases = [
        # no_retrieval
        "What does HbA1c stand for?",

        # single retrieval
        "What is the HbA1c target for elderly frail diabetic patients?",

        # rag_fusion — informal language
        "which diabetes injection is best for losing weight?",

        # decompose — complex comparison
        "Compare the cardiovascular outcome trial results for semaglutide, "
        "liraglutide and dulaglutide with exact hazard ratios and trial names",

        # IsSup stress test — exact numbers
        "What is the exact potassium target range during DKA management?"
    ]

    results = []
    for query in test_cases:
        result = answer(query)
        results.append(result)
        print("\n")
        
    
        
    
        
       
        
   
        
        
    
    
    
    