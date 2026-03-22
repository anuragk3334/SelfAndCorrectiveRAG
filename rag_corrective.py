#RAG corrective is used for corrective measures , when IsRel or IsSup fails , instead of giving up , the system takes a corrective action and try to fix 
from langchain.messages import HumanMessage, SystemMessage

from core.llm import llm
from core.vectorstore import query_pinecone


def corrective_isrel(query:str,retry_count:int ,index,embeddings)->list[dict]:
    """Corrective action for relevance judgement failure.

    Args:
        query (str): The user query.
        retry_count (int): Number of retries attempted.
        index: The vector store index.
        embeddings: The embedding model.
        
    Fires When IsRel has no relative chunks
    Retry 0->search wider(larger k)
    Retry 1-> Rewrite quesy and search again
    Retry 2 -> give up , Esclate or HIL
    Returns:
        list[dict]: List of relevant chunks after corrective action.
    """
    print(f"Relevance judgement failed for query: '{query}'. Attempting corrective action (Retry {retry_count})...")
    
    if retry_count==0:
        # First retry: search wider by increasing k
        
        new_chunks = query_pinecone(query, k=20)
        return new_chunks
    
    elif retry_count==1:
        systemMessage=SystemMessage(content="""Rewrite this diabetes query using 
formal clinical/medical terminology to improve document retrieval.
Return ONLY the rewritten query, nothing else.""" )
        humanMessage=HumanMessage(content=f"Original query: {query}")
        rewritten_query=llm.invoke([systemMessage,humanMessage]).content.strip()
        new_chunks = query_pinecone(rewritten_query, k=15)
        return new_chunks
    else:
         print("Maximum retries reached. Unable to find relevant information for the query.")
         return []
    

def corrective_issup(query, response, chunks, unsupported_claims, retry_count):

    print(f"\n[Corrective-IsSup] Attempt number: {retry_count + 1}")

    # ─────────────────────────────────────────────────────────
    # STEP 1: Build the context string from top 3 chunks
    # These chunks are the SOURCE OF TRUTH from your PDF
    # The LLM must stay within what these chunks say
    # ─────────────────────────────────────────────────────────

    context_parts = []

    for c in chunks[:3]:
        context_parts.append(c["content"])

    context = "\n\n".join(context_parts)

    print(f"[Corrective-IsSup] Context built from {len(context_parts)} chunks")
    print(f"[Corrective-IsSup] Context length: {len(context)} characters")


    # ─────────────────────────────────────────────────────────
    # STEP 2: Decide which fix to use
    #
    # SURGICAL FIX   → when we know exactly what is wrong
    #                  and this is the first attempt
    #
    # FULL REGENERATE → when surgical fix already failed
    #                   OR we don't know what is wrong
    # ─────────────────────────────────────────────────────────

    there_are_bad_claims = len(unsupported_claims) > 0
    this_is_first_attempt = retry_count == 0

    if there_are_bad_claims and this_is_first_attempt:

        # ─────────────────────────────────────────────────────
        # SURGICAL FIX
        # We know exactly which sentences are wrong
        # Just remove those sentences, keep everything else
        # ─────────────────────────────────────────────────────

        print(f"[Corrective-IsSup] Strategy: SURGICAL FIX")
        print(f"[Corrective-IsSup] Number of bad claims to remove: {len(unsupported_claims)}")

        # Build a bullet list of bad claims to show the LLM
        bad_claims_text = ""
        for claim in unsupported_claims:
            bad_claims_text = bad_claims_text + "- " + claim + "\n"

        print(f"[Corrective-IsSup] Bad claims:\n{bad_claims_text}")

        # Build the full message to send to the LLM
        human_message_text = (
            "Here is the response that has some wrong claims:\n"
            + response
            + "\n\n"
            + "These specific claims are NOT supported by the source document.\n"
            + "Remove ONLY these claims from the response:\n"
            + bad_claims_text
            + "\n"
            + "Here is the source document (the truth):\n"
            + context
        )

        # Call the LLM to do the surgical fix
        result = llm.invoke([
            SystemMessage(content=(
                "You are a medical editor. "
                "Your job is to remove specific wrong claims from a response. "
                "Remove ONLY the listed claims. "
                "Keep all other sentences exactly the same. "
                "Return only the corrected response text."
            )),
            HumanMessage(content=human_message_text)
        ])

        fixed_response = result.content

        print(f"[Corrective-IsSup] Surgical fix done")
        print(f"[Corrective-IsSup] Original length: {len(response)} chars")
        print(f"[Corrective-IsSup] Fixed length:    {len(fixed_response)} chars")

        return fixed_response

    else:

        # ─────────────────────────────────────────────────────
        # FULL REGENERATION
        # Happens when:
        #   - surgical fix already failed (retry_count is 1 or 2)
        #   - OR we don't know what exactly is wrong
        #
        # Throw away the bad response completely
        # Write a brand new answer from scratch
        # Use a MUCH stricter prompt this time
        # ─────────────────────────────────────────────────────

        if retry_count == 0:
            print(f"[Corrective-IsSup] Strategy: FULL REGENERATION (no specific claims identified)")
        else:
            print(f"[Corrective-IsSup] Strategy: FULL REGENERATION (surgical fix already failed)")

        # Build the message for full regeneration
        human_message_text = (
            "Source documents (use ONLY information from here):\n"
            + context
            + "\n\n"
            + "Question: "
            + query
        )

        # Call the LLM with a much stricter system prompt
        result = llm.invoke([
            SystemMessage(content=(
                "You are a diabetes clinical assistant. "
                "You must answer using ONLY the exact information "
                "from the source documents provided. "
                "Do NOT add any information from your general knowledge. "
                "Do NOT guess or assume anything. "
                "If the source documents do not contain enough information "
                "to answer the question, say exactly this: "
                "'The document does not contain enough information "
                "to fully answer this question.'"
            )),
            HumanMessage(content=human_message_text)
        ])

        new_response = result.content

        print(f"[Corrective-IsSup] Full regeneration done")
        print(f"[Corrective-IsSup] New response length: {len(new_response)} chars")

        return new_response
