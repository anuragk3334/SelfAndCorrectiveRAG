from langchain.messages import HumanMessage, SystemMessage
import json
from core.llm import llm
from core.vectorstore import query_pinecone


def generate_variants(query):
    system = SystemMessage(content="""Generate exactly 4 different ways to rephrase the user's question.
Return ONLY a JSON array of exactly 4 strings, like: ["rephrased1", "rephrased2", "rephrased3", "rephrased4"]
No explanations, no extra text.""")

    human = HumanMessage(content=f"Question: {query}")
    response = llm.invoke([system, human])
    variants = json.loads(response.content.strip())
    return [query] + variants  # Include original + 4 variants = 5 total


def retrieve_for_all_queries(all_queries, top_k=5):
    # For each query, get top_k chunks from pinecone
    # Store as list of lists (each inner list = ranked results for one query)
    ranked_lists = []
    for q in all_queries:
        chunks = query_pinecone(q, k=top_k)
        ranked_lists.append(chunks)
    return ranked_lists


def apply_rrf(ranked_lists, k=60):
    # RRF Score = sum of  1 / (rank + k)  for every list the doc appears in
    # Higher score = more relevant

    scores = {}
    doc_map = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_key = doc.page_content[:150]   # use content as unique key
            scores[doc_key] = scores.get(doc_key, 0) + 1 / (rank + k)
            doc_map[doc_key] = doc

    # sort by score descending
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

    print("\n--- RRF Scores ---")
    for key in sorted_keys:
        print(f"Score: {scores[key]:.4f} | {key[:80]}...")

    return sorted_keys, scores, doc_map


def get_top_chunks(sorted_keys, doc_map, final_k=5):
    return [doc_map[key] for key in sorted_keys[:final_k]]


def generate_answer(query, top_chunks):
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', '?')} | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in top_chunks
    )
    prompt = f"""Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""
    return llm.invoke(prompt).content.strip()

def retrieve(query: str,top_k:int=5) -> list[dict]:
    # Step 1 - generate query variants
    variants = generate_variants(query)
    all_queries = [query] + variants
    print(f"All queries: {all_queries}")

    # Step 2 - retrieve chunks for all queries
    ranked_lists = retrieve_for_all_queries(all_queries)

    # Step 3 - apply RRF to rank and deduplicate
    sorted_keys, scores, doc_map = apply_rrf(ranked_lists)

    # Step 4 - pick top chunks and convert to list of dicts
    top_chunks = get_top_chunks(sorted_keys, doc_map, final_k=top_k)
    print(f"\nSending {len(top_chunks)} chunks to LLM")

    result = []
    for key, chunk in zip(sorted_keys[:top_k], top_chunks):
        result.append({
            "content": chunk.page_content,
            "source": chunk.metadata.get("source", "?"),
            "page": chunk.metadata.get("page", "?"),
            "rrf_score": scores[key]
        })

    return result


if __name__ == "__main__":
    query = "which diabetes pill is good for the heart?"

    # Step 1 - generate query variants
    variants = generate_variants(query)
    all_queries = [query] + variants
    print(f"All queries: {all_queries}")

    # Step 2 - retrieve chunks for all queries
    ranked_lists = retrieve_for_all_queries(all_queries)

    # Step 3 - apply RRF to rank and deduplicate
    sorted_keys, scores, doc_map = apply_rrf(ranked_lists)

    # Step 4 - pick top 5 chunks
    top_chunks = get_top_chunks(sorted_keys, doc_map, final_k=5)
    print(f"\nSending {len(top_chunks)} chunks to LLM")

    # Step 5 - generate answer
    answer = generate_answer(query, top_chunks)
    print(f"\nAnswer: {answer}")
