import json

from core.llm import llm
from langchain.messages import SystemMessage, HumanMessage, AIMessage

def router(query:str)->dict:
    """
    Classify the query and return the routing decisions.
    
    Returns:
        strategy: no_retrieval |single| rag_fusion
        complexity: simple|moderate|complex
        reason: why this strategy was chosen
    
    """
    
    systemMessage = SystemMessage(content="""You are a helpful assistant that classifies user queries for routing in a RAG system. Classify the query into one of three strategies: no_retrieval, single, or rag_fusion. Also classify the complexity of the query as simple, moderate, or complex. Provide a brief reason for your classification.
                      Classify into ONE strategy:

"no_retrieval"  — general knowledge, definitions, basic medical terms
                      the LLM knows without looking anything up
                      Example: "What does insulin do?"

"single"        — specific factual question about one topic
                      one retrieval pass will find the answer
                      Example: "What is the HbA1c target for elderly patients?"

"rag_fusion"    — informal language, vague phrasing, or a concept
                      expressible in many different ways
                      Example: "which diabetes pill is good for the heart?"


Return JSON only:
{
  "strategy": "no_retrieval|single|rag_fusion|decompose",
  "complexity": "simple|moderate|complex",
  "reason": "one sentence"

}""")

    humanMessage = HumanMessage(content=f"query :{query}")
    
    response = llm.invoke([systemMessage,humanMessage])
    content=response.content
    print(f"Router response from Adaptive RAG is: {content}")
    routing=json.loads(content)
    print(f"Router response from Adaptive RAG in json is: {routing}")
    return routing

if __name__=="__main__":
    query="What is insulin?"
    routing=router(query)   
    print(f"Routing decision: {routing}")