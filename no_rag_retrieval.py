from langchain.messages import SystemMessage

from core.llm import llm
from rag_adaptive import router     

def answer(query:str)->str:
    """Answer the query directly from LLM without retrievel"""
    
    systemMessage = SystemMessage(content="""You are a helpful assistant that answers user queries based on your general knowledge. If you don't know the answer, say "I don't have that information." Do not attempt to make up an answer if you are unsure.""")
    humanMessage = SystemMessage(content=f"query :{query}")
    
    response = llm.invoke([systemMessage,humanMessage])
    return response.content.strip()
    