# setup pydantic model 

from langchain_groq import ChatGroq
from pydantic import   BaseModel 
from typing import List 
from ai_agent import get_response_from_ai_agent 

class RequestState(BaseModel):
    model_name:str 
    model_provider:str
    system_prompt:str 
    messages:List[str]
    allow_search:bool 


from fastapi import FastAPI 

ALLOWED_MODEL_NAMES=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

app=FastAPI(title="Langchain AI Agent ")

@app.post("/chat")

def chat_endpoint(request:RequestState):
    """
    Process incoming chat request
    """

    if request.model_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"model {request.model_name} not allowed")

    result = get_response_from_ai_agent(
        llm_id = request.model_name,
        query = request.messages,
        allow_search = request.allow_search,
        system_prompt= request.system_prompt,
        provider = request.model_provider,
    )
    return {"response":result} 

if __name__=="__main__":
    import uvicorn 
    uvicorn.run(app,host="127.0.0.1",port=8000) 





























