# step 1 : Setup API Keys for Groq  

from langchain_core.messages import AIMessage
import os 
from dotenv import load_dotenv 

load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")
# step 2 : LLMS & tools setup 

from langchain_groq import ChatGroq 
from langchain_community.tools.tavily_search import TavilySearchResults 



llm = ChatGroq(model="llama-3.1-8b-instant")

# search_tool=TavilySearchResults(max_results=2)
# tools=[search_tool]


# step 3 : Setup ai agent with Search Tool Functionality 

from langgraph.prebuilt import create_react_agent 

system_prompt= "Act as an AI chatbot who is smart and friendly "

def get_response_from_ai_agent(llm_id,query,allow_search,system_prompt,provider):
    # Select Model provider 
    provider_lower = provider.lower()
    if provider_lower == "groq":
        llm = ChatGroq(model=llm_id)

    else:
        raise ValueError(f"provider {provider} not found")

    tools=[TavilySearchResults(max_results=2)] if allow_search else []

    agent=create_react_agent(model=llm,tools=tools,prompt=system_prompt)  

    if isinstance(query, list):
        formatted_messages = []
        for i, msg in enumerate(query):
            role = "user" if i % 2 == 0 else "assistant"
            formatted_messages.append({"role": role, "content": msg})
    else:
        formatted_messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": formatted_messages})
    messages = response.get("messages", [])
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1] if ai_messages else "No response generated."

if __name__ == "__main__":
    print(get_response_from_ai_agent("llama-3.1-8b-instant", "What is the current weather in New York?", True, system_prompt, "groq"))

 






