from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import Dict, Any
from tavily import TavilyClient


load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
checkpointer = InMemorySaver()

model = init_chat_model("meta-llama/llama-4-scout-17b-16e-instruct",
                        model_provider = "groq",
                        temperature = 0.7)

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str
    
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    response = tavily_client.search(query)
    results = response.get("results", [])
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in results[:3]
    )

agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    context_schema=Context,
    tools=[web_search]
)

config = {"configurable": {"thread_id": "1"}}

print("🤖 Agent is ready! Type 'exit' to stop.\n")

# conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Agent: Goodbye 👋")
        break

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    print("Agent:", response["messages"][-1].content)