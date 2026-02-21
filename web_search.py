from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.messages import HumanMessage

load_dotenv()
checkpointer = InMemorySaver()

model = init_chat_model("meta-llama/llama-4-scout-17b-16e-instruct",
                        model_provider = "groq",
                        temperature = 0.7)

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    context_schema=Context
)

config = {"configurable": {"thread_id": "1"}}

print("🤖 Agent is ready! Type 'exit' to stop.\n")

# conversation loop
while True:
    
    # take user input
    user_input = input("You: ")
    
    # exit condition
    if user_input.lower() == "exit":
        print("Agent: Goodbye 👋")
        break

    question = HumanMessage(content = user_input )

    response = agent.invoke(
        {"messages":[question]},
        config=config,
        context=Context(user_id="1")
    )
    print("Agent:",response["messages"][-1].content)