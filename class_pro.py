import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel,set_tracing_disabled, AsyncOpenAI, function_tool
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent
from tavily import TavilyClient
from agents.extensions.visualization import draw_graph

set_tracing_disabled(disabled=True)
               
load_dotenv(find_dotenv())

travily_api_key=os.getenv("TRAVILY_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")


# Step-1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

# Step-2: Model
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash-exp",
    openai_client = provider,
)

web_development_agent = Agent(
    name="web development agent",
    model=model,
    handoff_description="Specialist Agent for web development",
    instructions="""you provide help in web development"""
)       

mobile_development_agent = Agent(
    name="mobile development agent",
    model=model,
    handoff_description="Specialist Agent for mobile development",
    instructions="""you provide help mobile web development"""
)    
planner_agent= Agent(
    name="planner_agent",
    model=model,
    handoff_description="Specialist Agent for planner",
    instructions="""You provide help planning the project """
)

planner_agent_tool=planner_agent.as_tool(
    tool_name="planner_agent_tool",
    tool_description="It pane the hole process carefully"
)

agenticAI_agent = Agent(
    name="agenticAI agent",
    model=model,
    handoff_description="Specialist Agent for agentic AI ",
    instructions="""you provide help in agentic AI""",
    tools=[planner_agent]
)            




panacloud = Agent(
    name="panacloud developer agent",
    model=model,
    handoff_description="Specialist Agent for panacloud for web",
    instructions="""you provide help for panacloud help in web development""",
    handoffs=[web_development_agent,mobile_development_agent,agenticAI_agent]
)      


draw_graph(panacloud, filename="agent_graph.png")

# Step-5: Start chat

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am the full-stack developer support agent. How can I help you today?").send()

# Step-6: Runner

@cl.on_message  # decorator function
async def main(message: cl.Message):

    # Show something on the screen
    msg = cl.Message(
        content="Thinking....\n\n",
    )
    await msg.send()

    # Get history
    history = cl.user_session.get("history")  
   
    # Add user message in history 
    history.append({"role": "user", "content": message.content})  
 

    agent_response = Runner.run_streamed(panacloud, history)
    async for event in agent_response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            raw_text = event.data.delta
            await msg.stream_token(raw_text)


    # Add agent message in history
    history.append(
        {"role": "assistant", "content": agent_response.final_output})
    
    # Setting the new history
    cl.user_session.set("history", history)