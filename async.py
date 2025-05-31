import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, AsyncOpenAI
import chainlit as cl

set_tracing_disabled(disabled=True)
               
load_dotenv(find_dotenv())

gemini_api_key=os.getenv("GEMINI_API_KEY")

# STEP-1 
provider=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# step-2 
model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=provider
)

# Step-3 
agent=Agent(
    name="frontend developer",
    instructions="You are the senior frontend developer specislist in next.js with five years experience" ,
    model=model
)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am frontend website maker and problems Support Agent. How we can help you today?").send()


@cl.on_message  # decorator function
async def main(message: cl.Message):

    # Show something on the screen
    msg = cl.Message(
        content="Thinking.....",
    )
    await msg.send()

    # Get history
    history = cl.user_session.get("history")  
   
    # Add user message in history 
    history.append({"role": "user", "content": message.content})  
 

    agent_response =await Runner.run(agent, history)
    
    print(agent_response.final_output)
    # Add agent message in history
    history.append(
        {"role": "assistant", "content": agent_response.final_output})
    
    # Setting the new history
    cl.user_session.set("history", history)