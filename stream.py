import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel,set_tracing_disabled, AsyncOpenAI
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

set_tracing_disabled(disabled=True)
               
load_dotenv(find_dotenv())

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



# Step-4: Agent
frontend_agent = Agent(
    name="frontend developer",
    model=model,
    handoff_description="Specialist Agent for frontend developement",
    instructions="""You provide help with:
                 - HTML for structuring web pages
                 - CSS / Tailwind / Bootstrap for styling
                 - JavaScript for interactivity
                 - Frameworks/Libraries like React, Next.js, Vue, Angular

                 Explain your reasoning at each step and include examples."""
)         

backend_agent = Agent(
    name="backend developer",
    model=model,
    handoff_description="Specialist Agent for backend developement",
    instructions="""You provide help for Languages: JavaScript (Node.js), Python (Django/Flask), PHP, Ruby, Java, etc.
                     Databases: SQL (MySQL, PostgreSQL), NoSQL (MongoDB),
                     Server/Hosting: Express.js (Node.js), Apache, Nginx
                     """
)

stripe_agent=Agent(
    name="Stripe payement agent",
    model=model,
    handoff_description="Specialist Agent for stripe payement method",
    instructions="You provide help stripe payement integration to get payment from users."
)

triage_agent = Agent(
    name="Triage Agent",
    model=model,
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[frontend_agent, backend_agent,stripe_agent]
)

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
        content="Thinking....",
    )
    await msg.send()

    # Get history
    history = cl.user_session.get("history")  
   
    # Add user message in history 
    history.append({"role": "user", "content": message.content})  
 

    agent_response = Runner.run_streamed(triage_agent, history)
    async for event in agent_response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            raw_text = event.data.delta
            await msg.stream_token(raw_text)


    # Add agent message in history
    history.append(
        {"role": "assistant", "content": agent_response.final_output})
    
    # Setting the new history
    cl.user_session.set("history", history)