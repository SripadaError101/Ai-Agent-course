import importlib.util
import os
from pathlib import Path

import chainlit as cl
import dotenv
from agents import InputGuardrailTripwireTriggered, Runner, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent

if __package__ in (None, ""):
    nutrition_module_path = Path(__file__).resolve().with_name("nutrition_agent.py")
    spec = importlib.util.spec_from_file_location(
        "_multi_agent_nutrition_agent", nutrition_module_path
    )
    nutrition_agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nutrition_agent_module)
    exa_search_mcp = nutrition_agent_module.exa_search_mcp
    nutrition_agent = nutrition_agent_module.nutrition_agent
else:
    from .nutrition_agent import exa_search_mcp, nutrition_agent

dotenv.load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    session = SQLiteSession("conversation_history")
    cl.user_session.set("agent_session", session)
    # This is the only change in this file compared to the chatbot/agentic_chatbot.py file
    await exa_search_mcp.connect()


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")

    result = Runner.run_streamed(
        nutrition_agent,
        message.content,
        session=session,
    )

    msg = cl.Message(content="")
    async for event in result.stream_events():
        # Stream final message text to screen
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)
            print(event.data.delta, end="", flush=True)

        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and len(event.data.item.arguments) > 0
        ):
            with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                step.input = event.data.item.arguments
                print(
                    f"\nTool call: {
                        event.data.item.name} with args: {
                        event.data.item.arguments}"
                )

    await msg.update()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == (
        os.getenv("CHAINLIT_USERNAME"),
        os.getenv("CHAINLIT_PASSWORD"),
    ):
        return cl.User(
            identifier="Student",
            metadata={"role": "student", "provider": "credentials"},
        )
    else:
        return None
