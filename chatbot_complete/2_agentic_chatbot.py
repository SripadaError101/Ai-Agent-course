import importlib.util
from pathlib import Path

import chainlit as cl
import dotenv
from openai.types.responses import ResponseTextDeltaEvent

from agents import Runner

nutrition_module_path = Path(__file__).resolve().with_name("nutrition_agent.py")
spec = importlib.util.spec_from_file_location(
    "_chatbot_complete_nutrition_agent", nutrition_module_path
)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load nutrition agent from {nutrition_module_path}")
nutrition_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nutrition_agent_module)
nutrition_agent = nutrition_agent_module.nutrition_agent

dotenv.load_dotenv()


@cl.on_message
async def on_message(message: cl.Message):

    result = Runner.run_streamed(
        nutrition_agent,
        message.content,
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
                    f"\nTool call: {event.data.item.name} "
                    f"with args: {event.data.item.arguments}"
                )

    await msg.update()
