import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import requests
import json
from datetime import datetime

load_dotenv()


class AgentContext(BaseModel):
    user_name: str = "User"


model = OpenAIModel("gpt-4o-mini")

# Create specialized agent
agent = Agent(
    model,
    deps_type=AgentContext,
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "You can help with calculations, provide current time, and answer questions. "
        "Use the available tools when appropriate."
    ),
)


result_sync = agent.run_sync("What is the capital of Italy?")
print(result_sync.output)
# > Rome


async def main():
    result = await agent.run("What is the capital of France?")
    print(result.output)
    # > Paris

    async with agent.run_stream("What is the capital of the UK?") as response:
        print(await response.get_output())
        # > London


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
