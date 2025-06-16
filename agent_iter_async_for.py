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


async def main():
    nodes = []
    async with agent.iter("What is the capital of the UK?") as agent_run:
        async for node in agent_run:
            # each node represent agent step execution
            nodes.append(node)
    # print(nodes)
    print(agent_run.result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
