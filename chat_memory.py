# Chatbot with memory
import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

load_dotenv()


# Define conversation context
class ConversationContext(BaseModel):
    conversation_history: List[dict[str, str]] = []
    user_name: str = "User"
    session_start: datetime = datetime.now()


# Initialize the OpenAI model
model = OpenAIModel("gpt-4o-mini")

# Create enhanced agent with context
agent = Agent(
    model,
    deps_type=ConversationContext,
    system_prompt=(
        "You are a helpful chatbot assistant named Ekraw. "
        "You have access to the conversation history and can reference previous messages. "
        "Be personable, remember what the user has told you, and maintain context throughout the conversation."
    ),
)


@agent.system_prompt
def add_context_to_prompt(ctx: RunContext[ConversationContext]) -> str:
    history_summary = ""
    if ctx.deps.conversation_history:
        recent_history = ctx.deps.conversation_history[-5:]  # Last 5 exchanges
        history_summary = "\n\nRecent conversation context:\n"
        for exchange in recent_history:
            history_summary += f"User: {exchange['user']}\nYou: {exchange['bot']}\n"

    return f"""You are Ekraw, a friendly chatbot assistant.
    User's name: {ctx.deps.user_name}
    Session started: {ctx.deps.session_start.strftime('%Y-%m-%d %H:%M')}
    {history_summary}
    
    Respond naturally and reference previous conversation when relevant."""


def main():
    print("🤖 Enhanced Chatbot (Alex) Started!")
    print("💡 I'll remember our conversation context")
    print("-" * 50)

    # Initialize conversation context
    context = ConversationContext()

    # Get user's name
    name = input("👤 What's your name? ").strip()
    if name:
        context.user_name = name
        print(f"Nice to meet you, {name}!")

    while True:
        user_input = input(f"\n👤 {context.user_name}: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 It was great chatting with you! Goodbye!")
            break

        if not user_input:
            continue

        try:
            # Get response from agent with context
            result = agent.run_sync(user_input, deps=context)
            bot_response = result.output

            print(f"🤖 Ekraw: {bot_response}")

            # Update conversation history
            context.conversation_history.append(
                {"user": user_input, "bot": bot_response}
            )

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
