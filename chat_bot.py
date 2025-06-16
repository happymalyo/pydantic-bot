# Chatbot simple
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables
load_dotenv()

# Initialize the OpenAI model
model = OpenAIModel("gpt-4o-mini")

# Create a basic agent
agent = Agent(
    model,
    system_prompt="You are a helpful and friendly chatbot assistant. Be conversational and helpful.",
)


def main():
    print("🤖 Chatbot Agent Started! (Type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            # Get response from the agent
            result = agent.run_sync(user_input)
            print(f"🤖 Bot: {result.output}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
