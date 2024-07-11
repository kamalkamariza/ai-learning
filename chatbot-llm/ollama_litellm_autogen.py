from openai import OpenAI
from autogen import AssistantAgent, UserProxyAgent

config_list = [
    # {
    #     "model": "llama3",
    #     "base_url": "http://localhost:11434/v1",
    #     "api_key": "ollama",
    # },
    {
        "model": "NotRequired",  # Loaded with LiteLLM command
        "api_key": "NotRequired",  # Not needed
        "base_url": "http://0.0.0.0:4000",  # Your LiteLLM URL
    },
]

helper = AssistantAgent(
    name="helper",
    max_consecutive_auto_reply=5,
    system_message="""
    You are a chatbot that answers user's question.
    Return TERMINATE after you have answered the user's question.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0,
    },
    human_input_mode="NEVER",
)

agent = UserProxyAgent(
    name="agent",
    max_consecutive_auto_reply=5,
    code_execution_config=False,
    system_message="You are a medium for user to ask question",
    llm_config={
        "config_list": config_list,
        "temperature": 0,
    },
    is_termination_msg=lambda content: content.get("content")
    and ("TERMINATE" in content.get("content") or "?" in content.get("content")),
    human_input_mode="NEVER",
)

from typing import Annotated, Literal

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")


# Register the tool signature with the assistant agent.
helper.register_for_llm(name="calculator", description="A simple calculator")(
    calculator
)

# Register the tool function with the user proxy agent.
agent.register_for_execution(name="calculator")(calculator)

agent.initiate_chat(helper, message="What is (44232 + 13312 / (232 - 32)) * 5?")
