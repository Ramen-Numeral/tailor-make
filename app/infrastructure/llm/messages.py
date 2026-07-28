"""LangChain message conversion utilities.

Handles conversion between application-level prompts (string) and
LangChain message formats. Centralizes message construction so it can
be easily adapted if needed.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def build_messages(
    prompt: str,
    system_prompt: str | None = None,
) -> list[BaseMessage]:
    """Build a list of LangChain messages from prompt strings.

    Args:
        prompt: The user message / main prompt.
        system_prompt: Optional system message to prepend.

    Returns:
        List of BaseMessage objects ready for LangChain.
    """
    messages: list[BaseMessage] = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    messages.append(HumanMessage(content=prompt))

    return messages
