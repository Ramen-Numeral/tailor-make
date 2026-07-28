"""LLM provider adapters.

Each module implements an adapter for a specific LLM provider.
All adapters implement the application LLMClient protocol.

Modules:
    - groq: ChatGroq adapter
    - ollama: ChatOllama adapter

These are internal to the LLM infrastructure layer.
Application code should use app.infrastructure.llm instead.
"""
