"""Model runtime layer (ADR 6.2 C3, 18). Model-agnostic by design (ADR 25)."""

from mimir.llm.base import (
    ChatModel,
    ChatResponse,
    ChunkType,
    EchoModel,
    GenerationOptions,
    LLMMessage,
    ModelError,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)
from mimir.llm.openai_compat import OpenAICompatModel, build_model
from mimir.llm.router import ModelRouter, TaskClass, extract_json, get_router

__all__ = [
    "ChatModel",
    "ChatResponse",
    "ChunkType",
    "EchoModel",
    "GenerationOptions",
    "LLMMessage",
    "ModelError",
    "ModelRouter",
    "OpenAICompatModel",
    "Role",
    "StreamChunk",
    "TaskClass",
    "ToolCall",
    "Usage",
    "build_model",
    "extract_json",
    "get_router",
]
