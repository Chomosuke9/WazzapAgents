# File: python/bridge/llm/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


LLM1_SCHEMA = {
  "name": "llm_should_response",
  "parameters": {
    "type": "object",
    "properties": {
      "should_response": {
        "type": "boolean",
        "description": "Indicates whether the LLM should respond (true) or not (false).",
      },
      "confidence": {
        "type": "integer",
        "description": "Confidence percentage (0-100) about the decision.",
        "minimum": 0,
        "maximum": 100,
      },
      "reason": {
        "type": "string",
        "description": (
          "A concise routing reason for downstream handoff. "
          "Write 1-3 short sentences (target 12-60 words) grounded in current context, "
          "without chain-of-thought."
        ),
        "minLength": 2,
        "maxLength": 320,
      },
    },
    "required": ["should_response", "confidence", "reason"],
    "additionalProperties": False,
  },
}

LLM1_TOOL = {
  "type": "function",
  "function": {
    "name": LLM1_SCHEMA["name"],
    "description": "Decide whether the WhatsApp agent should respond to the latest message.",
    "parameters": LLM1_SCHEMA["parameters"],
    "strict": True,
  },
}

LLM1_EXPRESS_SCHEMA = {
  "name": "llm_express",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": (
          "Either a single emoji to react to the message (e.g. 👍, 😂, ❤️, 🔥, 😢), "
          "or the exact sticker name from the available sticker catalog to send a sticker. "
          "Use an emoji when a quick reaction fits; use a sticker name when a sticker would be more expressive."
        ),
        "minLength": 1,
        "maxLength": 100,
      },
      "context_msg_id": {
        "type": "string",
        "description": (
          "The 6-digit contextMsgId of the target message. "
          "Use the id from current messages(burst). "
          "Use the last message id if targeting the most recent message."
        ),
        "minLength": 6,
        "maxLength": 6,
      },
      "confidence": {
        "type": "integer",
        "description": "Confidence percentage (0-100) about this decision.",
        "minimum": 0,
        "maximum": 100,
      },
      "reason": {
        "type": "string",
        "description": (
          "A concise reason for this action. "
          "1-2 short sentences (max 320 chars)."
        ),
        "minLength": 2,
        "maxLength": 320,
      },
    },
    "required": ["expression", "context_msg_id", "confidence", "reason"],
    "additionalProperties": False,
  },
}

LLM1_REACT_TOOL = {
  "type": "function",
  "function": {
    "name": LLM1_EXPRESS_SCHEMA["name"],
    "description": (
      "Express a non-text reaction to a message — either an emoji reaction or a sticker — "
      "instead of sending a text reply. "
      "Use this when the situation calls for a lightweight acknowledgement with no text needed."
    ),
    "parameters": LLM1_EXPRESS_SCHEMA["parameters"],
    "strict": True,
  },
}

LLM1_TOOLS = [LLM1_TOOL, LLM1_REACT_TOOL]


class LLM1Decision(BaseModel):
  should_response: bool = Field(..., description="Whether to respond")
  confidence: int = Field(..., ge=0, le=100)
  reason: str = Field(..., min_length=2, max_length=320)
  react_expression: str | None = Field(default=None, description="Emoji or sticker name for express-only decisions")
  react_context_msg_id: str | None = Field(default=None, description="Target message contextMsgId for react-only")
  input_tokens: int = Field(default=0, description="LLM1 input tokens used")
  output_tokens: int = Field(default=0, description="LLM1 output tokens used")
