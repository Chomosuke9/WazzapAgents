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


# ---------------------------------------------------------------------------
# LLM2 tool schemas
# ---------------------------------------------------------------------------

LLM2_REPLY_TOOL = {
  "type": "function",
  "function": {
    "name": "reply_message",
    "description": (
      "Send a text reply. Use context_msg_id to reply to a specific message, "
      "or 'none' to send without quoting. Inline mentions as @Name (senderRef). "
      "You can execute command with this tool too."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "context_msg_id": {
          "type": "string",
          "description": (
            "The 6-digit contextMsgId to reply to, or 'none' for a standalone message."
          ),
        },
        "text": {
          "type": "string",
          "description": "The reply text or command.",
          "minLength": 1,
        },
      },
      "required": ["context_msg_id", "text"],
      "additionalProperties": False,
    },
    "strict": True,
  },
}

LLM2_EXPRESS_TOOL = {
  "type": "function",
  "function": {
    "name": "llm_express",
    "description": (
      "Express a non-text reaction to a message — either an emoji reaction or a sticker — "
      "instead of sending a text reply."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "context_msg_id": {
          "type": "string",
          "description": "The 6-digit contextMsgId to target.",
          "minLength": 6,
          "maxLength": 6,
        },
        "expression": {
          "type": "string",
          "description": (
            "Either a single emoji to react with (e.g. 👍, 😂, ❤️), "
            "or the exact sticker name from the <sticker> catalog."
          ),
          "minLength": 1,
          "maxLength": 100,
        },
      },
      "required": ["context_msg_id", "expression"],
      "additionalProperties": False,
    },
    "strict": True,
  },
}

LLM2_DELETE_TOOL = {
  "type": "function",
  "function": {
    "name": "delete_messages",
    "description": (
      "Delete one or more messages by their contextMsgId. "
      "Only use when messages clearly violate rules."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "context_msg_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 6,
            "maxLength": 6,
          },
          "description": "List of 6-digit contextMsgIds to delete.",
          "minItems": 1,
        },
      },
      "required": ["context_msg_ids"],
      "additionalProperties": False,
    },
  },
}

LLM2_MUTE_TOOL = {
  "type": "function",
  "function": {
    "name": "mute_member",
    "description": (
      "Mute or unmute a member. "
      "Set duration_minutes > 0 to mute (auto-delete all their messages for that duration). "
      "Set duration_minutes = 0 to unmute (cancel an active mute). "
      "Use mute for persistent rule violators."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "sender_ref": {
          "type": "string",
          "description": "The senderRef of the member to mute or unmute.",
          "minLength": 1,
        },
        "anchor_context_msg_id": {
          "type": "string",
          "description": "The 6-digit contextMsgId of a recent message from this member.",
          "minLength": 6,
          "maxLength": 6,
        },
        "duration_minutes": {
          "type": "integer",
          "description": "How long to mute in minutes (1-1440). Use 0 to unmute.",
          "minimum": 0,
          "maximum": 1440,
        },
      },
      "required": ["sender_ref", "anchor_context_msg_id", "duration_minutes"],
      "additionalProperties": False,
    },
  },
}

LLM2_KICK_TOOL = {
  "type": "function",
  "function": {
    "name": "kick_members",
    "description": (
      "Remove members from the group. Cannot kick admins. "
      "Only use for serious or repeated violations."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "targets": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "sender_ref": {
                "type": "string",
                "description": "The senderRef of the member to kick.",
              },
              "anchor_context_msg_id": {
                "type": "string",
                "description": "A 6-digit contextMsgId from this member for validation.",
                "minLength": 6,
                "maxLength": 6,
              },
            },
            "required": ["sender_ref", "anchor_context_msg_id"],
            "additionalProperties": False,
          },
          "description": "List of members to kick.",
          "minItems": 1,
        },
      },
      "required": ["targets"],
      "additionalProperties": False,
    },
  },
}

LLM2_SUBAGENT_TOOL = {
  "type": "function",
  "function": {
    "name": "execute_subtask",
    "description": (
      "Delegate a complex task to a sub-agent for execution. "
      "The sub-agent will process the instruction and return a report. "
      "Use this for tasks that require multi-step reasoning, file processing, "
      "or operations that are too complex for a single LLM call. "
      "Any output files the sub-agent produces are automatically attached and "
      "sent to the chat after your text reply, one file per WhatsApp message — "
      "you do not need to mention file paths or upload them yourself. "
      "Make the instruction precise so the sub-agent only emits files the user "
      "actually wants delivered. "
      "Set high_quality=true for tasks requiring deeper reasoning, complex code "
      "generation, or analysis; set high_quality=false (default) for routine tasks "
      "like format conversion or simple scripting."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "instruction": {
          "type": "string",
          "description": "Clear, detailed instruction for the sub-agent to execute.",
          "minLength": 1,
        },
        "confirmation_text": {
          "type": "string",
          "description": (
            "A brief confirmation message to the user that the task has started. "
            "If input files are provided via context_msg_ids, this message will "
            "be sent as a reply to the last file ID to acknowledge receipt."
          ),
          "minLength": 1,
        },
        "context_msg_ids": {
          # OpenAI strict-mode forbids "optional" properties: every key in
          # `properties` MUST also appear in `required`. To keep this field
          # semantically optional, it accepts `null` as a value. Callers
          # that want to provide nothing send `null` (which downstream
          # action extraction normalises back to `[]` via `or []` in
          # messaging/actions.py::_extract_actions_from_tool_calls).
          "type": ["array", "null"],
          "items": {
            "type": "string",
            "minLength": 6,
            "maxLength": 6,
          },
          "description": (
            "List of 6-digit contextMsgIds whose messages contain media attachments "
            "or text content to provide as input to the sub-agent. The bridge resolves "
            "each ID to the corresponding file path automatically — for text-only messages, "
            "the text is converted to a .txt file. Only include IDs that are explicitly "
            "relevant to the instruction. Pass null when no input files are needed."
          ),
        },
        "high_quality": {
          "type": "boolean",
          "description": (
            "Set to true to use a higher-capability model for tasks requiring "
            "deeper reasoning, complex analysis, or code generation. "
            "Defaults to false for routine tasks like format conversion, "
            "simple lookups, or basic scripting."
          ),
        },
      },
      # Strict mode: every property name must be listed in `required`.
      # See note on `context_msg_ids` above for how optionality is modeled
      # via `["array", "null"]` instead of omitting the key.
      "required": ["instruction", "confirmation_text", "context_msg_ids", "high_quality"],
      "additionalProperties": False,
    },
    "strict": True,
  },
}

# Base tools always available to LLM2.
LLM2_BASE_TOOLS = [LLM2_REPLY_TOOL, LLM2_EXPRESS_TOOL]


def build_llm2_tools(
  *,
  allow_delete: bool = False,
  allow_mute: bool = False,
  allow_kick: bool = False,
  allow_subagent: bool = False,
) -> list[dict]:
  """Build the LLM2 tool list based on current chat permissions."""
  tools = list(LLM2_BASE_TOOLS)
  if allow_delete:
    tools.append(LLM2_DELETE_TOOL)
  if allow_mute:
    tools.append(LLM2_MUTE_TOOL)
  if allow_kick:
    tools.append(LLM2_KICK_TOOL)
  if allow_subagent:
    tools.append(LLM2_SUBAGENT_TOOL)
  return tools


class LLM1Decision(BaseModel):
  should_response: bool = Field(..., description="Whether to respond")
  confidence: int = Field(..., ge=0, le=100)
  reason: str = Field(..., min_length=2, max_length=320)
  react_expression: str | None = Field(default=None, description="Emoji or sticker name for express-only decisions")
  react_context_msg_id: str | None = Field(default=None, description="Target message contextMsgId for react-only")
  input_tokens: int = Field(default=0, description="LLM1 input tokens used")
  output_tokens: int = Field(default=0, description="LLM1 output tokens used")
