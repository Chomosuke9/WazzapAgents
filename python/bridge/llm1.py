# File: python/bridge/llm1.py
from __future__ import annotations

import json
import os
import time
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

try:
  from .history import WhatsAppMessage, assistant_name, format_history
  from .log import setup_logging, trunc, dump_json, env_flag
  from .media import build_visual_parts, llm1_media_enabled, redact_multimodal_content
  from .config import _parse_positive_int, _parse_positive_float, _parse_non_negative_int, _parse_non_negative_float
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage, assistant_name, format_history  # type: ignore
  from bridge.log import setup_logging, trunc, dump_json, env_flag  # type: ignore
  from bridge.media import build_visual_parts, llm1_media_enabled, redact_multimodal_content  # type: ignore
  from bridge.config import _parse_positive_int, _parse_positive_float, _parse_non_negative_int, _parse_non_negative_float  # type: ignore

logger = setup_logging()


@dataclass(frozen=True)
class LLM1Target:
  name: str
  model: str
  base_url: str
  api_key: str


def _llm1_history_limit() -> int:
  # Prefer LLM1-specific limit; fallback to global history limit.
  raw = os.getenv("LLM1_HISTORY_LIMIT")
  if raw is None or not raw.strip():
    raw = os.getenv("HISTORY_LIMIT")
  return _parse_positive_int(raw, 20)


def _llm1_message_max_chars() -> int:
  return _parse_positive_int(os.getenv("LLM1_MESSAGE_MAX_CHARS"), 500)


def _llm1_timeout(default: float = 8.0) -> float:
  return _parse_positive_float(os.getenv("LLM1_TIMEOUT"), default)


def _llm1_sdk_max_retries() -> int:
  return _parse_non_negative_int(os.getenv("LLM1_SDK_MAX_RETRIES"), 0)


def _llm1_temperature() -> float:
  return _parse_non_negative_float(os.getenv("LLM1_TEMPERATURE"), 0.0)


def _llm1_max_tokens() -> int | None:
  raw = os.getenv("LLM1_MAX_TOKENS")
  if raw is None:
    return None
  cleaned = raw.strip()
  if not cleaned:
    return None
  try:
    parsed = int(cleaned)
  except (TypeError, ValueError):
    return None
  return parsed if parsed > 0 else None


def _llm1_reasoning_effort() -> str | None:
  raw = os.getenv("LLM1_REASONING_EFFORT")
  if raw is None:
    return None
  cleaned = raw.strip().lower()
  if not cleaned:
    return None
  return cleaned


def _truncate_text(text: str | None, max_chars: int) -> str | None:
  if text is None or len(text) <= max_chars:
    return text
  if max_chars <= 3:
    return text[:max_chars]
  return f"{text[: max_chars - 3]}..."


def _truncate_burst_text(text: str | None, max_chars: int) -> str | None:
  if text is None:
    return None
  if not text.startswith("Burst messages ("):
    return _truncate_text(text, max_chars)
  lines = text.splitlines()
  if not lines:
    return text
  header = lines[0]
  body = lines[1:]
  truncated_body = [_truncate_text(line, max_chars) or "" for line in body]
  return "\n".join([header, *truncated_body])


def _truncate_message(msg: WhatsAppMessage, max_chars: int) -> WhatsAppMessage:
  return WhatsAppMessage(
    timestamp_ms=msg.timestamp_ms,
    sender=msg.sender,
    context_msg_id=msg.context_msg_id,
    sender_ref=msg.sender_ref,
    sender_is_admin=msg.sender_is_admin,
    text=_truncate_burst_text(msg.text, max_chars),
    media=msg.media,
    quoted_message_id=msg.quoted_message_id,
    quoted_sender=msg.quoted_sender,
    quoted_text=_truncate_text(msg.quoted_text, max_chars),
    quoted_media=msg.quoted_media,
    message_id=msg.message_id,
    role=msg.role,
  )


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

LLM1_REACT_SCHEMA = {
  "name": "llm_react_only",
  "parameters": {
    "type": "object",
    "properties": {
      "emoji": {
        "type": "string",
        "description": "A single emoji to react with (e.g. 👍, 😂, ❤️, 🔥, 😢).",
        "minLength": 1,
        "maxLength": 1,
      },
      "context_msg_id": {
        "type": "string",
        "description": (
          "The 6-digit contextMsgId of the message to react to. "
          "Use the id from current messages(burst). "
          "Use the last message id if reacting to the most recent message."
        ),
        "minLength": 6,
        "maxLength": 6,
      },
      "confidence": {
        "type": "integer",
        "description": "Confidence percentage (0-100) about the reaction decision.",
        "minimum": 0,
        "maximum": 100,
      },
      "reason": {
        "type": "string",
        "description": (
          "A concise reason for reacting. "
          "1-2 short sentences (max 320 chars)."
        ),
        "minLength": 2,
        "maxLength": 320,
      },
    },
    "required": ["emoji", "context_msg_id", "confidence", "reason"],
    "additionalProperties": False,
  },
}

LLM1_REACT_TOOL = {
  "type": "function",
  "function": {
    "name": LLM1_REACT_SCHEMA["name"],
    "description": (
      "React to a message with an emoji instead of sending a text reply. "
      "Use this when the situation calls for an emoji reaction only (no text response needed)."
    ),
    "parameters": LLM1_REACT_SCHEMA["parameters"],
    "strict": True,
  },
}

LLM1_TOOLS = [LLM1_TOOL, LLM1_REACT_TOOL]


class LLM1Decision(BaseModel):
  should_response: bool = Field(..., description="Whether to respond")
  confidence: int = Field(..., ge=0, le=100)
  reason: str = Field(..., min_length=2, max_length=320)
  react_emoji: str | None = Field(default=None, description="Emoji for react-only decisions")
  react_context_msg_id: str | None = Field(default=None, description="Target message contextMsgId for react-only")


def _render_prompt_override(base_system: str, prompt_override: str | None) -> str:
  rendered = base_system
  overide_text = (prompt_override or "").strip()
  rendered = rendered.replace("{{prompt_override}}", overide_text)
  rendered = rendered.replace("{{ prompt_override }}", overide_text)
  return rendered


def _group_description_block(group_description: str | None) -> str:
  cleaned = (group_description or "").strip()
  if cleaned:
    return cleaned
  return "(none)"


def _format_current_window(msg: WhatsAppMessage) -> str:
  # Burst windows are already serialized as multi-line chat entries.
  text = (msg.text or "").strip()
  if text.startswith("Burst messages ("):
    return text
  return format_history([msg])


def build_llm1_prompt(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  history_limit: int,
  message_max_chars: int,
  current_media_parts: Optional[list[dict]] = None,
  current_media_notes: Optional[list[str]] = None,
  metadata_block: str | None = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
):
  configured_assistant_name = assistant_name()
  history_list = list(history)[-history_limit:]
  prompt_history = [_truncate_message(msg, message_max_chars) for msg in history_list]
  current_prompt_msg = _truncate_message(current, message_max_chars)
  hist_text = format_history(prompt_history) or "(no older messages)"
  current_line = _format_current_window(current_prompt_msg) or "(no current messages)"
  group_text = _group_description_block(group_description)
  context_messages = (
    "older messages:\n"
    f"{hist_text}\n\n"
    "current messages(burst):\n"
    f"{current_line}\n"
  )
  current_content: str | list[dict] = context_messages
  if current_media_notes:
    current_content += "\nVisual attachments:\n" + "\n".join(
      f"- {note}" for note in current_media_notes
    )
  if current_media_parts:
    current_content = [{"type": "text", "text": current_content}]
    current_content.extend(current_media_parts)
  base_system = f"""
You are a WhatsApp router agent ({configured_assistant_name}). Decide whether to respond.
Core rule: Default state is SILENT. Respond only when evidence clearly justifies it. Being talked ABOUT is not being talked TO. An active conversation you were not invited into is not yours to join. When in doubt, stay silent — silence is the correct behavior for most messages.

Call exactly one tool — either `llm_should_response` or `llm_react_only`. No other text output.

`llm_should_response` — route to response generator or skip entirely.
Args: should_response (bool), confidence (0-100), reason (1-3 sentences, 12-60 words, max 320 chars).
Reason is forwarded to LLM2—keep it specific and actionable, no generic phrases or chain-of-thought.

`llm_react_only` — react to a message with an emoji instead of sending a text reply.

Mention token: @{configured_assistant_name} (bot). Always respond when mentioned.
Input: up to {_llm1_history_limit()} messages, each capped at {_llm1_message_max_chars()} chars.

## Input format
- `Current message metadata`: Helper section (mention/reply signals, recency, window size, join-event counts, conversation continuity) + Chat state.
- `Group description`: Topic/rules set by admins. Use it to judge relevance—respond when the message aligns with the group’s stated purpose; lean silent when it doesn’t.
- `older messages`: background history. `current messages(burst)`: trigger window.
- `current message window` = only `current messages(burst)`, not `older messages`.
- Message ids: 6-digit. `<system>`/`<pending>` = non-actionable markers.
- Burst may contain multiple combined messages—evaluate all, not just the last line.
- Sticker-only or media-without-text messages: treat as casual/non-verbal. Stay silent unless the bot is mentioned, replied to, or the media is a direct question (e.g., photo asking "what is this?").
- New member = explicit system join signal only (not first appearance or "hi").
- Conversation signals are hints, not hard rules. Use them together with message content to decide.

## Response tiers — evaluate in order, stop at first match

**MUST RESPOND** (confidence 90–95):
- Bot is @mentioned (metadata says "Bot is mentioned in this current message window")
- Message is a direct reply to the bot (metadata says "A message in this current message window replies to the bot")

**SHOULD RESPOND** (confidence 65–80) — only if no human has already answered adequately:
- Current window contains a clear unanswered question AND the topic is within bot’s domain
- Explicit open help request ("does anyone know?", "can someone help?", "anyone know?") with no human response yet

**MAY RESPOND** (confidence 40–60) — use careful judgment:
- Bot is in an active thread (last assistant reply was recent, within ~2 messages) AND the message is a direct follow-up question to the bot’s last reply specifically

**REACT-ONLY** — call `llm_react_only` with the appropriate emoji and target message id:
- Emotional/social content: jokes, congratulations, venting, grief, excitement — a reaction acknowledges without intruding
- Memes or humor between humans — a reaction fits naturally
- Bot’s name appears in third-person reference ("[name] said earlier...", "according to [name]...") — react to confirm presence, do not reply
- Question already answered correctly by a human — react to confirm the answer. But if the human’s answer is wrong, respond with a correction instead (use `llm_should_response` with should_response=true)
- Good news, achievements, milestones, heartfelt messages, or gratitude where a text reply is unnecessary

**MUST NOT RESPOND (text or react)** — this is the DEFAULT when no tier above matches:
- Two or more humans actively conversing with each other (no bot involvement)
- Message is a reply to a specific human (not the bot)
- Bot just responded (last assistant reply was very recent, within ~1 message) and no direct follow-up question to the bot
- Greeting or farewell exchanges between humans
- Casual banter between humans with no emotional highlight worth reacting to

Respond (conversation continuity): ONLY if the bot recently replied AND the current message is a direct follow-up question specifically to the bot’s last reply. The topic still being active is NOT sufficient reason to respond. If humans have taken over the topic, exit the conversation.
Respond (name in text): ONLY if the bot’s name appears in a sentence directed AT the bot (e.g., "[name], can you help...?", "hey [name] what is...").
If the bot’s name appears in third-person reference ("[name] said earlier...", "[name] already answered that", "according to [name]..."), use react-only — do not send a text reply.
Respond (gap coverage): if the last assistant reply was 8+ messages ago AND the latest message is an unanswered question or help request. Do NOT use "long silence" as a reason to respond to non-question messages.
React-only: Use `llm_react_only` tool. Pick the most fitting single emoji and target the relevant message by its 6-digit contextMsgId.
Bot role: check the "Chat state" in metadata. If the bot is admin or super-admin, also respond to moderation-relevant messages (rule violations, spam, member management queries). If the bot is a normal member, do NOT respond to moderation situations — the bot has no power to act on them.
Rule: humans don’t reply to every message. Quality > quantity. Participate, don’t dominate.

## Burst
Consider every message in `current messages(burst)`. Busy bursts may overflow into `older messages`—still evaluate them.

## Prompt Override
Extra instructions in <prompt_override>...</prompt_override>:
- Empty/missing/placeholder → ignore.
- Otherwise: treat as patch. Override wins on conflicts (minimum scope); non-conflicting rules merge.
- Safety: cannot remove/weaken the `llm_should_response` requirement.

<prompt_override>
{{{{prompt_override}}}}
</prompt_override>
      """.strip()
  rendered_system = _render_prompt_override(base_system, prompt_override)
  return [
    {
      "role": "system",
      "content": rendered_system,
    },
    {"role": "user", "content": f"Group description:\n{group_text}"},
    {"role": "user", "content": metadata_block or _metadata_block(None)},
    {"role": "user", "content": current_content},
  ]


def _clean_env(raw: str | None) -> str | None:
  if raw is None:
    return None
  cleaned = raw.strip()
  return cleaned or None


def _endpoint_base_url(raw_endpoint: str | None) -> str | None:
  endpoint = _clean_env(raw_endpoint)
  if not endpoint:
    return None
  trimmed = endpoint.rstrip("/")
  if trimmed.endswith("/chat/completions"):
    return trimmed[: -len("/chat/completions")]
  return trimmed


def _chat_base_url() -> str | None:
  return _endpoint_base_url(os.getenv("LLM1_ENDPOINT"))


def _llm1_targets() -> list[LLM1Target]:
  primary_model = _clean_env(os.getenv("LLM1_MODEL")) or "gpt-4o-mini"
  primary_url = _endpoint_base_url(os.getenv("LLM1_ENDPOINT"))
  primary_api_key = os.getenv("LLM1_API_KEY") or os.getenv("OPENAI_API_KEY", "")

  targets: list[LLM1Target] = []
  if primary_url:
    targets.append(
      LLM1Target(
        name="primary",
        model=primary_model,
        base_url=primary_url,
        api_key=primary_api_key,
      )
    )

  fallback_model_raw = _clean_env(os.getenv("LLM1_FALLBACK_MODEL"))
  fallback_url_raw = _clean_env(os.getenv("LLM1_FALLBACK_ENDPOINT"))
  fallback_api_key_raw = _clean_env(os.getenv("LLM1_FALLBACK_API_KEY"))
  fallback_enabled = any((fallback_model_raw, fallback_url_raw, fallback_api_key_raw))
  if not fallback_enabled:
    return targets

  fallback_url = _endpoint_base_url(fallback_url_raw) or primary_url
  if not fallback_url:
    return targets
  fallback_model = fallback_model_raw or primary_model
  fallback_api_key = fallback_api_key_raw or primary_api_key
  fallback_target = LLM1Target(
    name="fallback",
    model=fallback_model,
    base_url=fallback_url,
    api_key=fallback_api_key,
  )
  if targets:
    primary_target = targets[0]
    if (
      fallback_target.model == primary_target.model
      and fallback_target.base_url == primary_target.base_url
      and fallback_target.api_key == primary_target.api_key
    ):
      return targets
  targets.append(fallback_target)
  return targets


def get_llm1(
  *,
  model: str | None = None,
  base_url: str | None = None,
  api_key: str | None = None,
  timeout: float = 8.0,
  include_reasoning: bool = True,
) -> ChatOpenAI:
  resolved_model = model or _clean_env(os.getenv("LLM1_MODEL")) or "gpt-4o-mini"
  resolved_base_url = base_url if base_url is not None else _chat_base_url()
  resolved_api_key = api_key if api_key is not None else (os.getenv("LLM1_API_KEY") or os.getenv("OPENAI_API_KEY", ""))
  max_tokens = _llm1_max_tokens()
  kwargs = {
    "model": resolved_model,
    "api_key": resolved_api_key,
    "timeout": _llm1_timeout(timeout),
    "max_retries": _llm1_sdk_max_retries(),
    "temperature": _llm1_temperature(),
  }
  if max_tokens is not None:
    kwargs["max_tokens"] = max_tokens
  if resolved_base_url:
    kwargs["base_url"] = resolved_base_url
  reasoning_effort = _llm1_reasoning_effort() if include_reasoning else None
  if reasoning_effort:
    kwargs["reasoning_effort"] = reasoning_effort
  return ChatOpenAI(
    **kwargs,
  )


def _prompt_to_langchain_messages(prompt: list[dict]) -> list[SystemMessage | HumanMessage]:
  messages: list[SystemMessage | HumanMessage] = []
  for item in prompt:
    if not isinstance(item, dict):
      continue
    role = str(item.get("role") or "").strip().lower()
    content = item.get("content", "")
    if role == "system":
      messages.append(SystemMessage(content=content))
    else:
      messages.append(HumanMessage(content=content))
  return messages


def _is_timeout_error(err: Exception) -> bool:
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < 8:
    if "timeout" in type(current).__name__.lower():
      return True
    current = current.__cause__ or current.__context__
    depth += 1
  return False


def _error_chain(err: Exception, limit: int = 8) -> list[str]:
  chain: list[str] = []
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < limit:
    chain.append(type(current).__name__)
    current = current.__cause__ or current.__context__
    depth += 1
  return chain


def _error_text_chain(err: Exception, limit: int = 8) -> str:
  texts: list[str] = []
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < limit:
    text = str(current).strip()
    if text:
      texts.append(text.lower())
    current = current.__cause__ or current.__context__
    depth += 1
  return " | ".join(texts)


def _is_reasoning_unsupported_error(err: Exception) -> bool:
  text = _error_text_chain(err)
  if "reasoning_effort" not in text and "reasoning effort" not in text:
    return False
  unsupported_markers = (
    "unsupported",
    "not supported",
    "unknown",
    "invalid",
    "not allowed",
    "unrecognized",
  )
  return any(marker in text for marker in unsupported_markers)


def _llm1_ctx(
  current: WhatsAppMessage,
  *,
  provider: str,
  model: str,
  url: str | None,
  current_payload: dict | None = None,
) -> dict:
  payload = current_payload if isinstance(current_payload, dict) else {}
  chat_id = payload.get("chatId") or payload.get("chat_id")
  raw_chat_type = str(payload.get("chatType") or payload.get("chat_type") or "").strip().lower()
  if raw_chat_type not in {"group", "private"}:
    if isinstance(chat_id, str) and chat_id.endswith("@g.us"):
      raw_chat_type = "group"
    else:
      raw_chat_type = "group" if bool(payload.get("isGroup")) else "private"
  chat_name = (payload.get("chatName") or payload.get("chat_name")) if raw_chat_type == "group" else None
  return {
    "chat_id": chat_id or getattr(current, "sender", None),
    "chat_name": chat_name,
    "message_id": getattr(current, "message_id", None) or getattr(current, "id", None),
    "provider": provider,
    "model": model,
    "endpoint": url,
  }


def _log_llm1_decision(
  decision: LLM1Decision,
  *,
  ctx: dict,
  elapsed_ms: int,
  source: str,
) -> None:
  status = "respond" if decision.should_response else "skip"
  reason_text = trunc(" ".join((decision.reason or "").split()), 220)
  logger.info(
    'LLM1 decision final (%s): %s conf=%s%% reason="%s" elapsed=%sms',
    source,
    status,
    decision.confidence,
    reason_text,
    elapsed_ms,
    extra={
      **ctx,
      "source": source,
      "should_response": decision.should_response,
      "confidence": decision.confidence,
      "reason": decision.reason,
      "elapsed_ms": elapsed_ms,
      "raw": trunc(dump_json(decision.model_dump()), 400),
    },
  )


def _extract_tool_args(tool_call) -> dict:
  """Best-effort extraction of tool arguments across provider shapes."""
  raw_args = None
  raw_fn = None

  if isinstance(tool_call, dict):
    raw_args = (
      tool_call.get("args")
      or tool_call.get("arguments")
      or tool_call.get("input")
      or tool_call.get("parameters")
    )
    raw_fn = tool_call.get("function")
  else:
    raw_args = (
      getattr(tool_call, "args", None)
      or getattr(tool_call, "arguments", None)
      or getattr(tool_call, "input", None)
      or getattr(tool_call, "parameters", None)
    )
    raw_fn = getattr(tool_call, "function", None)

  # OpenAI-like shape: {"function": {"arguments": "..."}}
  if not raw_args and isinstance(raw_fn, dict):
    raw_args = (
      raw_fn.get("args")
      or raw_fn.get("arguments")
      or raw_fn.get("input")
      or raw_fn.get("parameters")
    )

  if isinstance(raw_args, str):
    try:
      raw_args = json.loads(raw_args)
    except json.JSONDecodeError:
      return {}

  return raw_args or {}


def _content_to_text(content) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: list[str] = []
    for item in content:
      if not isinstance(item, dict):
        parts.append(str(item))
        continue
      if item.get("type") == "text":
        parts.append(str(item.get("text") or ""))
        continue
      if item.get("type") == "image_url":
        parts.append("[image]")
        continue
      parts.append(f"[{item.get('type') or 'part'}]")
    return "\n".join(parts)
  return str(content)


def _extract_decision_from_content(content) -> dict:
  text = _content_to_text(content).strip()
  if not text:
    return {}

  candidates: list[str] = [text]
  fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
  if fenced:
    fenced_text = fenced.group(1).strip()
    if fenced_text:
      candidates.append(fenced_text)

  first_brace = text.find("{")
  last_brace = text.rfind("}")
  if first_brace >= 0 and last_brace > first_brace:
    candidates.append(text[first_brace : last_brace + 1].strip())

  for candidate in candidates:
    try:
      parsed = json.loads(candidate)
    except Exception:
      continue
    if isinstance(parsed, dict):
      return parsed
  return {}


def _metadata_block(current_payload: dict | None) -> str:
  payload = current_payload if isinstance(current_payload, dict) else {}
  bot_mentioned = bool(payload.get("botMentionedInWindow", payload.get("botMentioned")))
  replied_to_bot = bool(payload.get("repliedToBotInWindow", payload.get("repliedToBot")))
  bot_name_in_text = bool(payload.get("botNameMentionedInText"))
  since_assistant = payload.get("messagesSinceAssistantReply")
  assistant_replies_by_window = payload.get("assistantRepliesByWindow")
  human_window = payload.get("humanMessagesInWindow")
  explicit_join_events = payload.get("explicitJoinEventsInWindow")
  explicit_join_participants = payload.get("explicitJoinParticipantsInWindow")
  raw_chat_type = str(payload.get("chatType") or "").strip().lower()
  if raw_chat_type not in {"private", "group"}:
    raw_chat_type = "group" if bool(payload.get("isGroup")) else "private"
  if raw_chat_type == "group":
    scope_line = "This is a group chat. You're in a chat with multiple people at once."
  else:
    scope_line = "This is a private chat. You're directly chatting with one other person."
  if bool(payload.get("botIsSuperAdmin")):
    role_line = "Bot is a super admin (owner)."
  elif bool(payload.get("botIsAdmin")):
    role_line = "Bot is an admin."
  else:
    role_line = "Bot is a normal member."

  def _count_phrase(value, singular: str, plural: str) -> str:
    if value is None:
      return f"unknown {plural}"
    if isinstance(value, int):
      return f"{value} {singular if value == 1 else plural}"
    return f"{value} {plural}"

  def _is_singular_count(value) -> bool:
    return isinstance(value, int) and value == 1

  if bot_mentioned:
    mention_line = "- Bot is mentioned in this current message window."
  else:
    mention_line = "- Bot is not mentioned in this current message window."

  if replied_to_bot:
    reply_line = "- A message in this current message window replies to the bot."
  else:
    reply_line = "- No message in this current message window replies to the bot."

  if bot_name_in_text and not bot_mentioned:
    name_line = "- Bot's name is mentioned in the message text (without explicit @mention). Treat this as a soft mention — the user is likely talking to or about the bot."
  elif bot_name_in_text and bot_mentioned:
    name_line = "- Bot's name appears in the message text (already counted as @mention above)."
  else:
    name_line = None

  since_assistant_text = _count_phrase(since_assistant, "message", "messages")
  human_window_text = _count_phrase(human_window, "human message", "human messages")

  assistant_reply_lines: list[str] = []
  if isinstance(assistant_replies_by_window, dict):
    assistant_reply_values: list[tuple[int, int | str]] = []
    for raw_window, raw_count in assistant_replies_by_window.items():
      try:
        window = int(raw_window)
      except (TypeError, ValueError):
        continue
      assistant_reply_values.append((window, raw_count))
    assistant_reply_values.sort(key=lambda item: item[0])
    for window, count in assistant_reply_values:
      count_text = _count_phrase(count, "reply", "replies")
      assistant_reply_lines.append(
        f"- Assistant has sent {count_text} in the last {window} messages."
      )

  if not assistant_reply_lines:
    fallback_recent = payload.get("assistantRepliesInLast20")
    fallback_text = _count_phrase(fallback_recent, "reply", "replies")
    assistant_reply_lines.append(
      f"- Assistant has sent {fallback_text} in the last 20 messages."
    )

  if _is_singular_count(human_window):
    human_window_line = f"- There is {human_window_text} in this current message window."
  else:
    human_window_line = f"- There are {human_window_text} in this current message window."

  join_event_text = _count_phrase(explicit_join_events, "event", "events")
  join_participant_text = _count_phrase(explicit_join_participants, "participant", "participants")
  if isinstance(explicit_join_events, int):
    if explicit_join_events > 0:
      join_event_line = (
        "- Explicit system member-join signals in this current message window: "
        f"{join_event_text} ({join_participant_text})."
      )
    else:
      join_event_line = "- No explicit system member-join signal in this current message window."
  else:
    join_event_line = "- Explicit system member-join signal count is unknown for this current message window."

  assistant_reply_block = "\n".join(assistant_reply_lines)
  extra_signal_block = ""
  if name_line:
    extra_signal_block = f"\n{name_line}"
  return (
    "Current message metadata:\n"
    "Helper:\n"
    "- `current message window` = only `current messages(burst)` (exclude `older messages`).\n"
    f"{mention_line}\n"
    f"{reply_line}\n"
    f"- The last assistant reply was {since_assistant_text} ago.\n"
    f"{assistant_reply_block}\n"
    f"{human_window_line}\n"
    f"{join_event_line}"
    f"{extra_signal_block}\n"
    "Chat state:\n"
    f"{scope_line}\n"
    f"{role_line}"
  )


def _redact_messages_for_log(messages: list[dict]) -> list[dict]:
  redacted: list[dict] = []
  for msg in messages:
    if not isinstance(msg, dict):
      continue
    copied = dict(msg)
    copied["content"] = redact_multimodal_content(copied.get("content"))
    redacted.append(copied)
  return redacted


async def call_llm1(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  timeout: float = 8.0,
  client: Optional[ChatOpenAI] = None,
  current_payload: dict | None = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
) -> LLM1Decision:
  primary_endpoint = _clean_env(os.getenv("LLM1_ENDPOINT"))
  fallback_endpoint = _clean_env(os.getenv("LLM1_FALLBACK_ENDPOINT"))
  # If LLM1 is not configured, allow responding by default.
  if not primary_endpoint and not fallback_endpoint:
    logger.debug("LLM1 disabled (no LLM1_ENDPOINT set); defaulting to respond")
    return LLM1Decision(should_response=True, confidence=50, reason="llm1_disabled")

  targets = _llm1_targets()
  if client is not None and targets:
    targets = targets[:1]
  if not targets:
    logger.debug("LLM1 endpoint missing after normalization; defaulting to skip")
    return LLM1Decision(should_response=False, confidence=10, reason="llm1_missing_url")

  history_limit = _llm1_history_limit()
  message_max_chars = _llm1_message_max_chars()
  history_list = list(history)
  prompt_history = history_list[-history_limit:]
  current_media_parts: list[dict] = []
  current_media_notes: list[str] = []
  if llm1_media_enabled():
    current_media_parts, current_media_notes = build_visual_parts(current_payload)
  prompt = build_llm1_prompt(
    prompt_history,
    current,
    history_limit=history_limit,
    message_max_chars=message_max_chars,
    current_media_parts=current_media_parts,
    current_media_notes=current_media_notes,
    metadata_block=_metadata_block(current_payload),
    group_description=group_description,
    prompt_override=prompt_override,
  )
  prompt_text = "\n".join(
    [_content_to_text(m.get("content", "")) for m in prompt if isinstance(m, dict)]
  )

  last_failure: LLM1Decision | None = None
  total_targets = len(targets)
  llm1_temperature = _llm1_temperature()
  llm1_max_tokens = _llm1_max_tokens()

  for idx, target in enumerate(targets):
    has_next_target = idx < (total_targets - 1)
    reasoning_effort = _llm1_reasoning_effort()
    t0 = time.perf_counter()
    ctx = _llm1_ctx(
      current,
      provider=target.name,
      model=target.model,
      url=target.base_url,
      current_payload=current_payload,
    )
    llm = client if (client is not None and idx == 0) else get_llm1(
      model=target.model,
      base_url=target.base_url,
      api_key=target.api_key,
      timeout=timeout,
      include_reasoning=bool(reasoning_effort),
    )

    if env_flag("BRIDGE_LOG_PROMPT_FULL"):
      logger.info(
        "LLM1 prompt full",
        extra={
          **ctx,
          "history_limit": history_limit,
          "history_used": len(prompt_history),
          "message_max_chars": message_max_chars,
          "base_url": target.base_url,
          "media_parts": len(current_media_parts),
          "reasoning_effort": reasoning_effort,
          "messages": _redact_messages_for_log(prompt),
        },
      )

    logger.info(
      "LLM1 invoke start (model=%s, history=%s)",
      target.model,
      len(prompt_history),
      extra={
        **ctx,
        "history_used": len(prompt_history),
        "media_parts": len(current_media_parts),
        "temperature": llm1_temperature,
        "max_tokens": llm1_max_tokens,
        "reasoning_effort": reasoning_effort,
      },
    )

    logger.debug(
      "LLM1 request start",
      extra={
        **ctx,
        "history_limit": history_limit,
        "history_used": len(prompt_history),
        "message_max_chars": message_max_chars,
        "timeout_s": _llm1_timeout(timeout),
        "prompt_chars": len(prompt_text),
        "prompt_preview": trunc(prompt_text, 300),
        "media_parts": len(current_media_parts),
        "base_url": target.base_url,
        "temperature": llm1_temperature,
        "max_tokens": llm1_max_tokens,
        "reasoning_effort": reasoning_effort,
        "tool_names": [t["function"]["name"] for t in LLM1_TOOLS],
      },
    )

    async def _invoke_once(llm_client: ChatOpenAI):
      try:
        llm_with_tool = llm_client.bind_tools(
          LLM1_TOOLS,
          tool_choice="required",
        )
      except Exception as err:
        logger.warning(
          "LLM1 bind_tools with tool_choice=required failed; retrying default bind_tools",
          exc_info=err,
          extra={
            **ctx,
            "error_type": type(err).__name__,
          },
        )
        llm_with_tool = llm_client.bind_tools(LLM1_TOOLS)
      return await llm_with_tool.ainvoke(_prompt_to_langchain_messages(prompt))

    try:
      response = await _invoke_once(llm)
    except Exception as err:
      # Some providers reject `reasoning_effort`. Retry once without it when using
      # internally created client.
      if (
        client is None
        and reasoning_effort
        and _is_reasoning_unsupported_error(err)
      ):
        logger.warning(
          "LLM1 invoke rejected reasoning_effort; retrying without reasoning",
          exc_info=True,
          extra={
            **ctx,
            "reasoning_effort": reasoning_effort,
            "error_type": type(err).__name__,
            "error_chain": _error_chain(err),
          },
        )
        llm = get_llm1(
          model=target.model,
          base_url=target.base_url,
          api_key=target.api_key,
          timeout=timeout,
          include_reasoning=False,
        )
        reasoning_effort = None
        try:
          response = await _invoke_once(llm)
        except Exception as retry_err:
          elapsed_ms = int((time.perf_counter() - t0) * 1000)
          timeout_error = _is_timeout_error(retry_err)
          logger.error(
            "LLM1 invoke failed after reasoning fallback",
            exc_info=True,
            extra={
              **ctx,
              "elapsed_ms": elapsed_ms,
              "reasoning_effort": reasoning_effort,
              "error_type": type(retry_err).__name__,
              "error_chain": _error_chain(retry_err),
              "will_try_fallback_target": has_next_target,
            },
          )
          last_failure = LLM1Decision(
            should_response=False,
            confidence=10,
            reason="llm1_unreachable" if timeout_error else "llm1_exception",
          )
          if has_next_target:
            logger.warning(
              "LLM1 provider failed; trying fallback target",
              extra={
                **ctx,
                "next_provider": targets[idx + 1].name,
              },
            )
            continue
          return last_failure
      else:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        timeout_error = _is_timeout_error(err)
        logger.error(
          "LLM1 invoke failed",
          exc_info=True,
          extra={
            **ctx,
            "elapsed_ms": elapsed_ms,
            "reasoning_effort": reasoning_effort,
            "error_type": type(err).__name__,
            "error_chain": _error_chain(err),
            "will_try_fallback_target": has_next_target,
          },
        )
        last_failure = LLM1Decision(
          should_response=False,
          confidence=10,
          reason="llm1_unreachable" if timeout_error else "llm1_exception",
        )
        if has_next_target:
          logger.warning(
            "LLM1 provider failed; trying fallback target",
            extra={
              **ctx,
              "next_provider": targets[idx + 1].name,
            },
          )
          continue
        return last_failure

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    response_metadata = getattr(response, "response_metadata", None)
    usage_metadata = getattr(response, "usage_metadata", None)
    raw_tool_calls = getattr(response, "tool_calls", None) or []
    content = getattr(response, "content", None)
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    if not raw_tool_calls and isinstance(additional_kwargs, dict):
      maybe_tool_calls = additional_kwargs.get("tool_calls")
      if isinstance(maybe_tool_calls, list):
        raw_tool_calls = maybe_tool_calls

    logger.debug(
      "LLM1 response received",
      extra={
        **ctx,
        "elapsed_ms": elapsed_ms,
        "reasoning_effort": reasoning_effort,
        "response_metadata": response_metadata,
        "usage": usage_metadata,
        "tool_calls_count": len(raw_tool_calls),
        "content_preview": trunc(_content_to_text(content), 600),
      },
    )

    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(
        "LLM1 raw response",
        extra={
          **ctx,
          "raw": dump_json(getattr(response, "model_dump", lambda: str(response))()),
        },
      )

    tool_calls = raw_tool_calls or []
    if not tool_calls:
      parsed_fallback = _extract_decision_from_content(content)
      if parsed_fallback:
        try:
          decision = LLM1Decision.model_validate(parsed_fallback)
          logger.warning(
            "LLM1 response missing tool call; parsed JSON fallback",
            extra={
              **ctx,
              "reasoning_effort": reasoning_effort,
              "response_metadata": response_metadata,
              "fallback_args": parsed_fallback,
            },
          )
          _log_llm1_decision(
            decision,
            ctx=ctx,
            elapsed_ms=elapsed_ms,
            source="json_fallback",
          )
          return decision
        except ValidationError:
          pass
      logger.warning(
        "LLM1 response missing tool call",
        extra={
          **ctx,
          "reasoning_effort": reasoning_effort,
          "response_metadata": response_metadata,
          "will_try_fallback_target": has_next_target,
        },
      )
      last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_no_tool")
      if has_next_target:
        logger.warning(
          "LLM1 invalid response shape; trying fallback target",
          extra={
            **ctx,
            "next_provider": targets[idx + 1].name,
          },
        )
        continue
      return last_failure

    # Detect which tool was called: llm_should_response or llm_react_only
    respond_tool_name = LLM1_TOOL["function"]["name"]
    react_tool_name = LLM1_REACT_TOOL["function"]["name"]

    def _get_tool_call_name(tc) -> str | None:
      if isinstance(tc, dict):
        name = tc.get("name")
        if name:
          return str(name)
        fn = tc.get("function")
        if isinstance(fn, dict):
          return fn.get("name")
      else:
        name = getattr(tc, "name", None)
        if name:
          return str(name)
        fn = getattr(tc, "function", None)
        if isinstance(fn, dict):
          return fn.get("name")
      return None

    # Find the first recognized tool call
    tool_call = None
    called_tool_name = None
    for tc in tool_calls:
      tc_name = _get_tool_call_name(tc)
      if tc_name in (respond_tool_name, react_tool_name):
        tool_call = tc
        called_tool_name = tc_name
        break
    if tool_call is None:
      tool_call = tool_calls[0]
      called_tool_name = _get_tool_call_name(tool_call)

    args = _extract_tool_args(tool_call)
    if not args:
      logger.warning(
        "LLM1 tool args empty",
        extra={
          **ctx,
          "raw_tool_call": trunc(str(tool_call), 500),
          "will_try_fallback_target": has_next_target,
        },
      )
      last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_empty_tool")
      if has_next_target:
        logger.warning(
          "LLM1 invalid tool args; trying fallback target",
          extra={
            **ctx,
            "next_provider": targets[idx + 1].name,
          },
        )
        continue
      return last_failure

    # Handle llm_react_only tool call
    if called_tool_name == react_tool_name:
      react_emoji = str(args.get("emoji") or "").strip()
      react_context_msg_id = str(args.get("context_msg_id") or "").strip()
      react_confidence = args.get("confidence", 50)
      react_reason = str(args.get("reason") or "react-only").strip()
      if not react_emoji or not react_context_msg_id:
        logger.warning(
          "LLM1 llm_react_only missing emoji or context_msg_id",
          extra={**ctx, "raw_args": args, "will_try_fallback_target": has_next_target},
        )
        last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_invalid_react_tool")
        if has_next_target:
          continue
        return last_failure
      decision = LLM1Decision(
        should_response=False,
        confidence=react_confidence if isinstance(react_confidence, int) else 50,
        reason=react_reason[:320],
        react_emoji=react_emoji,
        react_context_msg_id=react_context_msg_id,
      )
      logger.info(
        'LLM1 react-only decision: emoji=%s target=%s conf=%s%% reason="%s" elapsed=%sms',
        react_emoji,
        react_context_msg_id,
        decision.confidence,
        trunc(" ".join((decision.reason or "").split()), 220),
        elapsed_ms,
        extra={
          **ctx,
          "source": "react_tool_call",
          "should_response": False,
          "react_emoji": react_emoji,
          "react_context_msg_id": react_context_msg_id,
          "confidence": decision.confidence,
          "reason": decision.reason,
          "elapsed_ms": elapsed_ms,
        },
      )
      return decision

    # Handle llm_should_response tool call (existing behavior)
    try:
      decision = LLM1Decision.model_validate(args)
    except ValidationError as err:
      logger.warning(
        "LLM1 tool args failed validation",
        exc_info=err,
        extra={**ctx, "raw_args": args, "will_try_fallback_target": has_next_target},
      )
      last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_invalid_tool")
      if has_next_target:
        logger.warning(
          "LLM1 invalid tool args; trying fallback target",
          extra={
            **ctx,
            "next_provider": targets[idx + 1].name,
          },
        )
        continue
      return last_failure

    _log_llm1_decision(
      decision,
      ctx=ctx,
      elapsed_ms=elapsed_ms,
      source="tool_call",
    )
    return decision

  return last_failure or LLM1Decision(should_response=False, confidence=10, reason="llm1_exception")
