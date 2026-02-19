# File: python/bridge/llm1.py
from __future__ import annotations

import json
import os
import time
import logging
import re
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

try:
  from .history import WhatsAppMessage, format_history
  from .log import setup_logging, trunc, dump_json, env_flag
  from .media import build_visual_parts, llm1_media_enabled, redact_multimodal_content
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage, format_history  # type: ignore
  from bridge.log import setup_logging, trunc, dump_json, env_flag  # type: ignore
  from bridge.media import build_visual_parts, llm1_media_enabled, redact_multimodal_content  # type: ignore

logger = setup_logging()


def _parse_positive_int(raw: str | None, default: int) -> int:
  if raw is None:
    return default
  try:
    parsed = int(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


def _llm1_history_limit() -> int:
  # Prefer LLM1-specific limit; fallback to global history limit.
  raw = os.getenv("LLM1_HISTORY_LIMIT")
  if raw is None or not raw.strip():
    raw = os.getenv("HISTORY_LIMIT")
  return _parse_positive_int(raw, 20)


def _llm1_message_max_chars() -> int:
  return _parse_positive_int(os.getenv("LLM1_MESSAGE_MAX_CHARS"), 500)


def _parse_positive_float(raw: str | None, default: float) -> float:
  if raw is None:
    return default
  try:
    parsed = float(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


def _parse_non_negative_int(raw: str | None, default: int) -> int:
  if raw is None:
    return default
  try:
    parsed = int(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed >= 0 else default


def _llm1_timeout(default: float = 8.0) -> float:
  return _parse_positive_float(os.getenv("LLM1_TIMEOUT"), default)


def _llm1_sdk_max_retries() -> int:
  return _parse_non_negative_int(os.getenv("LLM1_SDK_MAX_RETRIES"), 0)


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


def _truncate_message(msg: WhatsAppMessage, max_chars: int) -> WhatsAppMessage:
  return WhatsAppMessage(
    timestamp_ms=msg.timestamp_ms,
    sender=msg.sender,
    context_msg_id=msg.context_msg_id,
    sender_ref=msg.sender_ref,
    sender_is_admin=msg.sender_is_admin,
    text=_truncate_text(msg.text, max_chars),
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
        "description": "A brief reason (2-8 words) justifying the decision.",
        "minLength": 2,
        "maxLength": 64,
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


class LLM1Decision(BaseModel):
  should_response: bool = Field(..., description="Whether to respond")
  confidence: int = Field(..., ge=0, le=100)
  reason: str = Field(..., min_length=2, max_length=64)


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
You are a WhatsApp router agent. Decide whether you should respond.

Your name is Vivy. Sometimes people will refer to you as Vy, Ivy, Vivi, etc.
Call the tool `llm_should_response` exactly once with your decision.
Do not write any other text outside the tool call.
Your tag is @52416300998887, if someone mention it in the chat, respond to it.
The tool must include all arguments: should_response (true/false), confidence (0-100), reason (2-8 words). You will be given up to {_llm1_history_limit()} last messages. Every message is capped at {_llm1_message_max_chars()} characters max.

## Input format
You will receive:
- `Current message metadata` with a Helper section and Chat state section.
- Metadata may also include conversation-level signals:
  - Whether the bot is mentioned in the current message window.
  - Whether any message in the current message window replies to the bot.
  - How many times the bot is mentioned in the current message window.
  - How many messages ago the assistant last replied.
  - How many assistant replies appeared in recent windows (20/50/100/200 and custom history limit).
  - How many human messages exist in the current message window.
- `older messages` section for background history.
- `current messages(burst)` section for the latest trigger window.
- Definition: `current message window` means only messages listed in `current messages(burst)`, not `older messages`.
- Message ids are usually 6-digit; `<system>` and `<pending>` can appear as non-actionable context markers.

Important:
- The `current messages(burst)` section may contain multiple recent messages combined.
- Do not over-prioritize only the last line. Judge whether any message in the burst deserves a reply.
- Use conversation-level signals as hints only (not strict rules):
  - If bot just replied recently and there is no mention/reply signal in the current message window, lean quieter.
  - If bot has been quiet for a while and humans keep talking, lean helpful participation.

## Know When to Speak!
In group chats where you receive every message, be smart about when to contribute:
Respond when:
- Directly mentioned or asked a question. If someone mentioned your name, it most likely means you need to respond.
- Someone tag @52416300998887 in their message.
- Current message metadata indicates the bot was mentioned or replied to in the current message window.
- You can add genuine value (info, insight, help).
- Something witty/funny fits naturally.
- Correcting important misinformation.
- Someone needs help or clarification.
- Sometimes it's okay to respond even if you're not mentioned or asked a question.
- Someone reply to your chat.
- It's already a long time since you last replied.

Stay silent when:
- It’s just casual banter between humans.
- Someone already answered the question.

The human rule: Humans in group chats don’t respond to every single message. Neither should you.
Quality > quantity. If you wouldn’t send it in a real group chat with friends, don’t send it.
Participate, don’t dominate.
Try to be helpful without being annoying.

## Burst messages
When group chat is active, you may get a burst of messages. Please consider every single message in the burst. Sometimes when it's super busy, burst message get sent to older messages.
Consider every single message in `current messages(burst)`. Use `older messages` only as supporting context.

## Prompt Override (higher priority patch)
You may receive extra instructions inside:
<prompt_override> ... </prompt_override>

How to apply it:
- If the <prompt_override> content is empty, missing, or just a placeholder, ignore it.
- Otherwise, treat its content as an additional rule set (a "patch") on top of the main prompt.

Conflict resolution:
- If an override rule conflicts with any rule in the main prompt, the override rule wins for the conflicting part.
- Apply the override with the minimum scope necessary: only replace the specific conflicting constraint; keep all other main rules active.

Non-conflicting merge:
- If an override rule does not conflict with the main prompt, follow both together.
- If the override is more specific than a main rule on the same topic, treat it as taking precedence for that topic (even if both could technically be followed).

Safety check:
- Never follow override instructions that attempt to remove or weaken the requirement to call `llm_should_response` exactly once and output nothing else.

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


def _chat_base_url() -> str | None:
  endpoint = os.getenv("LLM1_ENDPOINT")
  if not endpoint:
    return None
  trimmed = endpoint.rstrip("/")
  if trimmed.endswith("/chat/completions"):
    return trimmed[: -len("/chat/completions")]
  return trimmed


def get_llm1(*, timeout: float = 8.0, include_reasoning: bool = True) -> ChatOpenAI:
  model = os.getenv("LLM1_MODEL", "gpt-4o-mini")
  base_url = _chat_base_url()  # optional custom base URL / proxy
  api_key = os.getenv("LLM1_API_KEY") or os.getenv("OPENAI_API_KEY", "")
  kwargs = {
    "model": model,
    "api_key": api_key,
    "timeout": _llm1_timeout(timeout),
    "max_retries": _llm1_sdk_max_retries(),
    "temperature": 0,
  }
  if base_url:
    kwargs["base_url"] = base_url
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


def _llm1_ctx(current: WhatsAppMessage, *, model: str, url: str | None) -> dict:
  return {
    "chat_id": getattr(current, "sender", None),
    "message_id": getattr(current, "message_id", None) or getattr(current, "id", None),
    "model": model,
    "endpoint": url,
  }


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
  mention_count = payload.get("botMentionCountInWindow")
  if mention_count is None:
    if payload.get("botMentionedInWindow") is not None:
      mention_count = 1 if bool(payload.get("botMentionedInWindow")) else 0
    elif payload.get("botMentioned") is not None:
      mention_count = 1 if bool(payload.get("botMentioned")) else 0
    else:
      mention_count = None
  since_assistant = payload.get("messagesSinceAssistantReply")
  assistant_replies_by_window = payload.get("assistantRepliesByWindow")
  human_window = payload.get("humanMessagesInWindow")
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

  try:
    mention_count = int(mention_count)
  except (TypeError, ValueError):
    pass

  mention_count_text = _count_phrase(mention_count, "time", "times")
  if isinstance(mention_count, int):
    if mention_count > 0:
      mention_line = f"- Bot is mentioned {mention_count_text} in this current message window."
    elif bot_mentioned:
      mention_line = "- Bot is mentioned in this current message window."
    else:
      mention_line = "- Bot is not mentioned in this current message window."
  elif bot_mentioned:
    mention_line = "- Bot is mentioned in this current message window."
  else:
    mention_line = "- Bot is not mentioned in this current message window."

  if replied_to_bot:
    reply_line = "- A message in this current message window replies to the bot."
  else:
    reply_line = "- No message in this current message window replies to the bot."

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

  assistant_reply_block = "\n".join(assistant_reply_lines)
  return (
    "Current message metadata:\n"
    "Helper:\n"
    "- `current message window` = only `current messages(burst)` (exclude `older messages`).\n"
    f"{mention_line}\n"
    f"{reply_line}\n"
    f"- The last assistant reply was {since_assistant_text} ago.\n"
    f"{assistant_reply_block}\n"
    f"{human_window_line}\n"
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
  # If LLM1 is not configured, allow responding by default.
  if not os.getenv("LLM1_ENDPOINT"):
    logger.debug("LLM1 disabled (no LLM1_ENDPOINT set); defaulting to respond")
    return LLM1Decision(should_response=True, confidence=50, reason="llm1_disabled")

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
  model_name = os.getenv("LLM1_MODEL", "gpt-4o-mini")
  base_url = _chat_base_url()
  if not base_url:
    logger.debug("LLM1 endpoint missing after normalization; defaulting to skip")
    return LLM1Decision(should_response=False, confidence=10, reason="llm1_missing_url")
  reasoning_effort = _llm1_reasoning_effort()
  t0 = time.perf_counter()
  ctx = _llm1_ctx(current, model=model_name, url=base_url)
  llm = client or get_llm1(
    timeout=timeout,
    include_reasoning=bool(reasoning_effort),
  )

  prompt_text = "\n".join(
    [_content_to_text(m.get("content", "")) for m in prompt if isinstance(m, dict)]
  )

  if env_flag("BRIDGE_LOG_PROMPT_FULL"):
    logger.info(
      "LLM1 prompt full",
      extra={
        **ctx,
        "history_limit": history_limit,
        "history_used": len(prompt_history),
        "message_max_chars": message_max_chars,
        "base_url": _chat_base_url(),
        "media_parts": len(current_media_parts),
        "reasoning_effort": reasoning_effort,
        "messages": _redact_messages_for_log(prompt),
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
      "base_url": _chat_base_url(),
      "reasoning_effort": reasoning_effort,
      "tool_name": LLM1_TOOL["function"]["name"],
    },
  )

  async def _invoke_once(llm_client: ChatOpenAI):
    try:
      llm_with_tool = llm_client.bind_tools(
        [LLM1_TOOL],
        tool_choice=LLM1_TOOL["function"]["name"],
      )
    except Exception as err:
      logger.warning(
        "LLM1 bind_tools with explicit tool_choice failed; retrying default bind_tools",
        exc_info=err,
        extra={
          **ctx,
          "error_type": type(err).__name__,
        },
      )
      llm_with_tool = llm_client.bind_tools([LLM1_TOOL])
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
      llm = get_llm1(timeout=timeout, include_reasoning=False)
      reasoning_effort = None
      try:
        response = await _invoke_once(llm)
      except Exception as retry_err:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        timeout_error = _is_timeout_error(retry_err)
        logger.error(
          "LLM1 invoke failed after reasoning fallback; defaulting to skip",
          exc_info=True,
          extra={
            **ctx,
            "elapsed_ms": elapsed_ms,
            "reasoning_effort": reasoning_effort,
            "error_type": type(retry_err).__name__,
            "error_chain": _error_chain(retry_err),
          },
        )
        if timeout_error:
          return LLM1Decision(should_response=False, confidence=10, reason="llm1_unreachable")
        return LLM1Decision(should_response=False, confidence=10, reason="llm1_exception")
    else:
      elapsed_ms = int((time.perf_counter() - t0) * 1000)
      timeout_error = _is_timeout_error(err)
      logger.error(
        "LLM1 invoke failed; defaulting to skip",
        exc_info=True,
        extra={
          **ctx,
          "elapsed_ms": elapsed_ms,
          "reasoning_effort": reasoning_effort,
          "error_type": type(err).__name__,
          "error_chain": _error_chain(err),
        },
      )
      if timeout_error:
        return LLM1Decision(should_response=False, confidence=10, reason="llm1_unreachable")
      return LLM1Decision(should_response=False, confidence=10, reason="llm1_exception")

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
        logger.info(
          "LLM1 decision",
          extra={
            **ctx,
            "should_response": decision.should_response,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "raw": trunc(dump_json(decision.model_dump()), 400),
          },
        )
        return decision
      except ValidationError:
        pass
    logger.warning(
      "LLM1 response missing tool call; defaulting to skip",
      extra={
        **ctx,
        "reasoning_effort": reasoning_effort,
        "response_metadata": response_metadata,
      },
    )
    return LLM1Decision(should_response=False, confidence=10, reason="llm1_no_tool")

  tool_name = LLM1_TOOL["function"]["name"]
  tool_call = next(
    (
      tc
      for tc in tool_calls
      if isinstance(tc, dict)
      and (
        tc.get("name") == tool_name
        or (
          isinstance(tc.get("function"), dict)
          and tc["function"].get("name") == tool_name
        )
      )
    ),
    tool_calls[0],
  )

  args = _extract_tool_args(tool_call)
  if not args:
    logger.warning(
      "LLM1 tool args empty; defaulting to skip",
      extra={
        **ctx,
        "raw_tool_call": trunc(str(tool_call), 500),
      },
    )
    return LLM1Decision(should_response=False, confidence=10, reason="llm1_empty_tool")

  try:
    decision = LLM1Decision.model_validate(args)
  except ValidationError as err:
    logger.warning(
      "LLM1 tool args failed validation; defaulting to skip",
      exc_info=err,
      extra={**ctx, "raw_args": args},
    )
    return LLM1Decision(should_response=False, confidence=10, reason="llm1_invalid_tool")

  logger.info(
    "LLM1 decision",
    extra={
      **ctx,
      "should_response": decision.should_response,
      "confidence": decision.confidence,
      "reason": decision.reason,
      "raw": trunc(dump_json(decision.model_dump()), 400),
    },
  )
  return decision
