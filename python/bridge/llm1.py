# File: python/bridge/llm1.py
from __future__ import annotations

import json
import os
import time
import logging
from typing import Iterable, Optional

import httpx
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


def build_llm1_prompt(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  history_limit: int,
  message_max_chars: int,
  current_media_parts: Optional[list[dict]] = None,
  current_media_notes: Optional[list[str]] = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
):
  history_list = list(history)[-history_limit:]
  prompt_history = [_truncate_message(msg, message_max_chars) for msg in history_list]
  current_prompt_msg = _truncate_message(current, message_max_chars)
  hist_text = format_history(prompt_history) or "(no history)"
  current_line = format_history([current_prompt_msg])
  group_text = _group_description_block(group_description)
  current_content: str | list[dict] = f"Current messages:\n{current_line}\n"
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
The tool must include all arguments: should_response (true/false), confidence (0-100), reason (2-8 words). You will be given up to {_llm1_history_limit()} last messages. Every message is capped at {_llm1_message_max_chars()} characters max.

## Know When to Speak!
In group chats where you receive every message, be smart about when to contribute:
Respond when:
- Directly mentioned or asked a question. If someone mentioned your name, it most likely means you need to respond.
- You can add genuine value (info, insight, help).
- Something witty/funny fits naturally.
- Correcting important misinformation.
- Someone needs help or clarification.
- Sometimes it's okay to respond even if you're not mentioned or asked a question.
- Someone reply to your chat.

Stay silent when:
- It’s just casual banter between humans.
- Someone already answered the question.
- Adding a message would interrupt the vibe.

The human rule: Humans in group chats don’t respond to every single message. Neither should you.
Quality > quantity. If you wouldn’t send it in a real group chat with friends, don’t send it.
Participate, don’t dominate.
If you haven’t sent a message in the last 15–20 messages, it’s fine to participate without context.
Note: The chat will be referred to as "LLM".

## Burst messages
When group chat is active, you may get a burst of messages. Please consider every single message in the burst. Sometimes when it's super busy, burst message get sent to older messages.

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
    {"role": "user", "content": f"Older messages:\n{hist_text}"},
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


def _chat_completions_url() -> str | None:
  endpoint = os.getenv("LLM1_ENDPOINT")
  if not endpoint:
    return None
  trimmed = endpoint.rstrip("/")
  if trimmed.endswith("/chat/completions"):
    return trimmed
  return f"{trimmed}/chat/completions"


def _llm1_ctx(current: WhatsAppMessage, *, model: str, url: str) -> dict:
  return {
    "chat_id": getattr(current, "sender", None),
    "message_id": getattr(current, "message_id", None) or getattr(current, "id", None),
    "model": model,
    "endpoint": url,
  }


def _resp_meta(resp: httpx.Response | None) -> dict:
  if resp is None:
    return {"status_code": None, "headers": None}

  hdrs: dict[str, str] = {}
  try:
    for k, v in resp.headers.items():
      lk = k.lower()
      # keep only debugging-relevant headers; avoid dumping everything
      if lk.startswith("x-") or lk in ("cf-ray", "cf-cache-status", "content-type"):
        hdrs[k] = v
  except Exception:
    hdrs = {}

  return {
    "status_code": resp.status_code,
    "headers": hdrs,
  }


def _resp_body_preview(resp: httpx.Response | None, limit: int = 800) -> dict:
  if resp is None:
    return {"body_preview": None, "body_type": None}

  try:
    data = resp.json()
    return {"body_type": "json", "body_preview": trunc(dump_json(data), limit)}
  except Exception:
    try:
      return {"body_type": "text", "body_preview": trunc(resp.text, limit)}
    except Exception:
      return {"body_type": "unknown", "body_preview": None}


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
  client: Optional[httpx.AsyncClient] = None,
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
    group_description=group_description,
    prompt_override=prompt_override,
  )
  model_name = os.getenv("LLM1_MODEL", "gpt-4o-mini")
  api_key = os.getenv("LLM1_API_KEY") or os.getenv("OPENAI_API_KEY")
  url = _chat_completions_url()
  if not url:
    logger.debug("LLM1 endpoint missing after normalization; defaulting to skip")
    return LLM1Decision(should_response=False, confidence=10, reason="llm1_missing_url")

  close_client = False
  if client is None:
    client = httpx.AsyncClient(timeout=timeout)
    close_client = True

  resp: httpx.Response | None = None
  t0 = time.perf_counter()
  ctx = _llm1_ctx(current, model=model_name, url=url)

  try:
    prompt_text = "\n".join(
      [_content_to_text(m.get("content", "")) for m in prompt if isinstance(m, dict)]
    )

    payload = {
      "model": model_name,
      "messages": prompt,
      "tools": [LLM1_TOOL],
      "tool_choice": {
        "type": "function",
        "function": {"name": LLM1_TOOL["function"]["name"]},
      },
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
      headers["Authorization"] = f"Bearer {api_key}"

    if env_flag("BRIDGE_LOG_PROMPT_FULL"):
      log_payload = dict(payload)
      log_payload["messages"] = _redact_messages_for_log(prompt)
      logger.info(
        "LLM1 prompt full",
        extra={
          **ctx,
          "history_limit": history_limit,
          "history_used": len(prompt_history),
          "message_max_chars": message_max_chars,
          "base_url": _chat_base_url(),
          "media_parts": len(current_media_parts),
          "request_payload": log_payload,
        },
      )

    logger.debug(
      "LLM1 request start",
      extra={
        **ctx,
        "history_limit": history_limit,
        "history_used": len(prompt_history),
        "message_max_chars": message_max_chars,
        "timeout_s": timeout,
        "prompt_chars": len(prompt_text),
        "prompt_preview": trunc(prompt_text, 300),
        "media_parts": len(current_media_parts),
        "base_url": _chat_base_url(),
        "tool_name": LLM1_TOOL["function"]["name"],
      },
    )

    try:
      resp = await client.post(url, json=payload, headers=headers)
      elapsed_ms = int((time.perf_counter() - t0) * 1000)

      logger.debug(
        "LLM1 response received",
        extra={
          **ctx,
          "elapsed_ms": elapsed_ms,
          **_resp_meta(resp),
        },
      )

      resp.raise_for_status()

    except httpx.HTTPStatusError as err:
      elapsed_ms = int((time.perf_counter() - t0) * 1000)
      logger.error(
        "LLM1 HTTP error; defaulting to skip",
        exc_info=err,
        extra={
          **ctx,
          "elapsed_ms": elapsed_ms,
          **_resp_meta(resp),
          **_resp_body_preview(resp, limit=900),
        },
      )
      return LLM1Decision(should_response=False, confidence=10, reason="llm1_http_error")

    except httpx.RequestError as err:
      elapsed_ms = int((time.perf_counter() - t0) * 1000)
      logger.error(
        "LLM1 network error; defaulting to skip",
        exc_info=err,
        extra={
          **ctx,
          "elapsed_ms": elapsed_ms,
          "error_type": type(err).__name__,
        },
      )
      return LLM1Decision(should_response=False, confidence=10, reason="llm1_unreachable")

    except Exception as err:
      elapsed_ms = int((time.perf_counter() - t0) * 1000)
      logger.error(
        "LLM1 unexpected error; defaulting to skip",
        exc_info=err,
        extra={**ctx, "elapsed_ms": elapsed_ms},
      )
      return LLM1Decision(should_response=False, confidence=10, reason="llm1_exception")

    # Parse JSON
    try:
      resp_json = resp.json() if resp is not None else {}
    except Exception as err:
      logger.warning(
        "LLM1 response not JSON; defaulting to skip",
        exc_info=err,
        extra={
          **ctx,
          **_resp_meta(resp),
          **_resp_body_preview(resp, limit=600),
        },
      )
      return LLM1Decision(should_response=False, confidence=10, reason="llm1_bad_json")

    provider = resp_json.get("provider")
    routed_model = resp_json.get("model")
    usage = resp_json.get("usage")
    err_obj = resp_json.get("error")

    logger.debug(
      "LLM1 response summary",
      extra={
        **ctx,
        "provider": provider,
        "routed_model": routed_model,
        "usage": usage,
        "error": err_obj,
      },
    )

    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(
        "LLM1 raw response",
        extra={
          **ctx,
          "message_dump": dump_json(resp_json),
          "content_preview": trunc(dump_json(resp_json), 600),
        },
      )

    choices = resp_json.get("choices") or []
    if not choices:
      logger.warning(
        "LLM1 response missing choices; defaulting to skip",
        extra={**ctx, "provider": provider, "routed_model": routed_model},
      )
      return LLM1Decision(should_response=False, confidence=10, reason="llm1_no_choices")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    tool_calls = (message or {}).get("tool_calls") if isinstance(message, dict) else None
    tool_calls = tool_calls or []

    if not tool_calls:
      logger.warning(
        "LLM1 response missing tool call; defaulting to skip",
        extra={
          **ctx,
          "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
          "provider": provider,
          "routed_model": routed_model,
        },
      )
      return LLM1Decision(should_response=False, confidence=10, reason="llm1_no_tool")

    tool_name = LLM1_TOOL["function"]["name"]
    tool_call = next(
      (
        tc
        for tc in tool_calls
        if isinstance(tc, dict)
        and isinstance(tc.get("function"), dict)
        and tc["function"].get("name") == tool_name
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

  finally:
    if close_client:
      await client.aclose()
