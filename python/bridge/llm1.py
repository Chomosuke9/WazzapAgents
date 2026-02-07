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
  from .log import setup_logging, trunc, dump_json
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage, format_history  # type: ignore
  from bridge.log import setup_logging, trunc, dump_json  # type: ignore

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
    text=_truncate_text(msg.text, max_chars),
    media=msg.media,
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


def build_llm1_prompt(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  history_limit: int,
  message_max_chars: int,
):
  history_list = list(history)[-history_limit:]
  prompt_history = [_truncate_message(msg, message_max_chars) for msg in history_list]
  current_prompt_msg = _truncate_message(current, message_max_chars)
  hist_text = format_history(prompt_history) or "(no history)"
  current_line = format_history([current_prompt_msg])
  return [
    {
      "role": "system",
      "content": f"""
You are a WhatsApp triage agent. Decide if we should respond.
Your name is Vivy. Sometimes people will refer you as Vy, Ivy, Vivi, etc.
Call the tool `llm_should_response` exactly once with your decision.
Do not write any other text outside the tool call.
The tool must include all arguments: should_response (true/false), confidence (0-100), reason (2-8 words). You will given up to {_llm1_history_limit()} last messages, Every message are capped at {_llm1_message_max_chars()} characters max

## Know When to Speak!
In group chats where you receive every message, be smart about when to contribute:
Respond when:
- Directly mentioned or asked a question
- You can add genuine value (info, insight, help)
- Something witty/funny fits naturally
- Correcting important misinformation
- Summarizing when asked

Stay silent when:
- It’s just casual banter between humans
- Someone already answered the question
- Your response would just be “yeah” or “nice”
- The conversation is flowing fine without you
- Adding a message would interrupt the vibe

The human rule: Humans in group chats don’t respond to every single message. Neither should you.
Quality > quantity. If you wouldn’t send it in a real group chat with friends, don’t send it.
Participate, don’t dominate.
      """.strip(),
    },
    {"role": "user", "content": f"Older message:\n{hist_text}"},
    {"role": "user", "content": f"Current message:\n{current_line}\n"},
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


async def call_llm1(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  timeout: float = 8.0,
  client: Optional[httpx.AsyncClient] = None,
) -> LLM1Decision:
  # If LLM1 is not configured, allow responding by default.
  if not os.getenv("LLM1_ENDPOINT"):
    logger.debug("LLM1 disabled (no LLM1_ENDPOINT set); defaulting to respond")
    return LLM1Decision(should_response=True, confidence=50, reason="llm1_disabled")

  history_limit = _llm1_history_limit()
  message_max_chars = _llm1_message_max_chars()
  history_list = list(history)
  prompt_history = history_list[-history_limit:]
  prompt = build_llm1_prompt(
    prompt_history,
    current,
    history_limit=history_limit,
    message_max_chars=message_max_chars,
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
    prompt_text = "\n".join([m.get("content", "") for m in prompt if isinstance(m, dict)])

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
