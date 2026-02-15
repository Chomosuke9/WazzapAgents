from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
  from .history import WhatsAppMessage, format_history
  from .log import setup_logging, trunc, dump_json, env_flag
  from .media import build_visual_parts, llm2_media_enabled, redact_multimodal_content
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage, format_history  # type: ignore
  from bridge.log import setup_logging, trunc, dump_json, env_flag  # type: ignore
  from bridge.media import build_visual_parts, llm2_media_enabled, redact_multimodal_content  # type: ignore

logger = setup_logging()
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "systemprompt.txt"
_SYSTEM_PROMPT_CACHE: str | None = None


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


def _llm2_timeout() -> float:
  return _parse_positive_float(os.getenv("LLM2_TIMEOUT"), 20.0)


def _llm2_retry_max() -> int:
  return _parse_non_negative_int(os.getenv("LLM2_RETRY_MAX"), 0)


def _llm2_retry_backoff_seconds() -> float:
  return _parse_positive_float(os.getenv("LLM2_RETRY_BACKOFF_SECONDS"), 0.8)


def _llm2_sdk_max_retries() -> int:
  return _parse_non_negative_int(os.getenv("LLM2_SDK_MAX_RETRIES"), 0)


def _llm2_reasoning_effort() -> str | None:
  raw = os.getenv("LLM2_REASONING_EFFORT")
  if raw is None:
    return "medium"
  cleaned = raw.strip().lower()
  if not cleaned:
    return None
  return cleaned


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


def _load_system_prompt() -> str:
  global _SYSTEM_PROMPT_CACHE
  if _SYSTEM_PROMPT_CACHE is not None:
    return _SYSTEM_PROMPT_CACHE
  text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
  _SYSTEM_PROMPT_CACHE = text
  return _SYSTEM_PROMPT_CACHE


def _render_system_prompt(
  base_system: str,
  *,
  prompt_override: str | None = None,
) -> str:
  overide_text = (prompt_override or "").strip()
  return (
    base_system
    .replace("{{prompt_override}}", overide_text)
    .replace("{{ prompt_override }}", overide_text)
  )


def _group_description_block(group_description: str | None) -> str:
  cleaned = (group_description or "").strip()
  if cleaned:
    return cleaned
  return "(none)"


def _normalize_chat_type(chat_type: str | None) -> str:
  lowered = (chat_type or "").strip().lower()
  if lowered in {"private", "group"}:
    return lowered
  return "private"


def _chat_state_header(chat_type: str, bot_is_admin: bool, bot_is_super_admin: bool) -> str:
  normalized_type = _normalize_chat_type(chat_type)
  return (
    f"CHAT_TYPE: {normalized_type}\n"
    f"BOT_ROLE: botIsAdmin={'true' if bot_is_admin else 'false'} "
    f"botIsSuperAdmin={'true' if bot_is_super_admin else 'false'}"
  )


def get_llm2() -> ChatOpenAI:
  model = os.getenv("LLM2_MODEL", "gpt-4.1")
  temperature = float(os.getenv("LLM2_TEMPERATURE", "0.5"))
  timeout = _llm2_timeout()
  max_retries = _llm2_sdk_max_retries()
  reasoning_effort = _llm2_reasoning_effort()
  base_url = os.getenv("LLM2_ENDPOINT")  # optional: custom base URL / proxy
  api_key = os.getenv("LLM2_API_KEY", "")  # optional: override OPENAI_API_KEY
  kwargs = {
    "model": model,
    "temperature": temperature,
    "base_url": base_url,
    "api_key": api_key,
    "timeout": timeout,
    "max_retries": max_retries,
  }
  if reasoning_effort:
    kwargs["reasoning_effort"] = reasoning_effort
  return ChatOpenAI(
    **kwargs,
  )


async def generate_reply(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  system: str | None = None,
  tools: Optional[list] = None,
  current_payload: dict | None = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
  chat_type: str | None = None,
  bot_is_admin: bool = False,
  bot_is_super_admin: bool = False,
):
  llm = get_llm2()
  model_name = os.getenv("LLM2_MODEL", "gpt-4.1")
  timeout_s = _llm2_timeout()
  retry_max = _llm2_retry_max()
  retry_backoff_s = _llm2_retry_backoff_seconds()
  sdk_max_retries = _llm2_sdk_max_retries()
  reasoning_effort = _llm2_reasoning_effort()
  base_system = (system or _load_system_prompt()).strip()
  rendered_system = _render_system_prompt(
    base_system,
    prompt_override=prompt_override,
  )
  history_list = list(history)
  hist_text = format_history(history_list) or "(no history)"
  current_line = format_history([current])
  group_text = _group_description_block(group_description)
  chat_state_text = _chat_state_header(chat_type or "private", bot_is_admin, bot_is_super_admin)
  older_messages_content = f"Chat state:\n{chat_state_text}\n\nOlder messages:\n{hist_text}"
  current_content_text = f"Chat state:\n{chat_state_text}\n\nCurrent messages:\n{current_line}"
  media_parts: list[dict] = []
  media_notes: list[str] = []
  if llm2_media_enabled():
    media_parts, media_notes = build_visual_parts(current_payload)
  if media_notes:
    current_content_text += "\n\nVisual attachments:\n" + "\n".join(
      f"- {note}" for note in media_notes
    )

  current_content: str | list[dict]
  if media_parts:
    current_content = [{"type": "text", "text": current_content_text}]
    current_content.extend(media_parts)
  else:
    current_content = current_content_text

  msgs = [SystemMessage(content=rendered_system)]
  msgs.append(HumanMessage(content=f"Group description:\n{group_text}"))
  msgs.append(HumanMessage(content=older_messages_content))
  msgs.append(HumanMessage(content=current_content))
  if tools:
    llm = llm.bind_tools(tools)
  if env_flag("BRIDGE_LOG_PROMPT_FULL"):
    logger.info(
      "LLM2 prompt full",
      extra={
        "chat_id": current.sender,
        "model": os.getenv("LLM2_MODEL", "gpt-4.1"),
        "messages": [
          {"role": "system", "content": rendered_system},
          {"role": "user", "content": f"Group description:\n{group_text}"},
          {"role": "user", "content": older_messages_content},
          {"role": "user", "content": redact_multimodal_content(current_content)},
        ],
      },
    )
  logger.debug(
    "LLM2 invoke",
    extra={
      "chat_id": current.sender,
      "history_len": len(history_list),
      "system_chars": len(rendered_system),
      "prompt_preview": trunc(
        group_text + '\n' + chat_state_text + '\n' + hist_text + '\n' + current_line + f"\n[visual_attachments={len(media_parts)}]",
        800,
      ),
      "model": model_name,
      "timeout_s": timeout_s,
      "retry_max": retry_max,
      "retry_backoff_s": retry_backoff_s,
      "sdk_max_retries": sdk_max_retries,
      "reasoning_effort": reasoning_effort,
    },
  )

  async def _invoke_with_retry(prompt_msgs, *, mode: str):
    attempts_total = retry_max + 1
    last_failure_kind: str | None = None
    for attempt in range(1, attempts_total + 1):
      started = time.perf_counter()
      try:
        response = await llm.ainvoke(prompt_msgs)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
          "LLM2 invoke success",
          extra={
            "chat_id": current.sender,
            "model": model_name,
            "mode": mode,
            "attempt": attempt,
            "attempts_total": attempts_total,
            "elapsed_ms": elapsed_ms,
            "timeout_s": timeout_s,
            "sdk_max_retries": sdk_max_retries,
          },
        )
        return response, None
      except Exception as err:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        timeout_error = _is_timeout_error(err)
        last_failure_kind = "timeout" if timeout_error else "error"
        can_retry = timeout_error and attempt < attempts_total
        logger.warning(
          "LLM2 invoke failed",
          exc_info=not can_retry,
          extra={
            "chat_id": current.sender,
            "model": model_name,
            "mode": mode,
            "attempt": attempt,
            "attempts_total": attempts_total,
            "elapsed_ms": elapsed_ms,
            "timeout_s": timeout_s,
            "retry_backoff_s": retry_backoff_s,
            "will_retry": can_retry,
            "error_type": type(err).__name__,
            "error_chain": _error_chain(err),
            "sdk_max_retries": sdk_max_retries,
          },
        )
        if not can_retry:
          return None, last_failure_kind
        await asyncio.sleep(retry_backoff_s * attempt)
    return None, last_failure_kind

  result, failure_kind = await _invoke_with_retry(msgs, mode="multimodal" if media_parts else "text")
  if result is None and media_parts:
    if failure_kind == "timeout":
      logger.warning(
        "LLM2 multimodal timeout; skipping text-only fallback",
        extra={
          "chat_id": current.sender,
          "model": model_name,
          "media_parts": len(media_parts),
          "timeout_s": timeout_s,
          "retry_max": retry_max,
          "sdk_max_retries": sdk_max_retries,
        },
      )
      return None

    logger.warning(
      "LLM2 multimodal failed; retrying text-only prompt",
      extra={
        "chat_id": current.sender,
        "model": model_name,
        "media_parts": len(media_parts),
      },
    )
    result, _ = await _invoke_with_retry(
      [
        SystemMessage(content=rendered_system),
        HumanMessage(content=f"Group description:\n{group_text}"),
        HumanMessage(content=older_messages_content),
        HumanMessage(content=current_content_text),
      ],
      mode="text_fallback",
    )
  if result is None:
    return None

  logger.debug(
    "LLM2 result",
    extra={
      "chat_id": current.sender,
      "reply_preview": trunc(getattr(result, 'content', ''), 800),
      "raw": dump_json(getattr(result, "model_dump", lambda: str(result))()),
    },
  )
  return result
